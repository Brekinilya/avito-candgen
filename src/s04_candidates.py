"""
Кандидаты и признаки для ранкера.

Кандидаты запроса - объединение нескольких списков:
  бит 0   BM25 + log P(локация), top-300
  бит 1   BM25 без локации, top-50
  бит 2   объявления, которые выбирали по похожим запросам в истории, top-50
  бит 3+2m  энкодер m + log P(локация), top-300
  бит 4+2m  энкодер m без локации, top-50 (онлайн-услуги, доставка и т.п.)
Номера источников сохраняются как признаки src*.

Скоры считаются пачками запросов плотными матрицами [пачка x корпус] на GPU.
Видимость: добавленные из валидации объявления и копии-конкуренты (s03c) видны только
своему фолду, бенчмарку виден ровно benchmark_items. Копия берёт тексты и эмбеддинг
исходного объявления (src_row), а локацию и координаты - свои; в выходе у копий row >= N.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from bm25 import BM25Index
from common import WORK, norm_expr
from history import History
from location import LocationModel

K_LOC = 300
K_GLOBAL = 50
K_HIST = 50
CHUNK = 192
W_PARAMS, W_DESC = 0.2, 0.5        # веса полей в сумме BM25, подбирались на валидации
BETA_BM25, BETA_DENSE = 2.0, 0.03  # вес log P(локация) при отборе
N_FOLDS = 7

# служебные слова фильтров поиска («Вид услуги ... Тип услуги ...»)
PARAM_KEYS = ["вид", "услуг", "тип", "рейтинг", "пользовател", "звезд", "выш", "онлайн", "запис"]


def dense_from_csr(m: sp.csr_matrix, dev) -> torch.Tensor:
    m = m.tocoo()
    t = torch.zeros(m.shape, dtype=torch.float32, device=dev)
    if m.nnz:
        t[torch.from_numpy(m.row.astype(np.int64)).to(dev), torch.from_numpy(m.col.astype(np.int64)).to(dev)] = \
            torch.from_numpy(m.data.astype(np.float32)).to(dev)
    return t


def rank_within(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Ранг внутри запроса, 0 - лучший; строки одного запроса должны идти подряд."""
    out = np.empty(len(x), dtype=np.float32)
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    ends = np.r_[starts[1:], len(x)]
    for s, e in zip(starts, ends):
        o = np.argsort(-x[s:e], kind="stable")
        r = np.empty(e - s, dtype=np.float32)
        r[o] = np.arange(e - s)
        out[s:e] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default="e5b_hn,e5s_hn,e5b", help="энкодеры через запятую")
    ap.add_argument("--out", default="cands")
    ap.add_argument("--no_competitors", action="store_true", help="валидация без конкурентов из s03c")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    dnames = args.dense.split(",")

    corpus = pl.read_parquet(WORK / "corpus.parquet")
    # символьные 3-4-граммы заголовка: опечатки, слитное написание («автоподбор» / «подбор авто»)
    cvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, sublinear_tf=True,
                           dtype=np.float32)
    T_char = cvec.fit_transform(corpus.select(norm_expr("item_title_raw"))["item_title_raw"].to_list()).tocsr()
    corpus = corpus.with_columns(
        pl.col("item_description_raw").str.len_chars().alias("dlen"),
        pl.col("item_title_raw").str.len_chars().alias("tlen"),
    ).drop("item_description_raw", "item_infm_params_text")
    N = len(corpus)
    # идём по фолдам: у каждого фолда свой набор копий-конкурентов
    queries = pl.read_parquet(WORK / "queries.parquet").with_row_index("qorder").sort(["fold", "qorder"])
    hist = pl.read_parquet(WORK / "hist.parquet", columns=[
        "search_query", "search_location_id", "item_id", "item_location_id", "item_latitude",
        "item_longitude", "item_microcat_id"]).with_columns(
        pl.col("item_latitude").cast(pl.Float64), pl.col("item_longitude").cast(pl.Float64))
    idx = BM25Index.load()
    lm = LocationModel(hist, corpus)
    item_ids = corpus["item_id"].to_numpy()
    htexts = pl.read_parquet(WORK / "hist_texts.parquet")["search_query"].to_list()
    H = History(hist, item_ids, htexts, np.load(WORK / f"emb_hist_txt_{dnames[0]}.npy"))
    comp = None
    if not args.no_competitors:
        comp = pl.read_parquet(WORK / "competitors.parquet").with_row_index("cid")
        print("competitors:", len(comp))
    print(f"loaded in {time.time() - t0:.0f}s")

    iloc = corpus["item_location_id"].to_numpy()
    uloc = np.unique(iloc if comp is None else np.concatenate([iloc, comp["loc"].to_numpy()]))
    ilat = corpus["item_latitude"].fill_null(np.nan).to_numpy()
    ilon = corpus["item_longitude"].fill_null(np.nan).to_numpy()
    imc_idx = np.array([H.mc_idx.get(int(m), -1) for m in corpus["item_microcat_id"].to_numpy()])
    # Категорию объявления в признаки не берём: в train выбирали только услуги, и ранкер
    # выучил бы «не услуга - нерелевантно», а у запросов бенчмарка с search_category=0
    # подходящими бывают и товары. Популярность объявления в истории тоже не берём: в
    # валидации выбранные объявления из того же периода, что и история, и признак там
    # сильнее, чем может быть на бенчмарке.
    item_feats = {
        "i_rating": corpus["item_rating"].fill_null(-1).to_numpy().astype(np.float32),
        "i_reviews": np.log1p(corpus["item_rating_reviews_count"].fill_null(0).to_numpy()).astype(np.float32),
        "i_price": np.log1p(corpus["item_price"].fill_null(0).clip(0, 1e9).to_numpy()).astype(np.float32),
        "i_phone_hidden": corpus["item_is_phone_hidden"].cast(pl.Float32).to_numpy(),
        "i_msg_forbidden": corpus["item_is_message_forbidden"].cast(pl.Float32).to_numpy(),
        "i_dlen": np.log1p(corpus["dlen"].fill_null(0).to_numpy()).astype(np.float32),
        "i_tlen": corpus["tlen"].fill_null(0).to_numpy().astype(np.float32),
    }
    in_bench = corpus["in_bench"].to_numpy()
    loc_cnt_map = dict(zip(*np.unique(iloc[in_bench], return_counts=True)))

    # vis[f] - что видно фолду f, vis[7] - бенчмарку
    vis = np.repeat(in_bench[None, :], N_FOLDS + 1, axis=0)
    for i, fl in enumerate(corpus["folds"].to_list()):
        if fl is not None:
            for f in fl:
                vis[f, i] = True
    vis_t = torch.from_numpy(vis).to(dev)

    I_all = {n: torch.from_numpy(np.load(WORK / f"emb_items_{n}.npy")).to(dev) for n in dnames}
    QE_all = {n: np.load(WORK / f"emb_queries_{n}.npy") for n in dnames}
    QT = np.load(WORK / f"emb_queries_txt_{dnames[0]}.npy")

    # есть ли термин в поле - для доли покрытых слов запроса
    Wb = {f: (idx.W[f] > 0).astype(np.float32).tocsr() for f in ("title", "params", "desc")}
    W_any = (Wb["title"] + Wb["params"] + Wb["desc"]).sign().tocsr()
    key_ids = {idx.vocab.stem2id[k] for k in PARAM_KEYS if k in idx.vocab.stem2id}

    lp_cache: dict[int, np.ndarray] = {}

    def lp_vec(s):
        if s not in lp_cache:
            lp_cache[s] = lm.log_p(s, uloc)
        return lp_cache[s]

    out_parts = []
    cand_recall = []
    NEG = torch.tensor(-1e9, device=dev)
    for fold in sorted(queries["fold"].unique().to_list()):
        qf = queries.filter(pl.col("fold") == fold)
        if comp is not None and fold >= 0:
            cf = comp.filter(pl.col("fold") == fold)
            ext_src = cf["src_row"].to_numpy().astype(np.int64)
            ext_gid = N + cf["cid"].to_numpy().astype(np.int64)
            ext_loc, ext_lat, ext_lon = cf["loc"].to_numpy(), cf["lat"].to_numpy(), cf["lon"].to_numpy()
        else:
            ext_src = np.zeros(0, dtype=np.int64)
            ext_gid = np.zeros(0, dtype=np.int64)
            ext_loc, ext_lat, ext_lon = np.zeros(0, dtype=iloc.dtype), np.zeros(0), np.zeros(0)
        n_ext = len(ext_src)
        # позиции 0..N-1 - корпус, дальше копии фолда
        src_of = np.concatenate([np.arange(N), ext_src])
        gid_of = np.concatenate([np.arange(N), ext_gid])
        iloc_x = np.concatenate([iloc, ext_loc])
        ilat_x = np.concatenate([ilat, ext_lat])
        ilon_x = np.concatenate([ilon, ext_lon])
        iloc_idx_x = torch.from_numpy(np.searchsorted(uloc, iloc_x).astype(np.int64)).to(dev)
        ext_src_t = torch.from_numpy(ext_src).to(dev)

        def ext(M):
            return torch.cat([M, M[:, ext_src_t]], 1) if n_ext else M

        for c0 in range(0, len(qf), CHUNK):
            qc = qf.slice(c0, CHUNK)
            B = len(qc)
            qpos = qc["qorder"].to_numpy()   # строка в queries.parquet и в эмбеддингах
            V = vis_t[torch.full((B,), fold if fold >= 0 else N_FOLDS, device=dev)]
            if n_ext:
                V = torch.cat([V, torch.ones((B, n_ext), dtype=torch.bool, device=dev)], 1)
            slocs = qc["search_location_id"].to_list()

            Q = idx.query_matrix(qc)
            Qu = idx.query_matrix(qc, bigrams=False)
            n_terms = np.asarray(Qu.sum(1)).ravel().astype(np.float32)
            bt = ext(dense_from_csr(idx.score(Q, "title"), dev))
            bp = ext(dense_from_csr(idx.score(Q, "params"), dev))
            bd = ext(dense_from_csr(idx.score(Q, "desc"), dev))
            comb = bt + W_PARAMS * bp + W_DESC * bd
            cov_t = ext(dense_from_csr((Qu @ Wb["title"]).tocsr(), dev))
            cov_any = ext(dense_from_csr((Qu @ W_any).tocsr(), dev))
            # насколько фильтры поиска совпадают с параметрами объявления
            Qp = idx.query_matrix(qc, col="search_infm_params_text", bigrams=False, exclude=key_ids)
            n_pterms = np.asarray(Qp.sum(1)).ravel().astype(np.float32)
            pcov = ext(dense_from_csr((Qp @ Wb["params"]).tocsr(), dev))
            LP = torch.from_numpy(np.stack([lp_vec(s) for s in slocs])).to(dev)[:, iloc_idx_x]

            def topk(S, K):
                v, i = torch.topk(S, K, dim=1)
                return i.cpu().numpy(), (v > -1e8).cpu().numpy()   # у BM25 совпадений может быть меньше K

            top = {0: topk(torch.where(V & (comb > 0), comb + BETA_BM25 * LP, NEG), K_LOC),
                   1: topk(torch.where(V & (comb > 0), comb, NEG), K_GLOBAL)}
            DS, dn_max, dnl_max = {}, {}, {}
            for m, n in enumerate(dnames):
                ds = ext((torch.from_numpy(QE_all[n][qpos]).to(dev) @ I_all[n].T).float())
                s_loc = torch.where(V, ds + BETA_DENSE * LP, NEG)
                top[3 + 2 * m] = topk(s_loc, K_LOC)
                dnl_max[n] = s_loc.max(1).values.cpu().numpy()
                s_glb = torch.where(V, ds, NEG)
                top[4 + 2 * m] = topk(s_glb, K_GLOBAL)
                dn_max[n] = s_glb.max(1).values.cpu().numpy()
                s_loc = s_glb = None   # иначе на 8 ГБ не помещается
                DS[n] = ds

            nb_i, nb_s = H.neighbours(QT[qpos], k=30)
            mc_dist = H.microcat_dist(nb_i, nb_s)
            qtexts = qc["search_query"].to_list()
            vrow = fold if fold >= 0 else N_FOLDS

            rows = []
            for b in range(B):
                s = slocs[b]
                # объявления, выбранные по похожим текстам; в чужой локации с весом 0.3
                hist_score: dict[int, float] = {}
                exact_cnt: dict[int, float] = {}
                exact_same: dict[int, float] = {}
                tq = H.text2idx.get(qtexts[b], -1)
                for j in range(nb_i.shape[1]):
                    t, w = int(nb_i[b, j]), float(nb_s[b, j])
                    if w < 0.80:
                        break
                    for r, s2, c in H.text_items.get(t, ()):
                        if not vis[vrow, r]:
                            continue
                        hist_score[r] = hist_score.get(r, 0.0) + w * c * (1.0 if s2 == s else 0.3)
                        if t == tq:
                            exact_cnt[r] = exact_cnt.get(r, 0) + c
                            if s2 == s:
                                exact_same[r] = exact_same.get(r, 0) + c
                h_top = sorted(hist_score, key=lambda r: -hist_score[r])[:K_HIST]

                src = {}
                for k, (ti, tv) in top.items():
                    for r in ti[b][tv[b]].tolist():
                        src[r] = src.get(r, 0) | (1 << k)
                for r in h_top:
                    src[r] = src.get(r, 0) | (1 << 2)
                cand = np.fromiter(src.keys(), dtype=np.int64)
                srcs = np.fromiter(src.values(), dtype=np.int64)
                rows.append((cand, srcs, hist_score, exact_cnt, exact_same))

            # признаки; cc - позиция в корпусе с копиями, base - исходная строка корпуса
            lens = [len(r[0]) for r in rows]
            bb = np.repeat(np.arange(B), lens)
            cc = np.concatenate([r[0] for r in rows])
            srcs = np.concatenate([r[1] for r in rows])
            base = src_of[cc]
            is_copy = cc >= N
            bb_t, cc_t = torch.from_numpy(bb).to(dev), torch.from_numpy(cc).to(dev)

            def g(M):
                return M[bb_t, cc_t].cpu().numpy()

            f = {
                "bm_t": g(bt), "bm_p": g(bp), "bm_d": g(bd), "bm_comb": g(comb),
                "cov_t": g(cov_t) / np.maximum(n_terms[bb], 1), "cov_any": g(cov_any) / np.maximum(n_terms[bb], 1),
                "pcov": np.where(n_pterms[bb] > 0, g(pcov) / np.maximum(n_pterms[bb], 1), -1),
                "lp": g(LP),
            }
            # скоры относительно лучшего кандидата запроса
            bm_max = torch.where(V, comb, NEG).max(1).values.cpu().numpy()
            f["bm_rel"] = f["bm_comb"] / np.maximum(bm_max[bb], 1e-6)
            for n in dnames:
                f[f"dense_{n}"] = g(DS[n])
                f[f"dense_rel_{n}"] = f[f"dense_{n}"] - dn_max[n][bb]
                f[f"dense_loc_rel_{n}"] = f[f"dense_{n}"] + BETA_DENSE * f["lp"] - dnl_max[n][bb]
                f[f"q_dn_max_{n}"] = dn_max[n][bb]
            if len(dnames) > 1:
                f["dense_loc_rel_mean"] = np.mean([f[f"dense_loc_rel_{n}"] for n in dnames], axis=0)
            Qc_char = cvec.transform(qc.select(norm_expr("search_query"))["search_query"].to_list()).tocsr()
            f["char_t"] = np.asarray(T_char[base].multiply(Qc_char[bb]).sum(1)).ravel()
            f["same_loc"] = (iloc_x[cc] == np.array(slocs)[bb]).astype(np.float32)
            dist = np.empty(len(cc), dtype=np.float32)
            for b in range(B):
                m_ = bb == b
                dist[m_] = lm.dist_km(slocs[b], ilat_x[cc[m_]], ilon_x[cc[m_]])
            f["dist"] = np.log1p(np.nan_to_num(dist, nan=5000.0))
            for k in sorted(top.keys() | {2}):
                f[f"src{k}"] = ((srcs >> k) & 1).astype(np.float32)
            # копия - то же объявление, признаки истории берём у оригинала
            f["hist_knn"] = np.array([rows[b][2].get(r, 0.0) for b, r in zip(bb, base)], dtype=np.float32)
            f["hist_exact"] = np.array([rows[b][3].get(r, 0.0) for b, r in zip(bb, base)], dtype=np.float32)
            f["hist_exact_same"] = np.array([rows[b][4].get(r, 0.0) for b, r in zip(bb, base)], dtype=np.float32)
            mci = imc_idx[base]
            f["mc_aff"] = np.where(mci >= 0, mc_dist[bb, np.maximum(mci, 0)], 0).astype(np.float32)
            for k, v in item_feats.items():
                f[k] = v[base]
            f["q_nterms"] = n_terms[bb]
            f["q_len"] = qc["search_query"].str.len_chars().to_numpy().astype(np.float32)[bb]
            f["q_has_params"] = (qc["search_infm_params_text"].str.len_chars().to_numpy() > 0).astype(np.float32)[bb]
            f["q_seen"] = np.array([H.text2idx.get(t, -1) >= 0 for t in qtexts], dtype=np.float32)[bb]
            f["q_nn_sim"] = nb_s[:, 0][bb]
            f["q_loc_self"] = np.array([lm.self_frac.get(s, -1) for s in slocs], dtype=np.float32)[bb]
            f["q_loc_hist"] = np.log1p(np.array([lm.total.get(s, 0) for s in slocs], dtype=np.float32))[bb]
            f["q_loc_items"] = np.log1p(np.array([loc_cnt_map.get(s, 0) for s in slocs], dtype=np.float32))[bb]
            f["q_bm_max"] = bm_max[bb]

            df = pl.DataFrame({"qid": np.array(qc["qid"].to_list())[bb], "fold": np.full(len(cc), fold, np.int8),
                               "row": gid_of[cc].astype(np.int32),
                               **{k: np.asarray(v, dtype=np.float32) for k, v in f.items()}})
            rel = qc["rel"].to_list()
            lab = np.zeros(len(cc), dtype=np.int8)
            for b in range(B):
                if rel[b] is None:
                    continue
                rs = set(rel[b])
                m_ = np.flatnonzero(bb == b)
                hit = np.array([(not is_copy[i]) and item_ids[cc[i]] in rs for i in m_])
                lab[m_[hit]] = 1
                cand_recall.append((fold, hit.sum() / len(rs)))
            out_parts.append(df.with_columns(pl.Series("label", lab)))
            if (c0 // CHUNK) % 5 == 0:
                cr = np.mean([x[1] for x in cand_recall]) if cand_recall else float("nan")
                print(f"fold {fold} {c0 + B}/{len(qf)} copies {n_ext} cands/q {np.mean(lens):.0f} "
                      f"cand recall {cr:.4f} {time.time() - t0:.0f}s", flush=True)

    res = pl.concat(out_parts).sort(["qid"], maintain_order=True)
    qg = res["qid"].to_numpy()
    for col in ["bm_comb", "hist_knn"] + [f"dense_{n}" for n in dnames]:
        res = res.with_columns(pl.Series(f"rk_{col}", rank_within(res[col].to_numpy(), qg)))
    for n in dnames:
        res = res.with_columns(pl.Series(f"rk_dense_loc_{n}", rank_within(
            (res[f"dense_{n}"] + BETA_DENSE * res["lp"]).to_numpy(), qg)))
    if len(dnames) > 1:
        res = res.with_columns(pl.Series("rk_dense_loc_mean", rank_within(res["dense_loc_rel_mean"].to_numpy(), qg)))
    res = res.with_columns(pl.Series("rk_bm_loc", rank_within(
        (res["bm_comb"] + BETA_BM25 * res["lp"]).to_numpy(), qg)))
    res.write_parquet(WORK / f"{args.out}.parquet")
    cr = np.array(cand_recall)
    print("candidate recall by fold:", {int(k): round(float(cr[cr[:, 0] == k, 1].mean()), 4) for k in np.unique(cr[:, 0])})
    print(f"rows {len(res)}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
