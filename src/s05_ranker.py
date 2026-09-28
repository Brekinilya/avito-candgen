"""
Ранкер LightGBM (LambdaRank) по кандидатам из s04 и итоговый answer.csv.

Валидация: для отчётного фолда e (по умолчанию 0 и 1) обучаемся на фолдах 0..5 без e,
ранняя остановка по фолду 6, Recall@50 считаем на e. Финальная модель учится на всех
фолдах со средним найденным числом деревьев; три сида, ранги предсказаний усредняются.
Запросы без единого релевантного кандидата для LambdaRank бесполезны и из обучения
выбрасываются, но в метрике считаются нулями.
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import polars as pl

from common import ROOT, WORK

TOPK = 50
NON_FEATURES = {"qid", "fold", "row", "label"}

# 31/63/127 листьев и lr 0.03/0.05 на двух фолдах дают 0.915-0.917, оставлен вариант получше
PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[TOPK], learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, lambdarank_truncation_level=TOPK + 20, verbose=-1, seed=42,
              num_threads=8, deterministic=True, force_col_wise=True)

# у запросов «во всех категориях» (search_category=0) до 5 последних мест отдаём объявлениям
# не из услуг, если они среди 100 самых близких по энкодерам
RESERVE = 5
RESERVE_MAX_RANK = 100


def recall_at(df: pl.DataFrame, score_col: str, n_rel: dict, k: int = TOPK) -> float:
    # n_rel - полное число релевантных, включая не попавшие в кандидаты
    top = (df.select("qid", "label", score_col)
           .with_columns(pl.col(score_col).rank("ordinal", descending=True).over("qid").alias("_r"))
           .filter(pl.col("_r") <= k).group_by("qid").agg(pl.col("label").sum().alias("hit")))
    hits = dict(top.iter_rows())
    return float(np.mean([hits.get(q, 0) / n for q, n in n_rel.items()]))


def to_lgb(df: pl.DataFrame, feats: list[str]):
    df = df.sort("qid", maintain_order=True)
    grp = df.group_by("qid", maintain_order=True).len()["len"].to_numpy()
    return lgb.Dataset(df.select(feats).to_numpy(), label=df["label"].to_numpy(), group=grp,
                       feature_name=feats, free_raw_data=True)


def drop_useless(df: pl.DataFrame) -> pl.DataFrame:
    pos = df.group_by("qid").agg(pl.col("label").max().alias("m")).filter(pl.col("m") > 0)["qid"]
    return df.filter(pl.col("qid").is_in(pos.implode()))


def reserve_non_service(bench: pl.DataFrame, queries: pl.DataFrame) -> pl.DataFrame:
    # В train поисков по всем категориям почти нет и выбирали там только услуги, так что
    # ранкер товары не поднимает. При этом в корпусе ~1.9k объявлений не из услуг, и 40%
    # из них ближе всего именно к cat-0 запросам (а таких запросов всего 9%).
    # Позиции 46-50 на валидации дают меньше 1% попаданий, поэтому ими и жертвуем.
    cat0 = set(queries.filter((pl.col("fold") == -1) & (pl.col("search_category") == 0))["qid"].to_list())
    key = "rk_dense_loc_mean" if "rk_dense_loc_mean" in bench.columns else \
        next(c for c in bench.columns if c.startswith("rk_dense_loc_"))
    out = []
    n_forced = 0
    ordered = bench.sort(["qid", "pred", "row"], descending=[False, True, False])
    for (q,), g in ordered.group_by("qid", maintain_order=True):
        top = g["item_id"].head(TOPK).to_list()
        if q in cat0:
            ns = (g.filter(~pl.col("is_service") & (pl.col(key) < RESERVE_MAX_RANK))
                  .sort([key, "row"])["item_id"].head(RESERVE).to_list())
            extra = [x for x in ns if x not in top]
            if extra:
                n_forced += len(extra)
                top = top[:TOPK - len(extra)] + extra
        out.append((q, top))
    print(f"non-service items added for cat-0 queries: {n_forced} ({len(cat0)} queries)")
    return pl.DataFrame(out, schema=["qid", "item_id"], orient="row")


def write_answer(top: pl.DataFrame, path: str):
    bq = pl.read_parquet(WORK / "queries.parquet").filter(pl.col("fold") == -1).select("qid")
    ans = bq.join(top, on="qid", how="left", maintain_order="left")
    corpus = pl.read_parquet(WORK / "corpus.parquet", columns=["item_id", "in_bench"])
    bench_ids = set(corpus.filter(pl.col("in_bench"))["item_id"].to_list())
    out_q, out_a = [], []
    for q, items in ans.iter_rows():
        items = [x for x in dict.fromkeys(items or []) if x in bench_ids][:TOPK]
        assert len(q) == 16
        out_q.append(q)
        out_a.append(" ".join(items))
    assert len(set(out_q)) == len(out_q) == len(bq)
    pl.DataFrame({"query_id": out_q, "answer": out_a}).write_csv(path)
    lens = [len(a.split()) for a in out_a]
    print(f"{path}: {len(out_q)} rows, items per query min {min(lens)}, mean {np.mean(lens):.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default="cands")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--eval_folds", default="0,1")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--answer", default=str(ROOT / "answer.csv"))
    ap.add_argument("--no_final", action="store_true", help="только валидация")
    args = ap.parse_args()

    data = pl.read_parquet(WORK / f"{args.cands}.parquet")
    queries = pl.read_parquet(WORK / "queries.parquet")
    feats = [c for c in data.columns if c not in NON_FEATURES]
    print("features:", len(feats))
    n_rel = {q: len(r) for q, r in queries.filter(pl.col("fold") >= 0).select("qid", "rel").iter_rows()}

    # одиночные источники для сравнения
    val = data.filter(pl.col("fold") == 0)
    n_rel0 = {q: n for q, n in n_rel.items() if q.startswith("v0_")}
    base = {c: recall_at(val.with_columns((-pl.col(c)).alias("_s")), "_s", n_rel0)
            for c in data.columns if c.startswith("rk_")}
    print("single sources, Recall@50 fold 0:", {k: round(v, 4) for k, v in base.items()})

    res, iters = {}, []
    for e in [int(x) for x in args.eval_folds.split(",")]:
        tr = drop_useless(data.filter(pl.col("fold").is_between(0, 5) & (pl.col("fold") != e)))
        es = drop_useless(data.filter(pl.col("fold") == 6))
        model = lgb.train(PARAMS, to_lgb(tr, feats), num_boost_round=args.rounds, valid_sets=[to_lgb(es, feats)],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
        ve = data.filter(pl.col("fold") == e)
        ve = ve.with_columns(pl.Series("pred", model.predict(ve.select(feats).to_numpy(),
                                                             num_iteration=model.best_iteration)))
        res[e] = recall_at(ve, "pred", {q: n for q, n in n_rel.items() if q.startswith(f"v{e}_")})
        iters.append(model.best_iteration)
        print(f"ranker Recall@50 fold {e}: {res[e]:.4f} (best_iter {model.best_iteration})", flush=True)
    best = int(np.mean(iters))
    print(f"ranker Recall@50 mean: {np.mean(list(res.values())):.4f}")
    imp = sorted(zip(feats, model.feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", [(f, int(g)) for f, g in imp[:20]])
    report = {"ranker_recall50": res, "best_iter": best, "single_sources_fold0": base}
    with open(WORK / f"report_{args.cands}.json", "w") as fh:
        json.dump(report, fh, indent=1)
    if args.no_final:
        return

    full = drop_useless(data.filter(pl.col("fold") >= 0))
    dfull = to_lgb(full, feats)
    bench = data.filter(pl.col("fold") == -1)
    Xb = bench.select(feats).to_numpy()
    pred = np.zeros(len(bench))
    for sd in range(args.seeds):
        final = lgb.train(dict(PARAMS, seed=42 + sd), dfull, num_boost_round=int(best * 1.1))
        final.save_model(str(WORK / f"ranker_{args.cands}_s{sd}.txt"))
        p = pl.DataFrame({"qid": bench["qid"], "p": final.predict(Xb)})
        pred += p.select(pl.col("p").rank("average").over("qid") / pl.len().over("qid"))["p"].to_numpy()
    bench = bench.with_columns(pl.Series("pred", pred))
    corpus = pl.read_parquet(WORK / "corpus.parquet", columns=["item_id", "item_category_id"])
    rows = bench["row"].to_numpy()
    bench = bench.with_columns(pl.Series("item_id", corpus["item_id"].to_numpy()[rows]),
                               pl.Series("is_service", corpus["item_category_id"].to_numpy()[rows] == 114))
    write_answer(reserve_non_service(bench, queries), args.answer)


if __name__ == "__main__":
    main()
