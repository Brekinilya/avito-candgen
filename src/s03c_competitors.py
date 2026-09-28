"""
Конкуренты для валидационных запросов.

У запроса бенчмарка в его локации заметно больше похожих объявлений, чем у того же
запроса в случайной локации: судя по всему, в корпус попали объявления, показанные
по этим запросам. У валидационных запросов такого нет, и без поправки валидация
получается намного проще бенчмарка (~0.96 против ~0.92 после поправки).

Поэтому для каждого валидационного запроса берём похожие на него объявления
бенчмарка из других локаций и добавляем их копии, перенесённые в локацию поиска.
Сколько копий добавить - сэмплируем из того, насколько в бенчмарке похожих
объявлений в своей локации больше, чем при перемешанных локациях.

Похожесть считаем по исходной e5-base без дообучения: если отбирать конкурентов
нашими же энкодерами, валидация будет несправедливо штрафовать именно их.
Копии видны только своему фолду и никогда не бывают релевантными.
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl
import torch

from common import SEED, WORK
from location import LocationModel

POOL = 150   # копии выбираются из стольких самых похожих объявлений


def sims(names, qidx, dev="cuda"):
    s = 0
    for n in names:
        I = torch.from_numpy(np.load(WORK / f"emb_items_{n}.npy")).to(dev)
        Q = torch.from_numpy(np.load(WORK / f"emb_queries_{n}.npy")[qidx]).to(dev)
        s = s + (Q @ I.T).float()
    return s / len(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default="e5b_raw")
    args = ap.parse_args()
    names = args.dense.split(",")
    rng = np.random.default_rng(SEED)
    dev = "cuda"
    corpus = pl.read_parquet(WORK / "corpus.parquet", columns=["item_id", "item_location_id", "item_latitude",
                                                                "item_longitude", "in_bench"])
    queries = pl.read_parquet(WORK / "queries.parquet")
    hist = pl.read_parquet(WORK / "hist.parquet", columns=["search_location_id", "item_location_id",
                                                           "item_latitude", "item_longitude"])
    hist = hist.with_columns(pl.col("item_latitude", "item_longitude").cast(pl.Float64))
    lm = LocationModel(hist, corpus)
    iloc = torch.from_numpy(corpus["item_location_id"].to_numpy().copy()).to(dev)
    inb = torch.from_numpy(corpus["in_bench"].to_numpy().copy()).to(dev)
    folds = queries["fold"].to_numpy()
    locs = queries["search_location_id"].to_numpy()

    # порог похожести: 25-й перцентиль близости запроса к его выбранному объявлению,
    # так он не зависит от шкалы косинуса модели
    id2row = {x: i for i, x in enumerate(corpus["item_id"].to_list())}
    qv_sample = np.flatnonzero(folds >= 0)[:5000]
    rel_sims = []
    for c0 in range(0, len(qv_sample), 256):
        ids = qv_sample[c0:c0 + 256]
        s = sims(names, ids).cpu().numpy()
        for j, qi in enumerate(ids):
            rel_sims += [s[j, id2row[x]] for x in queries["rel"][int(qi)]]
    thr = float(np.quantile(rel_sims, 0.25))
    print(f"similarity threshold: {thr:.4f}")

    # насколько в бенчмарке похожих объявлений в своей локации больше, чем в чужой
    qb = np.flatnonzero(folds == -1)
    shuf = rng.permutation(locs[qb])
    excess = []
    for c0 in range(0, len(qb), 256):
        ids = qb[c0:c0 + 256]
        m = (sims(names, ids) >= thr) & inb[None, :]
        own = (m & (iloc[None, :] == torch.from_numpy(locs[ids]).to(dev)[:, None])).sum(1)
        other = (m & (iloc[None, :] == torch.from_numpy(shuf[c0:c0 + 256]).to(dev)[:, None])).sum(1)
        excess.append((own - other).clamp(min=0).cpu().numpy())
    excess = np.concatenate(excess)
    print(f"bench excess: mean {excess.mean():.2f}, median {np.median(excess):.0f}, "
          f"p90 {np.quantile(excess, 0.9):.0f}")

    # координаты копии берём у случайного реального объявления её новой локации;
    # с «центр города + шум» ранкер начинал узнавать копии по расстоянию
    pts = pl.concat([
        df.select(pl.col("item_location_id").alias("l"), pl.col("item_latitude").alias("a"),
                  pl.col("item_longitude").alias("o"))
        for df in (corpus, hist)]).drop_nulls()
    loc_pts = {l: np.stack([np.asarray(a), np.asarray(o)], 1)
               for l, a, o in pts.group_by("l").agg("a", "o").iter_rows()}

    qv = np.flatnonzero(folds >= 0)
    rows = []
    for c0 in range(0, len(qv), 256):
        ids = qv[c0:c0 + 256]
        s = sims(names, ids)
        other_loc = iloc[None, :] != torch.from_numpy(locs[ids]).to(dev)[:, None]
        s = torch.where(inb[None, :] & other_loc, s, torch.tensor(-1.0, device=dev))
        v, ix = torch.topk(s, POOL, dim=1)
        v, ix = v.cpu().numpy(), ix.cpu().numpy()
        for j, qi in enumerate(ids):
            n = int(excess[rng.integers(len(excess))])
            pool = ix[j][v[j] >= thr]
            if n == 0 or len(pool) == 0:
                continue
            pick = rng.choice(pool, size=min(n, len(pool)), replace=False)
            s0 = int(locs[qi])
            # у города копия почти всегда в нём же, у «региона» - в дочерних городах,
            # по тем же долям, что в истории
            trans = sorted(lm.trans.get(s0, {}).items())
            for r in pick:
                if trans:
                    ls, cs = zip(*trans)
                    l = int(ls[rng.choice(len(ls), p=np.array(cs, dtype=float) / sum(cs))])
                else:
                    l = s0
                p = loc_pts.get(l)
                if p is not None:
                    lat, lon = p[rng.integers(len(p))] + rng.normal(0, 0.003, 2)
                else:
                    c = lm.centroid.get(l) or lm.centroid.get(s0) or (np.nan, np.nan)
                    lat, lon = c[0] + rng.normal(0, 0.03), c[1] + rng.normal(0, 0.05)
                rows.append((int(folds[qi]), queries["qid"][int(qi)], int(r), l, float(lat), float(lon)))
    comp = pl.DataFrame(rows, schema=["fold", "qid", "src_row", "loc", "lat", "lon"], orient="row")
    comp.write_parquet(WORK / "competitors.parquet")
    print("competitors:", len(comp), "per val query:", round(len(comp) / len(qv), 1))


if __name__ == "__main__":
    main()
