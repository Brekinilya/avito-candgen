"""
Разбиение train на историю и валидационные запросы, сборка общего корпуса.

Тексты запросов в бенчмарке уникальны, и 37.0% из них встречаются в train. Если
брать из train уникальные тексты равномерно и для каждого одну поисковую группу,
выходит 37.5% - то есть бенчмарк, судя по всему, собран так же. Поэтому валидация:
7 фолдов по 2500 таких групп; их строки целиком убираются из train, остаток (hist)
идёт на обучение энкодеров и статистики истории.

Выбранные объявления валидационных запросов добавляются в корпус, но каждое видно
только своему фолду, а бенчмарку не видно совсем.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from common import DATA, GROUP_KEY, ITEM_COLS, SEED, WORK, read_bench_queries, read_items, read_train

N_FOLDS = 7
FOLD_SIZE = 2500


def main():
    rng = np.random.default_rng(SEED)
    tr = read_train().with_row_index("row")
    groups = tr.group_by(GROUP_KEY, maintain_order=True).agg(pl.col("row")).with_row_index("gid")
    print("train rows", len(tr), "groups", len(groups))

    texts = groups["search_query"].unique().sort()
    n_val = N_FOLDS * FOLD_SIZE
    chosen_texts = texts.sample(n_val, seed=SEED, shuffle=True)
    g_sub = groups.filter(pl.col("search_query").is_in(chosen_texts.implode()))
    # по одной случайной группе на текст
    g_sub = (g_sub.with_columns(pl.Series("rnd", rng.random(len(g_sub))))
             .sort("rnd").group_by("search_query", maintain_order=True).first())
    g_sub = g_sub.with_columns(pl.Series("fold", rng.permutation(np.arange(n_val) % N_FOLDS)))
    g_sub = g_sub.with_columns(pl.format("v{}_{}", pl.col("fold"), pl.col("gid")).alias("qid"))

    held_rows = g_sub.select("qid", "fold", "row").explode("row")
    held = tr.join(held_rows, on="row", how="inner", maintain_order="left")
    hist = tr.filter(~pl.col("row").is_in(held_rows["row"].implode())).drop("row")
    print("held rows", len(held), "hist rows", len(hist))
    hist.write_parquet(WORK / "hist.parquet")

    # валидационные запросы и бенчмарк в одной таблице, у бенчмарка fold = -1
    val_q = (held.group_by("qid", maintain_order=True)
             .agg([pl.col(c).first() for c in GROUP_KEY]
                  + [pl.col("fold").first(), pl.col("item_id").unique(maintain_order=True).alias("rel")]))
    bq = read_bench_queries().rename({"query_id": "qid"}).with_columns(
        pl.lit(-1).alias("fold"), pl.lit(None, dtype=pl.List(pl.String)).alias("rel"))
    cols = ["qid", "fold"] + GROUP_KEY + ["rel"]
    queries = pl.concat([val_q.select(cols), bq.select(cols)], how="vertical_relaxed")
    queries.write_parquet(WORK / "queries.parquet")

    # корпус = объявления бенчмарка + выбранные объявления валидации (folds - кому видны)
    bench_items = read_items(DATA / "benchmark_items.parquet")
    extra = (held.filter(~pl.col("item_id").is_in(bench_items["item_id"].implode()))
             .group_by("item_id", maintain_order=True).agg(
                 [pl.col(c).first() for c in ITEM_COLS if c != "item_id"]
                 + [pl.col("fold").unique(maintain_order=True).alias("folds")]))
    corpus = pl.concat([
        bench_items.select(ITEM_COLS).with_columns(pl.lit(None, dtype=pl.List(pl.Int64)).alias("folds")),
        extra.select(ITEM_COLS + ["folds"]).with_columns(pl.col("folds").cast(pl.List(pl.Int64))),
    ], how="vertical_relaxed")
    corpus = corpus.with_columns(pl.col("folds").is_null().alias("in_bench"))
    corpus.write_parquet(WORK / "corpus.parquet")
    print("corpus", len(corpus), "extra", len(extra))

    vq = queries.filter(pl.col("fold") >= 0)
    seen_text = vq["search_query"].is_in(hist["search_query"].unique().implode()).mean()
    hk = hist.select("search_query", "search_location_id").unique().with_columns(pl.lit(1).alias("x"))
    seen_tl = vq.join(hk, on=["search_query", "search_location_id"], how="left")["x"].is_not_null().mean()
    print(f"text seen in hist: {seen_text:.3f} (bench 0.370), (text, loc) seen: {seen_tl:.3f} (bench 0.065)")


if __name__ == "__main__":
    main()
