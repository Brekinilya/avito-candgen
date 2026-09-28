"""
Дообучение энкодера на hist и расчёт эмбеддингов корпуса и запросов.

  python s03_train_dense.py --name e5s                      # обучить и закодировать
  python s03_train_dense.py --name e5s --encode_only        # только закодировать (веса уже в work/dense_e5s)
  python s03_train_dense.py --name e5b_raw --base intfloat/multilingual-e5-base --encode_only
                                                           # исходная модель без дообучения
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl

from common import SEED, WORK
from dense import D_MAXLEN, Q_MAXLEN, Encoder, item_texts, query_texts, train_biencoder

# у популярных запросов тысячи пар («маникюр» - 6.5k), а бенчмарк равномерен по текстам,
# так что берём не больше 30 пар на текст
CAP_PER_QUERY = 30


def build_pairs(hist: pl.DataFrame) -> pl.DataFrame:
    pairs = hist.unique(subset=["search_query", "search_infm_params_text", "item_id"], keep="first",
                        maintain_order=True)
    pairs = (pairs.with_columns(pl.int_range(pl.len()).shuffle(seed=SEED).over("search_query").alias("rk"))
             .filter(pl.col("rk") < CAP_PER_QUERY).drop("rk"))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="e5s")
    ap.add_argument("--base", default="intfloat/multilingual-e5-small")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--hardneg", default="", help="parquet с колонками row, neg_text")
    ap.add_argument("--encode_only", action="store_true")
    args = ap.parse_args()
    out = WORK / f"dense_{args.name}"

    if not args.encode_only:
        pairs = build_pairs(pl.read_parquet(WORK / "hist.parquet"))
        print("pairs", len(pairs))
        q = query_texts(pairs)
        d = item_texts(pairs)
        neg = None
        if args.hardneg:
            hn = pl.read_parquet(args.hardneg)
            pairs = pairs.with_row_index("row").join(hn, on="row", how="left", maintain_order="left")
            neg = pairs["neg_text"].to_list()
            # где трудного негатива не нашлось - случайное объявление
            rnd = np.random.default_rng(SEED).permutation(len(d))
            neg = [n if n is not None else d[rnd[i]] for i, n in enumerate(neg)]
        t0 = time.time()
        train_biencoder(q, d, neg, out, epochs=args.epochs, batch=args.batch, lr=args.lr, base=args.base)
        print(f"trained in {time.time() - t0:.0f}s")

    enc = Encoder(str(out) if out.exists() else args.base).to("cuda").half()
    t0 = time.time()
    queries = pl.read_parquet(WORK / "queries.parquet")
    corpus = pl.read_parquet(WORK / "corpus.parquet")
    np.save(WORK / f"emb_items_{args.name}.npy", enc.encode(item_texts(corpus), D_MAXLEN, batch=256))
    np.save(WORK / f"emb_queries_{args.name}.npy", enc.encode(query_texts(queries), Q_MAXLEN, batch=512))
    # для поиска похожих запросов в истории - только текст, без фильтров
    qtxt = [f"query: {t}" for t in queries["search_query"].to_list()]
    np.save(WORK / f"emb_queries_txt_{args.name}.npy", enc.encode(qtxt, Q_MAXLEN, batch=512))
    htexts = pl.read_parquet(WORK / "hist.parquet", columns=["search_query"])["search_query"].unique().sort()
    htexts.to_frame().write_parquet(WORK / "hist_texts.parquet")
    np.save(WORK / f"emb_hist_txt_{args.name}.npy",
            enc.encode([f"query: {t}" for t in htexts.to_list()], Q_MAXLEN, batch=512))
    print(f"encoded in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
