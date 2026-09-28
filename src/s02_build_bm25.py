"""BM25-индекс по объединённому корпусу."""
import time

import polars as pl

from bm25 import BM25Index
from common import WORK


def main():
    t0 = time.time()
    corpus = pl.read_parquet(WORK / "corpus.parquet")
    idx = BM25Index().build(corpus)
    idx.save()
    print(f"docs={idx.n_docs} vocab={len(idx.vocab)}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
