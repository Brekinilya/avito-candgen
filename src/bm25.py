"""
BM25 по трём полям объявления: заголовок, параметры, описание.

Для поля хранится матрица W[term, doc] с уже посчитанным весом BM25, тогда скор
пачки запросов - это одно разреженное умножение Q @ W. В заголовке индексируются
ещё и биграммы соседних стемов: запросы короткие, и совпадение фразы целиком
(«натяжные потолки») заметно полезнее совпадения отдельных слов.
"""
from __future__ import annotations

import pickle

import numpy as np
import polars as pl
import scipy.sparse as sp

from common import WORK, Vocab, tokenize_column

# поле: (колонка, k1, b, сколько первых токенов брать, биграммы)
FIELDS = {
    "title": ("item_title_raw", 1.2, 0.3, None, True),
    "params": ("item_infm_params_text", 1.2, 0.75, 400, False),
    "desc": ("item_description_raw", 1.2, 0.75, 600, False),
}


def _bigram_ids(rows: np.ndarray, tids: np.ndarray, vocab: Vocab, add: bool):
    same = rows[1:] == rows[:-1]   # соседние токены одного документа
    a, b = tids[:-1][same], tids[1:][same]
    r = rows[1:][same]
    if len(a) == 0:
        return r, a
    key = a.astype(np.int64) * 10_000_000 + b
    uniq, back = np.unique(key, return_inverse=True)
    names = [f"§{u // 10_000_000}_{u % 10_000_000}" for u in uniq]
    ids = np.array(vocab.ids(names, add=add), dtype=np.int64)[back]
    ok = ids >= 0
    return r[ok], ids[ok]


class BM25Index:
    def __init__(self):
        self.vocab = Vocab()
        self.W: dict[str, sp.csr_matrix] = {}
        self.n_docs = 0

    def build(self, corpus: pl.DataFrame):
        self.n_docs = len(corpus)
        raw = {}
        for f, (col, k1, b, max_tokens, use_bigrams) in FIELDS.items():
            r, t = tokenize_column(corpus, col, self.vocab, add=True, max_tokens=max_tokens)
            if use_bigrams:
                r2, t2 = _bigram_ids(r, t, self.vocab, add=True)
                r, t = np.concatenate([r, r2]), np.concatenate([t, t2])
            raw[f] = (r, t, k1, b)
            print(f"{f}: {len(r)} tokens, vocab {len(self.vocab)}")
        V = len(self.vocab)
        for f, (r, t, k1, b) in raw.items():
            tf = sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, t)), shape=(self.n_docs, V))
            tf.sum_duplicates()
            dl = np.asarray(tf.sum(axis=1)).ravel()
            avg = dl.mean() if dl.mean() > 0 else 1.0
            df = np.bincount(tf.indices, minlength=V)
            idf = np.log(1 + (self.n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
            tf = tf.tocoo()
            denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avg)
            w = idf[tf.col] * tf.data * (k1 + 1) / denom
            self.W[f] = sp.csr_matrix((w.astype(np.float32), (tf.col, tf.row)), shape=(V, self.n_docs))
        return self

    def query_matrix(self, queries: pl.DataFrame, col: str = "search_query", bigrams: bool = True,
                     exclude: set[int] | None = None) -> sp.csr_matrix:
        """Бинарная матрица [запрос, термин]; незнакомые индексу слова отбрасываются."""
        r, t = tokenize_column(queries, col, self.vocab, add=False, drop_stop=True)
        if exclude:
            keep = ~np.isin(t, list(exclude))
            r, t = r[keep], t[keep]
        if bigrams:
            r2, t2 = _bigram_ids(r, t, self.vocab, add=False)
            r, t = np.concatenate([r, r2]), np.concatenate([t, t2])
        Q = sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, t)), shape=(len(queries), len(self.vocab)))
        Q.data[:] = 1.0   # повтор слова в запросе вес не увеличивает
        return Q

    def score(self, Q: sp.csr_matrix, field: str) -> sp.csr_matrix:
        return (Q @ self.W[field]).tocsr()

    def save(self, path=WORK / "bm25.pkl"):
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path=WORK / "bm25.pkl") -> "BM25Index":
        with open(path, "rb") as fh:
            return pickle.load(fh)
