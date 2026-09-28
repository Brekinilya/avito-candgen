"""
История выборов из train: что выбирали по такому же или похожему запросу.

Похожие тексты ищем по эмбеддингам запросов (kNN по уникальным текстам hist).
От соседей берём объявления корпуса, которые по ним выбирали (источник кандидатов
и признак), и распределение подкатегорий выбранных объявлений.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch


class History:
    def __init__(self, hist: pl.DataFrame, corpus_ids: np.ndarray, hq_texts: list[str], hq_emb: np.ndarray):
        # hq_texts - уникальные тексты hist в том же порядке, что и строки hq_emb
        self.text2idx = {t: i for i, t in enumerate(hq_texts)}
        id2row = {x: i for i, x in enumerate(corpus_ids)}
        h = hist.select("search_query", "search_location_id", "item_id", "item_microcat_id")
        h = h.with_columns(pl.col("search_query").replace_strict(self.text2idx, default=-1).alias("t"))

        # P(microcat | текст)
        mc = h.group_by("t", "item_microcat_id").agg(pl.len().alias("c"))
        tot = mc.group_by("t").agg(pl.col("c").sum().alias("tot"))
        mc = mc.join(tot, on="t").with_columns((pl.col("c") / pl.col("tot")).alias("p"))
        self.mc_list = sorted(set(mc["item_microcat_id"].to_list()))
        self.mc_idx = {m: i for i, m in enumerate(self.mc_list)}
        self.P_mc = sp.csr_matrix(
            (mc["p"].to_numpy().astype(np.float32),
             (mc["t"].to_numpy(), mc["item_microcat_id"].replace_strict(self.mc_idx).to_numpy())),
            shape=(len(hq_texts), len(self.mc_list)))

        # текст -> [(строка корпуса, локация поиска, сколько раз выбирали)]
        hc = (h.with_columns(pl.col("item_id").replace_strict(id2row, default=-1).alias("r"))
              .filter(pl.col("r") >= 0)
              .group_by("t", "r", "search_location_id").agg(pl.len().alias("c"))
              .sort("t", "r", "search_location_id"))
        self.text_items: dict[int, list[tuple[int, int, int]]] = {}
        for t, r, s, c in hc.iter_rows():
            self.text_items.setdefault(t, []).append((r, s, c))
        self.hq_emb = torch.from_numpy(hq_emb).cuda().half()

    def neighbours(self, q_emb: np.ndarray, k: int = 30):
        """k ближайших исторических текстов: (индексы, косинусы)."""
        sims = torch.from_numpy(q_emb).cuda().half() @ self.hq_emb.T
        v, i = torch.topk(sims, k, dim=1)
        return i.cpu().numpy(), v.float().cpu().numpy()

    def microcat_dist(self, nb_idx: np.ndarray, nb_sim: np.ndarray, temp: float = 0.05) -> np.ndarray:
        """Смесь распределений подкатегорий соседей с весами softmax(sim / temp)."""
        w = np.exp((nb_sim - nb_sim[:, :1]) / temp)
        w = w / w.sum(1, keepdims=True)
        out = np.zeros((len(nb_idx), len(self.mc_list)), dtype=np.float32)
        for j in range(nb_idx.shape[1]):
            out += w[:, j:j + 1] * self.P_mc[nb_idx[:, j]].toarray()
        return out
