"""
Трудные негативы для второго прохода обучения энкодера.

Для каждой пары (запрос, выбранное объявление) берём самые похожие по текущей модели
объявления из hist в той же локации и выбираем одно случайное с позиций 3..30.
Первые позиции пропускаем - там часто дубли позитива или равноценные варианты.
Объявления, которые выбирали по этому же тексту запроса, негативами не считаются.
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl
import torch

from common import SEED, WORK
from dense import D_MAXLEN, Q_MAXLEN, Encoder, item_texts, query_texts
from s03_train_dense import build_pairs

SKIP, TOP = 3, 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="e5s")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    pairs = build_pairs(pl.read_parquet(WORK / "hist.parquet")).with_row_index("row")
    items = pairs.unique(subset=["item_id"], keep="first", maintain_order=True)
    print("pairs", len(pairs), "unique items", len(items))

    enc = Encoder(str(WORK / f"dense_{args.model}")).to("cuda").half()
    d_txt = item_texts(items)
    I = torch.from_numpy(enc.encode(d_txt, D_MAXLEN)).cuda()
    Q = enc.encode(query_texts(pairs), Q_MAXLEN)

    id2i = {x: i for i, x in enumerate(items["item_id"].to_list())}
    it_loc = torch.from_numpy(items["item_location_id"].to_numpy().copy()).cuda()
    chosen = {q: {id2i[x] for x in lst}
              for q, lst in pairs.group_by("search_query").agg(pl.col("item_id")).iter_rows()}
    qtext = pairs["search_query"].to_list()
    ploc = torch.from_numpy(pairs["item_location_id"].to_numpy().copy()).cuda()

    neg = np.full(len(pairs), -1, dtype=np.int64)
    for c0 in range(0, len(pairs), 1024):
        s = (torch.from_numpy(Q[c0:c0 + 1024]).cuda() @ I.T).float()
        s = torch.where(it_loc[None, :] == ploc[c0:c0 + 1024, None], s, torch.tensor(-1e9, device="cuda"))
        v, ix = torch.topk(s, TOP + 5, dim=1)
        v, ix = v.cpu().numpy(), ix.cpu().numpy()
        for j in range(len(ix)):
            bad = chosen[qtext[c0 + j]]
            cand = [x for x, sv in zip(ix[j], v[j]) if sv > -1e8 and x not in bad][SKIP:TOP]
            if cand:
                neg[c0 + j] = cand[rng.integers(len(cand))]
    ok = neg >= 0
    print("pairs with hard negative:", ok.mean())
    out = pl.DataFrame({"row": pairs["row"].to_numpy()[ok].astype(np.uint32),
                        "neg_text": [d_txt[k] for k in neg[ok]]})
    out.write_parquet(WORK / f"hardneg_{args.model}.parquet")


if __name__ == "__main__":
    main()
