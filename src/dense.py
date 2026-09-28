"""
Bi-encoder на базе intfloat/multilingual-e5 (small / base), дообучается на парах
(запрос, выбранное объявление) из train.

Loss - InfoNCE: позитив - выбранное объявление, негативы - объявления остальных
запросов батча и, если заданы, трудные негативы (s03b_mine_hardneg.py).
"""
from __future__ import annotations

import math
import re
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

BASE_MODEL = "intfloat/multilingual-e5-small"
Q_MAXLEN = 32
D_MAXLEN = 160

# шаблонные куски item_infm_params_text (цены, графики работы и т.п.), для поиска бесполезны
_BOILER = re.compile(
    r"(Начальная цена|Тип стоимости за (услугу|час|урок|день|м²|м2|км|смену|ед\.?)|Стоимость \d+|"
    r"График работы(, дни недели)? (от|до)?\s*\S+|Время работы, (с|до) \S+|Дни (пн|вт|ср|чт|пт|сб|вс)|"
    r"Рабочие дни \S+|Продолжительность [^А-Я]*|Признак предзаполнения прайс листа \d+|Услуга Своя услуга|"
    r"Минимальная сумма заказа \d+|Выполняю заказы от \d+|Опыт работы [^А-Я]*)")
_WS = re.compile(r"\s+")


def clean_params(s: str, max_chars: int = 400) -> str:
    s = _BOILER.sub(" ", s or "")
    return _WS.sub(" ", s).strip()[:max_chars]


def item_texts(df: pl.DataFrame) -> list[str]:
    # заголовок + параметры без шаблонов + начало описания; префиксы query:/passage: как у e5
    out = []
    for t, p, d in zip(df["item_title_raw"].to_list(), df["item_infm_params_text"].to_list(),
                       df["item_description_raw"].to_list()):
        d = _WS.sub(" ", d or "")[:600]
        out.append(f"passage: {t or ''}. {clean_params(p, 250)}. {d}")
    return out


def query_texts(df: pl.DataFrame) -> list[str]:
    # фильтры поиска («Вид услуги ...») дописываем к запросу
    out = []
    for q, p in zip(df["search_query"].to_list(), df["search_infm_params_text"].to_list()):
        p = (p or "").strip()
        out.append(f"query: {q}" + (f" | {p}" if p else ""))
    return out


class Encoder(torch.nn.Module):
    def __init__(self, name_or_path: str = BASE_MODEL):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(name_or_path)
        self.model = AutoModel.from_pretrained(name_or_path)

    def forward(self, texts: list[str], max_len: int) -> torch.Tensor:
        dev = next(self.model.parameters()).device
        b = self.tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
        h = self.model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)   # mean pooling
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def encode(self, texts: list[str], max_len: int, batch: int = 256) -> np.ndarray:
        self.eval()
        order = np.argsort([len(t) for t in texts])   # меньше паддинга в батчах
        res = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float16)
        for i in range(0, len(texts), batch):
            idx = order[i:i + batch]
            with torch.autocast("cuda", dtype=torch.float16):
                e = self([texts[j] for j in idx], max_len)
            res[idx] = e.float().cpu().numpy().astype(np.float16)
        return res

    def save(self, path):
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)


def train_biencoder(q_txt: list[str], d_txt: list[str], neg_txt: list[str] | None, out_dir, epochs: int = 1,
                    batch: int = 128, lr: float = 5e-5, scale: float = 20.0, base: str = BASE_MODEL, seed: int = 42):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda"
    enc = Encoder(base).to(dev)
    enc.model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler()
    total = (len(q_txt) // batch) * epochs
    warm = max(1, int(0.05 * total))
    # линейный разогрев 5% шагов, дальше косинус
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    for ep in range(epochs):
        # в одном батче не должно быть двух пар с одинаковым запросом, иначе ложные негативы
        batches, cur, seen = [], [], set()
        for i in rng.permutation(len(q_txt)):
            if q_txt[i] in seen:
                continue
            cur.append(i)
            seen.add(q_txt[i])
            if len(cur) == batch:
                batches.append(cur)
                cur, seen = [], set()
        enc.train()
        t0 = time.time()
        for bi, b in enumerate(batches):
            qs = [q_txt[i] for i in b]
            ds = [d_txt[i] for i in b]
            if neg_txt is not None:
                ds = ds + [neg_txt[i] for i in b]
            with torch.autocast("cuda", dtype=torch.float16):
                qe = enc(qs, Q_MAXLEN)
                de = enc(ds, D_MAXLEN)
                logits = (qe @ de.T).float() * scale
                labels = torch.arange(len(qs), device=dev)
                # основное направление запрос -> объявление, плюс немного обратного
                loss = F.cross_entropy(logits, labels) + 0.3 * F.cross_entropy(logits[:, :len(qs)].T, labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if bi % 200 == 0:
                print(f"ep {ep} step {bi}/{len(batches)} loss {loss.item():.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
    enc.save(out_dir)
    return enc
