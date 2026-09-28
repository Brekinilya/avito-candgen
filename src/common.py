"""Пути, чтение данных и токенизация."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import polars as pl
import Stemmer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WORK = ROOT / "work"   # сюда пишутся все промежуточные файлы
WORK.mkdir(exist_ok=True)

SEED = 42

# один запрос бенчмарка соответствует одной такой группе в train
GROUP_KEY = ["search_query", "search_location_id", "search_infm_params_text",
             "search_category", "search_is_delivery_search"]

ITEM_COLS = ["item_id", "item_title_raw", "item_description_raw", "item_infm_params_text",
             "item_category_id", "item_microcat_id", "item_price", "item_rating",
             "item_rating_reviews_count", "item_location_id", "item_latitude",
             "item_longitude", "item_is_phone_hidden", "item_is_message_forbidden"]


# parquet читаем через polars: старый pyarrow (<20) падает на benchmark_queries.parquet
def read_items(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    return df.with_columns(
        pl.col("item_price").cast(pl.Float64),
        pl.col("item_latitude").cast(pl.Float64),
        pl.col("item_longitude").cast(pl.Float64),
        pl.col("item_title_raw").fill_null(""),
        pl.col("item_description_raw").fill_null(""),
        pl.col("item_infm_params_text").fill_null(""),
    )


def read_train(columns: list[str] | None = None) -> pl.DataFrame:
    df = pl.read_parquet(DATA / "train.parquet", columns=columns)
    casts = [pl.col(c).cast(pl.Float64) for c in ("item_price", "item_latitude", "item_longitude")
             if c in df.columns]
    fills = [pl.col(c).fill_null("") for c in ("search_infm_params_text", "item_title_raw",
                                                "item_description_raw", "item_infm_params_text")
             if c in df.columns]
    return df.with_columns(casts + fills)


def read_bench_queries() -> pl.DataFrame:
    return pl.read_parquet(DATA / "benchmark_queries.parquet").with_columns(
        pl.col("search_infm_params_text").fill_null(""))


_RU = Stemmer.Stemmer("russian")
_EN = Stemmer.Stemmer("english")

STOP = set("""и в во на по с со к ко у о об от до за из для при про без над под
через а но или ли же бы не ни то это как что так уже все всё вы мы я он она они
его ее её их мой ваш наш свой""".split())


def norm_expr(col: str | pl.Expr) -> pl.Expr:
    """lower, ё -> е, всё кроме букв и цифр -> пробел."""
    e = pl.col(col) if isinstance(col, str) else col
    return (e.fill_null("").str.to_lowercase().str.replace_all("ё", "е")
            .str.replace_all(r"[^a-zа-я0-9]+", " ").str.strip_chars())


def stem_token(t: str) -> str:
    if re.search("[а-я]", t):
        return _RU.stemWord(t)
    return _EN.stemWord(t)


class Vocab:
    """stem -> id, стемминг кэшируется по токенам."""

    def __init__(self):
        self.stem2id: dict[str, int] = {}
        self._tok2stem: dict[str, str] = {}

    def stems(self, tokens):
        out = []
        cache = self._tok2stem
        for t in tokens:
            s = cache.get(t)
            if s is None:
                s = stem_token(t)
                cache[t] = s
            out.append(s)
        return out

    def ids(self, stems, add=True):
        d = self.stem2id
        res = []
        for s in stems:
            i = d.get(s)
            if i is None:
                if not add:
                    res.append(-1)
                    continue
                i = len(d)
                d[s] = i
            res.append(i)
        return res

    def __len__(self):
        return len(self.stem2id)


def tokenize_column(df: pl.DataFrame, col: str, vocab: Vocab, add: bool = True,
                    drop_stop: bool = True, max_tokens: int | None = None):
    """Возвращает (номер строки, id термина) для каждого вхождения токена, повторы = tf."""
    toks = df.select(norm_expr(col).str.split(" ").alias("t")).with_row_index("r")
    if max_tokens is not None:
        toks = toks.with_columns(pl.col("t").list.head(max_tokens))
    ex = toks.explode("t").filter(pl.col("t").is_not_null() & (pl.col("t") != ""))
    if drop_stop:
        ex = ex.filter(~pl.col("t").is_in(list(STOP)))
    # стеммим только уникальные токены, так в разы быстрее
    uniq = ex["t"].unique(maintain_order=True).to_list()
    ids = vocab.ids(vocab.stems(uniq), add=add)
    m = pl.DataFrame({"t": pl.Series(uniq, dtype=pl.String), "tid": np.array(ids, dtype=np.int64)})
    # порядок токенов нужен для биграмм
    ex = ex.join(m, on="t", how="left", maintain_order="left").filter(pl.col("tid") >= 0)
    return ex["r"].to_numpy().astype(np.int64), ex["tid"].to_numpy().astype(np.int64)
