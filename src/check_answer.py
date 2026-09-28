"""Проверка формата answer.csv по исходным parquet (без промежуточных файлов пайплайна)."""
import csv
import re
import sys

import polars as pl

from common import DATA, ROOT

path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "answer.csv")
q = pl.read_parquet(DATA / "benchmark_queries.parquet")["query_id"].to_list()
items = set(pl.read_parquet(DATA / "benchmark_items.parquet", columns=["item_id"])["item_id"].to_list())

with open(path, encoding="utf-8", newline="") as fh:
    rows = list(csv.reader(fh))
assert rows[0] == ["query_id", "answer"], f"bad header {rows[0]}"
body = rows[1:]
assert all(len(r) == 2 for r in body), "expected exactly 2 columns"
ids = [r[0] for r in body]
assert len(ids) == len(set(ids)), "duplicate query_id"
assert set(ids) == set(q), f"missing {len(set(q) - set(ids))}, extra {len(set(ids) - set(q))}"
assert all(len(x) == 16 for x in ids)
hexre = re.compile(r"^[0-9a-f]{16}$")
n_items = []
for qid, ans in body:
    lst = ans.split(" ") if ans else []
    assert len(lst) <= 50, f"{qid}: more than 50 items"
    assert len(lst) == len(set(lst)), f"{qid}: duplicates"
    assert all(hexre.match(x) for x in lst), f"{qid}: bad item_id"
    assert all(x in items for x in lst), f"{qid}: item not in corpus"
    n_items.append(len(lst))
print(f"OK: {len(body)} rows, items per row {min(n_items)}..{max(n_items)}")
