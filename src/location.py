"""
Локации.

В train 83% выбранных объявлений лежат в локации поиска. Остальное - «регионы»
(107620 похоже на Московскую область, 107621 - на Ленинградскую, 621540 - на всю
Россию; объявления там из дочерних городов), соседние города и услуги без привязки
к месту. Поэтому жёстко по локации не фильтруем, а используем P(локация объявления
| локация поиска) по истории и расстояние до центра локации поиска.
"""
from __future__ import annotations

import numpy as np
import polars as pl


class LocationModel:
    def __init__(self, hist: pl.DataFrame, items: pl.DataFrame, alpha: float = 1.0):
        coords = pl.concat([
            df.select(pl.col("item_location_id").alias("loc"), pl.col("item_latitude").alias("lat"),
                      pl.col("item_longitude").alias("lon"))
            for df in (items, hist)]).drop_nulls()
        cen = coords.group_by("loc").agg(pl.col("lat").median(), pl.col("lon").median())
        self.centroid = {loc: (lat, lon) for loc, lat, lon in cen.iter_rows()}
        # регионы сами локацией объявления не бывают - для них центр считаем
        # по объявлениям, которые там выбирали
        sc = (hist.select(pl.col("search_location_id").alias("loc"), pl.col("item_latitude").alias("lat"),
                          pl.col("item_longitude").alias("lon")).drop_nulls()
              .group_by("loc").agg(pl.col("lat").median(), pl.col("lon").median()))
        for loc, lat, lon in sc.iter_rows():
            self.centroid.setdefault(loc, (lat, lon))

        # сколько раз из локации поиска s выбирали объявление в локации l
        tr = hist.group_by("search_location_id", "item_location_id").agg(pl.len().alias("c"))
        self.trans: dict[int, dict[int, int]] = {}
        for s, l, c in tr.iter_rows():
            self.trans.setdefault(s, {})[l] = c
        self.total = {s: sum(t.values()) for s, t in self.trans.items()}
        # насколько поиск в этой локации «локальный»
        self.self_frac = {s: self.trans[s].get(s, 0) / self.total[s] for s in self.trans}
        self.alpha = alpha

    def dist_km(self, s: int, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        c = self.centroid.get(s)
        if c is None:
            return np.full(len(lat), np.nan, dtype=np.float32)
        la1, lo1 = np.radians(c[0]), np.radians(c[1])
        la2, lo2 = np.radians(lat), np.radians(lon)
        a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
        return (2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).astype(np.float32)

    def log_p(self, s: int, item_locs: np.ndarray) -> np.ndarray:
        """log P(item_loc | s) со сглаживанием; если s в истории не встречалась - только своя локация."""
        t = self.trans.get(s)
        if not t:
            return np.where(item_locs == s, np.log(0.8), np.log(1e-4)).astype(np.float32)
        uniq, inv = np.unique(item_locs, return_inverse=True)
        vals = np.array([t.get(int(u), 0) for u in uniq], dtype=np.float64)
        p = (vals + self.alpha * 1e-3) / (self.total[s] + self.alpha)
        return np.log(p)[inv].astype(np.float32)
