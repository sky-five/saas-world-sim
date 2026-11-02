"""A fake SaaS API over the simulated world: paginated, with an `updated_at` cursor."""
import json
from collections.abc import Iterator
from datetime import date, datetime

import numpy as np
import pandas as pd
import pyarrow as pa

from saas_world_sim.config import SEED
from saas_world_sim.world import TABLES, world_as_of


def _to_records(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    for c in out.select_dtypes("datetime").columns:
        out[c] = out[c].dt.strftime("%Y-%m-%dT%H:%M:%S").where(out[c].notna(), None)
    out = out.astype(object).where(out.notna(), None)
    records = out.to_dict("records")
    for r in records:  # numpy scalars -> plain Python, so json.dumps works
        for k, v in r.items():
            if isinstance(v, np.generic):
                r[k] = v.item()
    return records


class MockApi:
    def __init__(self, as_of: date, seed: int = SEED):
        self.as_of = as_of
        self._world = world_as_of(as_of, seed)

    def _rows(self, table: str, updated_since: datetime | None) -> pd.DataFrame:
        spec = TABLES[table]  # KeyError for unknown tables
        df = self._world[table]
        if updated_since is not None:
            df = df[df.updated_at > pd.Timestamp(updated_since)]
        df = df.sort_values(["updated_at", spec.primary_key], kind="stable")
        if table == "events":  # the event payload arrives as a JSON document
            df = df.assign(properties=[json.dumps({"value_usd": v}) if pd.notna(v) else "{}"
                                       for v in df.value_usd]).drop(columns="value_usd")
        return df

    def fetch(self, table: str, updated_since: datetime | None = None,
              page_size: int = 5000) -> Iterator[list[dict]]:
        """JSON-style pages, like a REST API."""
        df = self._rows(table, updated_since)
        for start in range(0, len(df), page_size):
            records = _to_records(df.iloc[start:start + page_size])
            if table == "events":
                for r in records:
                    r["properties"] = json.loads(r["properties"])
            yield records

    def fetch_arrow(self, table: str, updated_since: datetime | None = None,
                    page_size: int = 250_000) -> Iterator[pa.Table]:
        """The same pages as Arrow tables, for fast bulk loading."""
        df = self._rows(table, updated_since)
        for start in range(0, len(df), page_size):
            yield pa.Table.from_pandas(df.iloc[start:start + page_size], preserve_index=False)
