import csv
import json
import os
from typing import Dict, List

import pandas as pd

from .utils import ensure_dir


def save_csv(path: str, rows: List[Dict]) -> None:
    ensure_dir(path)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def save_json(path: str, rows: List[Dict]) -> None:
    ensure_dir(path)
    existing: List[Dict] = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except Exception:
                existing = []
    existing.extend(rows)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


def save_any(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    ext = os.path.splitext(path)[1].lower()
    if ext in {".csv"}:
        save_csv(path, rows)
    elif ext in {".json"}:
        save_json(path, rows)
    elif ext in {".db", ".sqlite", ".sqlite3"}:
        ensure_dir(path)
        df = pd.DataFrame(rows)
        import sqlite3

        conn = sqlite3.connect(path)
        try:
            df.to_sql("results", conn, if_exists="append", index=False)
        finally:
            conn.close()
    else:
        save_csv(path, rows)


