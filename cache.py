"""
cache.py
Simple pickle cache for backtest results.
Keyed by MD5 of: last_date + start_yr + fee_bps + split + lookback
"""

import hashlib
import pickle
from pathlib import Path

CACHE_DIR     = Path("/tmp/p2_arima_cache")
CACHE_VERSION = "v1"
CACHE_DIR.mkdir(exist_ok=True)


def make_cache_key(last_date, start_yr, fee_bps, split, lookback):
    raw = f"{CACHE_VERSION}_{last_date}_{start_yr}_{fee_bps}_{split}_{lookback}"
    return hashlib.md5(raw.encode()).hexdigest()


def make_lb_cache_key(last_date, start_yr, split):
    raw = f"{CACHE_VERSION}_lb_{last_date}_{start_yr}_{split}"
    return hashlib.md5(raw.encode()).hexdigest()


def save_cache(key, payload):
    try:
        with open(CACHE_DIR / f"{key}.pkl", "wb") as f:
            pickle.dump(payload, f)
    except Exception:
        pass


def load_cache(key):
    path = CACHE_DIR / f"{key}.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            path.unlink(missing_ok=True)
    return None
