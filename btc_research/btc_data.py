# btc_data.py — Standalone BTC data fetcher (OHLCV + funding + L/S + OI)
#
# Khusus untuk riset BTC-only. TIDAK mengubah data_fetcher.fetch_btc_data() yang
# masih dipakai pipeline cross-section. Cache terpisah di _btc_research.parquet.

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data_fetcher import (
    get_exchange,
    fetch_ohlcv,
    fetch_funding_rate,
    fetch_long_short_ratio,
    fetch_open_interest,
)


BTC_SYMBOL  = "BTC/USDT:USDT"
CACHE_PATH  = f"{config.CACHE_DIR}/_btc_research.parquet"
DEFAULT_DAYS = 365   # 1 tahun — stress-test multi-regime


def fetch_btc_research_data(force_refresh: bool = False,
                            days: int = DEFAULT_DAYS) -> pd.DataFrame:
    """BTC 4H OHLCV + funding + L/S + open_interest, merged on datetime index."""
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    if not force_refresh and os.path.exists(CACHE_PATH):
        age_h = (time.time() - os.path.getmtime(CACHE_PATH)) / 3600
        if age_h < config.DATA_CACHE_HOURS:
            return pd.read_parquet(CACHE_PATH)

    exchange = get_exchange()

    ohlcv = fetch_ohlcv(exchange, BTC_SYMBOL, days=days)
    if ohlcv.empty:
        raise RuntimeError(f"OHLCV fetch returned empty for {BTC_SYMBOL}")
    df = ohlcv.copy()

    # Drop candle terakhir kalau belum closed — Binance return candle ongoing
    # sebagai row terakhir (close masih bergerak). Cache harus berisi candle
    # complete saja, kalau tidak run berikutnya baca harga stale dari mid-candle.
    if len(df) > 1:
        last_close_time = df.index[-1] + pd.Timedelta(hours=4)
        if last_close_time > pd.Timestamp.utcnow().tz_localize(None):
            df = df.iloc[:-1]

    funding = fetch_funding_rate(exchange, BTC_SYMBOL, days=days)
    ls      = fetch_long_short_ratio(exchange, BTC_SYMBOL, days=days)
    oi      = fetch_open_interest(exchange, BTC_SYMBOL, days=days)

    for extra in (funding, ls, oi):
        if not extra.empty:
            df = df.join(extra, how='left')

    df = df.ffill(limit=2).bfill(limit=2)
    df.to_parquet(CACHE_PATH)
    return df


if __name__ == '__main__':
    df = fetch_btc_research_data(force_refresh=False)
    print(f"shape={df.shape}, columns={df.columns.tolist()}")
    print(df.tail(3))
    for col in ('funding_rate', 'long_short_ratio', 'open_interest'):
        if col in df.columns:
            n_nan = df[col].isna().sum()
            print(f"  {col:20s} non-null={len(df)-n_nan}/{len(df)}")
