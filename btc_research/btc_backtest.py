# btc_backtest.py — Single-asset BTC timing backtest dari composite factor.
#
# Composite = signed equal-weight average dari faktor yang robust di 4-bulan window
# (sign dari arah IC; |IC|≥0.05 & sign-stable H1/H2). Position di-update tiap candle,
# cost diaplikasikan saat position berubah.
#
# CAVEAT: pemilihan & sign factor in-sample (look-ahead). First pass diagnostic.

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_research.btc_data    import fetch_btc_research_data
from btc_research.btc_factors import build_btc_factors, FORWARD_PERIODS


# Factors dipilih dari robust list di FP=6 (24h) — sweet spot horizon
# (sign = arah IC, lihat hasil btc_factor_research.py)
COMPOSITE_FACTORS = {
    'mom_3d'        : -1,
    'mom_1d'        : -1,
    'mom_12h'       : -1,
    'mom_7d'        : -1,
    'stoch_rev'     : +1,
    'price_vs_wvwap': +1,
}

CANDLES_PER_YEAR = 6 * 365      # 4H × 6/day × 365


def build_composite(fdf: pd.DataFrame) -> pd.Series:
    """Equal-weight average of signed factor z-scores."""
    parts = []
    for f, sign in COMPOSITE_FACTORS.items():
        if f in fdf.columns:
            parts.append(sign * fdf[f])
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def run_backtest(close: pd.Series, composite: pd.Series, threshold: float,
                 slippage: float, mode: str) -> dict:
    """
    Continuous: setiap candle, position = signal(composite, threshold).
    P&L per candle = position(t-1) × ret(t).
    Cost diaplikasikan setiap position change (turnover × slippage).
    """
    sig = composite.copy()
    ret = close.pct_change()

    if mode == 'long_flat':
        pos = np.where(sig > threshold, 1.0, 0.0)
    elif mode == 'long_short':
        pos = np.where(sig > threshold, 1.0,
              np.where(sig < -threshold, -1.0, 0.0))
    else:
        raise ValueError(f"unknown mode: {mode}")

    pos = pd.Series(pos, index=sig.index).fillna(0)
    # position taken AT t holds until t+1's close → P&L of period t→t+1 = pos[t] × ret[t+1]
    pos_lag = pos.shift(1).fillna(0)

    turn  = (pos - pos.shift(1)).abs().fillna(0)
    cost  = turn * slippage
    pnl   = pos_lag * ret - cost

    # Drop the window of NaN factor rows at the start
    mask  = composite.notna() & ret.notna()
    pnl   = pnl[mask]
    pos_lag = pos_lag[mask]
    ret_m   = ret[mask]

    if len(pnl) < 10:
        return None

    equity = (1 + pnl).cumprod()
    total  = equity.iloc[-1] - 1
    sharpe = pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(CANDLES_PER_YEAR)
    dd     = (equity / equity.cummax() - 1).min()

    in_mkt = (pos_lag != 0)
    n_in   = int(in_mkt.sum())
    pct_in = n_in / len(pos_lag)

    # win rate per candle dalam posisi
    if n_in > 0:
        wins = ((pos_lag * ret_m) > 0) & in_mkt
        win_rate = wins.sum() / n_in
        per_candle_in = pnl[in_mkt].mean()
    else:
        win_rate = np.nan
        per_candle_in = np.nan

    n_changes = int((turn[mask] > 0).sum())

    return {
        'mode': mode, 'threshold': threshold, 'slippage': slippage,
        'total_return': total, 'sharpe': sharpe, 'max_dd': dd,
        'pct_time_in_mkt': pct_in, 'n_position_changes': n_changes,
        'win_rate_in_mkt': win_rate, 'avg_pnl_per_candle_in': per_candle_in,
        'equity': equity,
    }


def buy_hold(close: pd.Series, composite: pd.Series) -> dict:
    """Benchmark: BTC buy-and-hold, dengan satu kali entry cost & exit cost @0% (proxy)."""
    ret  = close.pct_change()
    mask = composite.notna() & ret.notna()
    pnl  = ret[mask]
    equity = (1 + pnl).cumprod()
    return {
        'mode': 'buy_hold', 'threshold': None, 'slippage': 0.0,
        'total_return': equity.iloc[-1] - 1,
        'sharpe': pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(CANDLES_PER_YEAR),
        'max_dd': (equity / equity.cummax() - 1).min(),
        'pct_time_in_mkt': 1.0, 'n_position_changes': 1,
        'win_rate_in_mkt': (pnl > 0).mean(), 'avg_pnl_per_candle_in': pnl.mean(),
        'equity': equity,
    }


def _fmt_pct(x, p=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '   nan'
    return f"{x*100:+{6}.{p}f}%"


def _fmt(x, w=7, p=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ' ' * (w - 3) + 'nan'
    return f"{x:>{w}.{p}f}"


def print_table(rows: list, title: str):
    print(f"\n=== {title} ===")
    print(f"  {'mode':<11}{'thr':>5}{'slip':>7}  {'totRet':>8}{'Sharpe':>7}"
          f"{'maxDD':>8}{'pctIn':>6}{'wins':>6}{'#chg':>5}{'pnl/c':>9}")
    for r in rows:
        thr = '-' if r['threshold'] is None else f"{r['threshold']:.1f}"
        print(
            f"  {r['mode']:<11}{thr:>5}{_fmt_pct(r['slippage'], 2):>7}  "
            f"{_fmt_pct(r['total_return']):>8}{_fmt(r['sharpe'])}"
            f"{_fmt_pct(r['max_dd'])}"
            f"{_fmt_pct(r['pct_time_in_mkt'], 0):>6}"
            f"{_fmt_pct(r['win_rate_in_mkt'], 0):>6}"
            f"{r['n_position_changes']:>5}"
            f"{_fmt_pct(r['avg_pnl_per_candle_in'], 3):>9}"
        )


def main():
    print("Loading BTC data...")
    btc_df = fetch_btc_research_data(force_refresh=False)
    print(f"  shape={btc_df.shape}  range={btc_df.index.min()} → {btc_df.index.max()}")

    fdf = build_btc_factors(btc_df)
    close = btc_df['close']
    composite = build_composite(fdf)

    print(f"\nComposite factors (in-sample sign, equal-weight):")
    for f, s in COMPOSITE_FACTORS.items():
        in_fdf = f in fdf.columns
        print(f"  {f:<18} sign={s:+d}  in_panel={in_fdf}")

    # Composite distribution
    desc = composite.describe()
    print(f"\nComposite z-score stats: mean={desc['mean']:+.3f} std={desc['std']:.3f} "
          f"min={desc['min']:+.2f} max={desc['max']:+.2f}")

    rows = [buy_hold(close, composite)]

    THRESHOLDS = [0.0, 0.5, 1.0]
    SLIPS      = [0.0, 0.0005, 0.001, 0.002]   # 0, 5bp, 10bp, 20bp per side
    MODES      = ['long_flat', 'long_short']

    for mode in MODES:
        for thr in THRESHOLDS:
            for slip in SLIPS:
                r = run_backtest(close, composite, threshold=thr,
                                 slippage=slip, mode=mode)
                if r is not None:
                    rows.append(r)

    print_table(rows, "Backtest (composite from FP=6 robust factors, continuous)")

    # Highlight best Sharpe by mode
    print("\n=== Best Sharpe per mode (at varied slippage) ===")
    for mode in ['buy_hold', 'long_flat', 'long_short']:
        candidates = [r for r in rows if r['mode'] == mode]
        if not candidates:
            continue
        best = max(candidates, key=lambda r: r['sharpe'])
        thr = '-' if best['threshold'] is None else f"thr={best['threshold']}"
        slip = f"slip={best['slippage']*100:.2f}%"
        print(f"  {mode:<11} {thr:<10} {slip:<14} "
              f"Sharpe={best['sharpe']:+.2f}  ret={best['total_return']*100:+.1f}%  "
              f"DD={best['max_dd']*100:+.1f}%")


if __name__ == '__main__':
    main()
