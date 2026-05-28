# btc_backtest_wf.py — Walk-forward backtest: composite weights refit per window.
#
# Untuk setiap test window:
#   1. Hitung IC tiap factor di train (Spearman, OOS-aman: train hanya berisi data
#      sebelum test start, dengan label forward_ret_FP yang sudah valid).
#   2. Pilih factor dengan |IC train| ≥ IC_THRESHOLD; sign = sign(IC train).
#   3. Composite test = mean(signed factor z-scores) di test window.
#   4. Continuous long-flat / long-short di test window dengan threshold + slippage.
#   5. Concatenate OOS P&L semua window → stats agregat.
#
# Z-scores per-candle adalah rolling-window causal (window=120 di build_btc_factors),
# jadi composite test tidak look-ahead — knowledge yang ter-transfer dari train hanya
# (a) factor mana yang dipakai, (b) sign-nya.

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_research.btc_data    import fetch_btc_research_data
from btc_research.btc_factors import build_btc_factors, FACTOR_COLS, FACTOR_GROUP


# ── Walk-forward config ───────────────────────────────────────
CANDLES_PER_DAY  = 6
TRAIN_DAYS       = 45
TEST_DAYS        = 10
STEP_DAYS        = 10                # = TEST_DAYS → non-overlap OOS
FP_TARGET        = 6                  # composite untuk 24h horizon
IC_THRESHOLD     = 0.05               # selection: |IC train| ≥ ini
MIN_FACTORS      = 1                  # minimum factor untuk membentuk composite

# Reuse hasil best dari in-sample backtest
SIGNAL_THRESHOLDS = [1.0, 1.5, 2.0]   # sweep — semakin tinggi = semakin selektif
SLIPPAGES         = [0.0, 0.0005, 0.001, 0.002]
MODES             = ['long_flat', 'long_short']

# Funding holding cost: posisi long bayar fr>0, short bayar fr<0 (proxy: fr × 0.5 per
# 4H candle karena funding di-pay tiap 8h). Toggle via APPLY_FUNDING_COST.
APPLY_FUNDING_COST = True

CANDLES_PER_YEAR  = 6 * 365


# ── OOS factor selection ─────────────────────────────────────

def select_factors_train(fdf_train: pd.DataFrame, fp: int,
                         threshold: float) -> dict:
    """Return {factor: +1 or -1} untuk factor dengan |IC train| ≥ threshold."""
    target = fdf_train[f'forward_ret_{fp}']
    chosen = {}
    for f in FACTOR_COLS:
        if f not in fdf_train.columns:
            continue
        x = fdf_train[f]
        mask = x.notna() & target.notna()
        if mask.sum() < 50:
            continue
        ic, _ = spearmanr(x[mask], target[mask])
        if np.isnan(ic) or abs(ic) < threshold:
            continue
        chosen[f] = +1.0 if ic > 0 else -1.0
    return chosen


def composite_on(fdf_segment: pd.DataFrame, signs: dict) -> pd.Series:
    if not signs:
        return pd.Series(np.nan, index=fdf_segment.index)
    parts = [signs[f] * fdf_segment[f] for f in signs]
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


# ── Per-window backtest segment ──────────────────────────────

def positions_from(composite: pd.Series, threshold: float, mode: str) -> pd.Series:
    if mode == 'long_flat':
        pos = np.where(composite > threshold, 1.0, 0.0)
    elif mode == 'long_short':
        pos = np.where(composite > threshold, 1.0,
              np.where(composite < -threshold, -1.0, 0.0))
    else:
        raise ValueError(mode)
    return pd.Series(pos, index=composite.index).fillna(0.0)


def compute_atr(btc_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR(14) di 4H candle untuk trailing stop."""
    h, l, c = btc_df['high'], btc_df['low'], btc_df['close']
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period // 2).mean()


def apply_bracket_stop(pos_signal: pd.Series, close: pd.Series,
                       high: pd.Series, low: pd.Series,
                       atr: pd.Series, sl_mult: float,
                       tp_mult: float = None,
                       init_pos: float = 0.0) -> tuple:
    """ATR bracket: SL = entry ± sl_mult×ATR. Optional TP = entry ± tp_mult×ATR.
    Kalau tp_mult=None → SL-only (no take-profit; let signal-flip jadi exit profit).
    Exit pada touch pertama intra-candle (pakai high/low). OR-logic dengan signal.
    Setelah bracket hit, BLOCK re-entry sampai signal direction berubah.

    Returns: (pos_modified, pnl_override).
      pnl_override[i] = exact exit P&L untuk candle i (NaN = pakai close-to-close).
    """
    n   = len(pos_signal)
    pos = pos_signal.copy().astype(float)
    pnl_ov = pd.Series(np.nan, index=pos.index)
    if n == 0 or atr is None:
        return pos, pnl_ov

    held       = float(init_pos)
    entry_px   = None
    sl = tp    = None
    blocked    = False
    last_sig_sign = float(np.sign(init_pos))

    for i in range(n):
        sig      = float(pos_signal.iloc[i])
        sig_sign = float(np.sign(sig))
        if sig_sign != last_sig_sign:
            blocked = False
        last_sig_sign = sig_sign

        if blocked:
            held = 0.0
            pos.iloc[i] = 0.0
            continue

        if sig == 0:
            held = 0.0
            entry_px = sl = tp = None
            pos.iloc[i] = 0.0
            continue

        if held == 0:
            held     = sig
            entry_px = float(close.iloc[i])
            a        = atr.iloc[i] if i < len(atr) else np.nan
            if not pd.isna(a):
                if held > 0:
                    sl = entry_px - sl_mult * a
                    tp = entry_px + tp_mult * a if tp_mult is not None else None
                else:
                    sl = entry_px + sl_mult * a
                    tp = entry_px - tp_mult * a if tp_mult is not None else None
            else:
                sl = tp = None
            pos.iloc[i] = held
        else:
            h_i = float(high.iloc[i])
            l_i = float(low.iloc[i])
            hit_at = None
            if sl is not None:
                if held > 0:
                    if l_i <= sl:
                        hit_at = sl
                    elif tp is not None and h_i >= tp:
                        hit_at = tp
                else:
                    if h_i >= sl:
                        hit_at = sl
                    elif tp is not None and l_i <= tp:
                        hit_at = tp

            if hit_at is not None:
                prev_close = float(close.iloc[i-1]) if i > 0 else entry_px
                pnl_ov.iloc[i] = held * (hit_at / prev_close - 1.0)
                held     = 0.0
                entry_px = sl = tp = None
                blocked  = True
                pos.iloc[i] = 0.0
            else:
                pos.iloc[i] = held

    return pos, pnl_ov


def apply_trailing_stop(pos_signal: pd.Series, close: pd.Series,
                        atr: pd.Series, mult: float,
                        init_pos: float = 0.0) -> pd.Series:
    """OR-logic trailing stop. Exit kalau signal flip ATAU trailing stop kena.
    Setelah stop hit, BLOCK re-entry sampai signal direction berubah (anti-whipsaw).
    Stop = high-watermark price ± mult × ATR(t), updated per-candle.
    """
    n = len(pos_signal)
    out = pos_signal.copy().astype(float)
    if n == 0 or atr is None:
        return out

    held       = float(init_pos)
    trail_stop = None
    blocked    = False
    last_sig_sign = float(np.sign(init_pos))

    for i in range(n):
        sig      = float(pos_signal.iloc[i])
        sig_sign = float(np.sign(sig))
        c        = float(close.iloc[i])
        a        = atr.iloc[i] if i < len(atr) else np.nan

        # Unblock saat signal direction berubah
        if sig_sign != last_sig_sign:
            blocked = False
        last_sig_sign = sig_sign

        if blocked:
            held = 0.0
            out.iloc[i] = 0.0
            continue

        if sig == 0:
            held = 0.0
            trail_stop = None
            out.iloc[i] = 0.0
            continue

        if held == 0:
            # Entry baru
            held = sig
            if not pd.isna(a):
                trail_stop = c - mult * a if sig > 0 else c + mult * a
            else:
                trail_stop = None
        else:
            # Update trailing stop & cek
            if not pd.isna(a) and trail_stop is not None:
                if held > 0:
                    trail_stop = max(trail_stop, c - mult * a)
                    if c <= trail_stop:
                        held = 0.0
                        trail_stop = None
                        blocked = True
                        out.iloc[i] = 0.0
                        continue
                else:
                    trail_stop = min(trail_stop, c + mult * a)
                    if c >= trail_stop:
                        held = 0.0
                        trail_stop = None
                        blocked = True
                        out.iloc[i] = 0.0
                        continue

        out.iloc[i] = held

    return out


def pnl_segment(close: pd.Series, composite: pd.Series,
                threshold: float, slippage: float, mode: str,
                prev_pos: float = 0.0,
                funding: pd.Series = None,
                atr: pd.Series = None,
                trail_atr_mult: float = None,
                bracket_atr_mult: float = None,
                sl_only_atr_mult: float = None,
                high: pd.Series = None,
                low: pd.Series = None) -> tuple:
    """Continuous: pos[t] di-decide saat candle t, P&L = pos[t-1]*ret[t].
    Optional ATR-based exit (mutually exclusive):
      - trail_atr_mult     : trailing stop close-based
      - bracket_atr_mult   : bracket SL+TP RR 1:1, intra-candle touch
      - sl_only_atr_mult   : SL only (no TP), intra-candle touch
    Plus exit pada signal-flip (always active)."""
    exit_rules_set = sum(x is not None for x in (trail_atr_mult, bracket_atr_mult, sl_only_atr_mult))
    if exit_rules_set > 1:
        raise ValueError("exit rules mutually exclusive — pilih salah satu")

    pos_sig = positions_from(composite, threshold, mode)
    pnl_override = None

    if trail_atr_mult is not None and atr is not None:
        pos = apply_trailing_stop(pos_sig, close, atr, trail_atr_mult,
                                  init_pos=prev_pos)
    elif bracket_atr_mult is not None and atr is not None and high is not None:
        pos, pnl_override = apply_bracket_stop(pos_sig, close, high, low, atr,
                                               sl_mult=bracket_atr_mult,
                                               tp_mult=bracket_atr_mult,
                                               init_pos=prev_pos)
    elif sl_only_atr_mult is not None and atr is not None and high is not None:
        pos, pnl_override = apply_bracket_stop(pos_sig, close, high, low, atr,
                                               sl_mult=sl_only_atr_mult,
                                               tp_mult=None,
                                               init_pos=prev_pos)
    else:
        pos = pos_sig

    ret      = close.pct_change()

    pos_prev = pos.shift(1)
    pos_prev.iloc[0] = prev_pos
    turn     = (pos - pos_prev).abs().fillna(0)
    cost     = turn * slippage

    pos_lag  = pos.shift(1).fillna(prev_pos)
    pnl      = pos_lag * ret - cost

    # Override pnl pada candle bracket-exit (exact P&L dari entry ke SL/TP price)
    if pnl_override is not None:
        mask = pnl_override.notna()
        pnl  = pnl.where(~mask, pnl_override - cost)

    if funding is not None:
        fr            = funding.reindex(pos.index).ffill().fillna(0.0)
        funding_cost  = pos_lag * fr * 0.5
        pnl           = pnl - funding_cost

    last_pos = float(pos.iloc[-1]) if len(pos) else prev_pos
    return pnl, last_pos


def stats(pnl: pd.Series) -> dict:
    pnl = pnl.dropna()
    if len(pnl) < 5:
        return dict(total_return=np.nan, sharpe=np.nan, max_dd=np.nan,
                    win_rate=np.nan, n=len(pnl))
    equity = (1 + pnl).cumprod()
    return dict(
        total_return = equity.iloc[-1] - 1,
        sharpe       = pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(CANDLES_PER_YEAR),
        max_dd       = (equity / equity.cummax() - 1).min(),
        win_rate     = (pnl > 0).mean(),
        n            = int(len(pnl)),
    )


# ── Walk-forward driver ──────────────────────────────────────

def walk_forward(btc_df: pd.DataFrame, fdf: pd.DataFrame,
                 mode: str, sig_threshold: float, slippage: float,
                 apply_funding: bool = APPLY_FUNDING_COST,
                 trail_atr_mult: float = None,
                 bracket_atr_mult: float = None,
                 sl_only_atr_mult: float = None,
                 verbose: bool = False) -> tuple:
    close   = btc_df['close']
    need_hl = bracket_atr_mult is not None or sl_only_atr_mult is not None
    high    = btc_df['high']  if need_hl else None
    low     = btc_df['low']   if need_hl else None
    funding = btc_df['funding_rate'] if (apply_funding and 'funding_rate' in btc_df.columns) else None
    atr     = compute_atr(btc_df) if (trail_atr_mult or bracket_atr_mult or sl_only_atr_mult) else None
    train_n = TRAIN_DAYS * CANDLES_PER_DAY
    test_n  = TEST_DAYS  * CANDLES_PER_DAY
    step_n  = STEP_DAYS  * CANDLES_PER_DAY

    n = len(fdf)
    drop_tail = FP_TARGET

    all_pnl  = []
    logs     = []
    last_pos = 0.0
    start    = train_n

    while start + test_n <= n:
        train_slice = fdf.iloc[start - train_n : start - drop_tail]
        test_slice  = fdf.iloc[start : start + test_n]
        test_close  = close.iloc[max(start - 1, 0) : start + test_n]
        test_high   = high.iloc[max(start - 1, 0) : start + test_n] if high is not None else None
        test_low    = low.iloc[max(start - 1, 0) : start + test_n] if low is not None else None
        test_funding = funding.iloc[max(start - 1, 0) : start + test_n] if funding is not None else None
        test_atr     = atr.iloc[max(start - 1, 0) : start + test_n] if atr is not None else None

        signs = select_factors_train(train_slice, fp=FP_TARGET, threshold=IC_THRESHOLD)

        if len(signs) < MIN_FACTORS:
            pnl  = pd.Series(0.0, index=test_slice.index)
            if last_pos != 0.0:
                pnl.iloc[0] = -abs(last_pos) * slippage
                last_pos = 0.0
            sig_used = None
        else:
            comp = composite_on(test_slice, signs)
            pnl, last_pos = pnl_segment(test_close, comp, sig_threshold, slippage, mode,
                                        prev_pos=last_pos, funding=test_funding,
                                        atr=test_atr,
                                        trail_atr_mult=trail_atr_mult,
                                        bracket_atr_mult=bracket_atr_mult,
                                        sl_only_atr_mult=sl_only_atr_mult,
                                        high=test_high, low=test_low)
            pnl = pnl.loc[test_slice.index]
            sig_used = signs

        all_pnl.append(pnl)
        logs.append({
            'idx': len(logs),
            'train_start' : fdf.index[start - train_n],
            'test_start'  : test_slice.index[0],
            'test_end'    : test_slice.index[-1],
            'n_selected'  : len(signs),
            'signs'       : sig_used,
            'window_stats': stats(pnl),
        })

        start += step_n

    full = pd.concat(all_pnl) if all_pnl else pd.Series(dtype=float)
    return full, logs


# ── Reporting ────────────────────────────────────────────────

def _pct(x, p=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '   nan'
    return f"{x*100:+.{p}f}%"


def print_window_log(logs: list):
    print(f"\n=== Per-window OOS results ({len(logs)} windows) ===")
    print(f"  {'#':<3}{'test_start':<20}{'test_end':<20}{'sel':>4}  {'ret':>7}{'Shp':>6}{'DD':>7}  factors")
    for L in logs:
        s = L['window_stats']
        facs = L['signs']
        fac_str = ', '.join(
            f"{name}{('+' if sign>0 else '-')}" for name, sign in facs.items()
        ) if facs else '(none)'
        # truncate if too long
        if len(fac_str) > 70:
            fac_str = fac_str[:67] + '...'
        print(f"  {L['idx']:<3}{str(L['test_start']):<20}{str(L['test_end']):<20}"
              f"{L['n_selected']:>4}  {_pct(s['total_return'])}"
              f"{s['sharpe']:>6.2f}{_pct(s['max_dd'])}  {fac_str}")


def print_summary(rows: list, btc_oos_stats: dict, title: str = "Aggregate OOS stats"):
    print(f"\n=== {title} ===")
    print(f"  {'mode':<11}{'thr':>5}{'slip':>7}  {'totRet':>8}{'Sharpe':>7}{'maxDD':>8}{'wins':>6}{'n':>6}")
    print(f"  {'btc_hold':<11}{'-':>5}{'-':>7}  "
          f"{_pct(btc_oos_stats['total_return']):>8}{btc_oos_stats['sharpe']:>7.2f}"
          f"{_pct(btc_oos_stats['max_dd']):>8}"
          f"{_pct(btc_oos_stats['win_rate'], 0):>6}{btc_oos_stats['n']:>6}")
    for r in rows:
        s = r['stats']
        print(f"  {r['mode']:<11}{r['threshold']:>5.1f}{_pct(r['slippage'], 2):>7}  "
              f"{_pct(s['total_return']):>8}{s['sharpe']:>7.2f}{_pct(s['max_dd']):>8}"
              f"{_pct(s['win_rate'], 0):>6}{s['n']:>6}")


def print_factor_stability(all_logs: list):
    counter = Counter()
    sign_counter = {}
    total_windows = len(all_logs)
    for L in all_logs:
        if not L['signs']:
            continue
        for f, sign in L['signs'].items():
            counter[f] += 1
            sign_counter.setdefault(f, []).append(sign)

    print(f"\n=== Factor selection stability across {total_windows} OOS windows ===")
    print(f"  {'factor':<18}{'group':<10}{'picked':>8}{'sign+':>7}{'sign-':>7}")
    for f, n in counter.most_common():
        signs = sign_counter[f]
        n_pos = sum(1 for s in signs if s > 0)
        n_neg = sum(1 for s in signs if s < 0)
        print(f"  {f:<18}{FACTOR_GROUP.get(f, '?'):<10}"
              f"{n:>4}/{total_windows:<3}{n_pos:>7}{n_neg:>7}")


def buy_hold_oos(btc_df: pd.DataFrame, logs: list) -> dict:
    """B&H stats di union dari semua test windows (no compounding antar gap)."""
    pnl_segs = []
    for L in logs:
        s, e   = L['test_start'], L['test_end']
        seg    = btc_df['close'].loc[s:e]
        pnl_segs.append(seg.pct_change().dropna())
    full = pd.concat(pnl_segs) if pnl_segs else pd.Series(dtype=float)
    return stats(full)


def main():
    print("Loading BTC data...")
    btc_df = fetch_btc_research_data(force_refresh=False)
    print(f"  shape={btc_df.shape}  range={btc_df.index.min()} → {btc_df.index.max()}")
    print(f"  total days = {(btc_df.index[-1] - btc_df.index[0]).days}")

    fdf = build_btc_factors(btc_df)

    print(f"\nWalk-forward config: train={TRAIN_DAYS}d, test={TEST_DAYS}d, step={STEP_DAYS}d, "
          f"FP={FP_TARGET}, IC_threshold={IC_THRESHOLD}")
    print(f"Funding holding cost: {'ON' if APPLY_FUNDING_COST else 'OFF'}")
    print(f"Threshold sweep: {SIGNAL_THRESHOLDS}  ·  Slippage sweep: {SLIPPAGES}")

    ref_pnl, ref_logs = walk_forward(btc_df, fdf, mode='long_short',
                                     sig_threshold=1.0, slippage=0.0005,
                                     apply_funding=APPLY_FUNDING_COST)
    print(f"\nReference run: {len(ref_logs)} OOS windows")
    print_factor_stability(ref_logs)

    bh = buy_hold_oos(btc_df, ref_logs)

    # Exit variants — params (trail_mult, bracket_mult, sl_only_mult), pilih satu non-None
    EXIT_VARIANTS = [
        ('no-stop',       None, None, None),
        ('trail-1.5ATR',  1.5,  None, None),
        ('bracket-1.5RR1', None, 1.5, None),
        ('SL-only-2.0ATR', None, None, 2.0),
        ('SL-only-1.5ATR', None, None, 1.5),
    ]

    print(f"\n=== Exit-rule head-to-head (slip=5bp, funding ON) ===")
    print(f"  {'mode':<12}{'thr':>5}{'exit_rule':>17}  {'totRet':>8}{'Sharpe':>7}{'maxDD':>8}")
    for mode in MODES:
        for thr in SIGNAL_THRESHOLDS:
            for label, trail_m, bracket_m, sl_only_m in EXIT_VARIANTS:
                pnl, _ = walk_forward(btc_df, fdf, mode=mode, sig_threshold=thr,
                                      slippage=0.0005, apply_funding=APPLY_FUNDING_COST,
                                      trail_atr_mult=trail_m,
                                      bracket_atr_mult=bracket_m,
                                      sl_only_atr_mult=sl_only_m)
                s = stats(pnl)
                print(f"  {mode:<12}{thr:>5.1f}{label:>17}  "
                      f"{_pct(s['total_return']):>8}{s['sharpe']:>7.2f}{_pct(s['max_dd']):>8}")


if __name__ == '__main__':
    main()
