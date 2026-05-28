# btc_trade_mae_analysis.py — Analisis distribusi MAE/MFE per trade.
#
# Untuk setiap trade (signal entry → signal exit), track:
#   MAE = max adverse excursion (terjauh melawan posisi, dalam unit ATR_at_entry)
#   MFE = max favorable excursion (terjauh searah, dalam unit ATR_at_entry)
# Output: distribusi MAE per kelompok (winners vs losers) + simulasi SL impact.
#
# Insight: kalau LOSERS punya MAE tinggi & WINNERS MAE rendah → SL bisa filter
# losers tanpa cut winners. Kalau keduanya tinggi → SL menyakitkan strategi.

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_research.btc_data        import fetch_btc_research_data
from btc_research.btc_factors     import build_btc_factors
from btc_research.btc_backtest_wf import (
    TRAIN_DAYS, TEST_DAYS, STEP_DAYS, FP_TARGET, IC_THRESHOLD,
    CANDLES_PER_DAY, MIN_FACTORS, APPLY_FUNDING_COST,
    select_factors_train, composite_on, positions_from, compute_atr,
)


SIGNAL_THRESHOLD = 1.5      # focal config dari best walk-forward result
MODE             = 'long_short'


def extract_trades(close, high, low, atr, pos):
    """Walk through pos series, record each contiguous-direction trade with MAE/MFE."""
    n = len(pos)
    trades = []
    in_pos    = False
    entry_idx = None
    entry_px  = None
    direction = 0
    atr_entry = None
    mae       = 0.0
    mfe       = 0.0

    for i in range(n):
        p = float(pos.iloc[i])

        if p != 0 and not in_pos:
            in_pos    = True
            entry_idx = i
            entry_px  = float(close.iloc[i])
            direction = p
            atr_entry = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else np.nan
            mae = mfe = 0.0
            continue

        if in_pos:
            # Update MAE/MFE pakai intra-candle high/low
            h_i, l_i = float(high.iloc[i]), float(low.iloc[i])
            if direction > 0:
                adv = (entry_px - l_i) / entry_px
                fav = (h_i - entry_px) / entry_px
            else:
                adv = (h_i - entry_px) / entry_px
                fav = (entry_px - l_i) / entry_px
            mae = max(mae, adv)
            mfe = max(mfe, fav)

            # Exit kondisi: pos = 0 atau direction flip
            exit_now = (p == 0) or (np.sign(p) != np.sign(direction))
            if exit_now:
                exit_px = float(close.iloc[i])
                final_pnl = direction * (exit_px / entry_px - 1.0)
                if atr_entry and atr_entry > 0:
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx' : i,
                        'duration' : i - entry_idx,
                        'direction': int(direction),
                        'entry_px' : entry_px,
                        'atr_entry': atr_entry,
                        'mae_pct'  : mae,
                        'mfe_pct'  : mfe,
                        'mae_atr'  : mae * entry_px / atr_entry,
                        'mfe_atr'  : mfe * entry_px / atr_entry,
                        'pnl_pct'  : final_pnl,
                        'win'      : final_pnl > 0,
                    })
                in_pos = False
                entry_idx = None
                entry_px = None
                direction = 0
                # Kalau direction flip (sign ganti, bukan 0), buka trade baru
                if p != 0 and np.sign(p) != 0:
                    in_pos    = True
                    entry_idx = i
                    entry_px  = float(close.iloc[i])
                    direction = p
                    atr_entry = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else np.nan
                    mae = mfe = 0.0

    return pd.DataFrame(trades)


def walk_forward_trades(btc_df, fdf, threshold, mode):
    """Replay walk-forward dan accumulate trades dari seluruh OOS segments."""
    close = btc_df['close']
    high  = btc_df['high']
    low   = btc_df['low']
    atr   = compute_atr(btc_df)

    train_n = TRAIN_DAYS * CANDLES_PER_DAY
    test_n  = TEST_DAYS  * CANDLES_PER_DAY
    step_n  = STEP_DAYS  * CANDLES_PER_DAY
    n       = len(fdf)
    drop_tail = FP_TARGET

    all_trades = []
    start = train_n
    while start + test_n <= n:
        train_slice = fdf.iloc[start - train_n : start - drop_tail]
        test_slice  = fdf.iloc[start : start + test_n]
        signs = select_factors_train(train_slice, fp=FP_TARGET, threshold=IC_THRESHOLD)
        if len(signs) < MIN_FACTORS:
            start += step_n
            continue
        comp = composite_on(test_slice, signs)
        pos  = positions_from(comp, threshold, mode)

        seg_close = close.loc[test_slice.index]
        seg_high  = high.loc[test_slice.index]
        seg_low   = low.loc[test_slice.index]
        seg_atr   = atr.loc[test_slice.index]

        tr = extract_trades(seg_close, seg_high, seg_low, seg_atr, pos)
        all_trades.append(tr)
        start += step_n

    return pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()


def print_distribution(label, series, pcts=[10, 25, 50, 75, 90, 95]):
    s = series.dropna()
    if len(s) == 0:
        print(f"  {label:<20} (no data)")
        return
    pct_vals = np.percentile(s, pcts)
    print(f"  {label:<20} n={len(s):>4}  mean={s.mean():.2f}  std={s.std():.2f}  "
          + "  ".join(f"p{p}={v:.2f}" for p, v in zip(pcts, pct_vals)))


def simulate_sl_impact(trades, sl_atr_mults=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0]):
    """Untuk setiap SL multiplier, hitung % trade yang akan hit SL + impact ke total P&L."""
    if trades.empty:
        return
    print(f"\n=== Simulasi SL: % trade hit + P&L impact ===")
    print(f"  {'SL×ATR':>8}{'%hit':>8}{'%hit_W':>8}{'%hit_L':>8}"
          f"{'gross_PnL%':>13}{'net_PnL%':>11}{'savings':>10}")
    gross_pnl = trades['pnl_pct'].sum()
    for k in sl_atr_mults:
        hit_mask = trades['mae_atr'] >= k
        pct_hit  = hit_mask.mean() * 100
        # P&L kalau stopped: -k×ATR/entry_px (loss = k × ATR sebagai pct dari entry)
        sl_loss_pct = -k * trades['atr_entry'] / trades['entry_px']
        # Final P&L per trade kalau SL: min(actual_pnl, sl_loss_pct)
        # Kalau hit: gunakan sl_loss_pct (worst case kena duluan).
        # Kalau ga hit: pnl_pct asli.
        net_pnl_each = np.where(hit_mask, sl_loss_pct, trades['pnl_pct'])
        net_pnl = net_pnl_each.sum()
        # winners hit (cut prematurely): bad
        winners_hit = hit_mask & trades['win']
        losers_hit  = hit_mask & (~trades['win'])
        pct_hit_w = winners_hit.sum() / trades['win'].sum() * 100 if trades['win'].sum() > 0 else 0
        pct_hit_l = losers_hit.sum() / (~trades['win']).sum() * 100 if (~trades['win']).sum() > 0 else 0
        savings = (net_pnl - gross_pnl) * 100
        print(f"  {k:>8.1f}{pct_hit:>7.1f}%{pct_hit_w:>7.1f}%{pct_hit_l:>7.1f}%"
              f"{gross_pnl*100:>+12.2f}%{net_pnl*100:>+10.2f}%{savings:>+9.2f}%")


def main():
    print(f"Loading BTC data (1y)...")
    btc_df = fetch_btc_research_data(force_refresh=False)
    print(f"  shape={btc_df.shape}, range={btc_df.index.min()} → {btc_df.index.max()}")

    fdf = build_btc_factors(btc_df)
    print(f"  factor panel shape={fdf.shape}")
    print(f"\nWalk-forward config: train={TRAIN_DAYS}d, test={TEST_DAYS}d, "
          f"FP={FP_TARGET}, mode={MODE}, signal_threshold={SIGNAL_THRESHOLD}")

    trades = walk_forward_trades(btc_df, fdf, SIGNAL_THRESHOLD, MODE)
    print(f"\nExtracted {len(trades)} trades total")

    if trades.empty:
        return

    win = trades[trades['win']]
    los = trades[~trades['win']]
    print(f"  Winners: {len(win)} ({len(win)/len(trades)*100:.1f}%)  avg_pnl={win['pnl_pct'].mean()*100:+.2f}%")
    print(f"  Losers : {len(los)} ({len(los)/len(trades)*100:.1f}%)  avg_pnl={los['pnl_pct'].mean()*100:+.2f}%")
    print(f"  Total PnL (gross, all trades): {trades['pnl_pct'].sum()*100:+.2f}%")
    print(f"  Avg duration: {trades['duration'].mean():.1f} candles ({trades['duration'].mean()*4:.1f}h)")

    print(f"\n=== MAE distribution (in units of ATR-at-entry) ===")
    print_distribution('All trades', trades['mae_atr'])
    print_distribution('Winners', win['mae_atr'])
    print_distribution('Losers', los['mae_atr'])

    print(f"\n=== MFE distribution (in units of ATR-at-entry) ===")
    print_distribution('All trades', trades['mfe_atr'])
    print_distribution('Winners', win['mfe_atr'])
    print_distribution('Losers', los['mfe_atr'])

    simulate_sl_impact(trades)

    # Rekomendasi sweet spot
    print(f"\n=== Interpretation ===")
    median_mae_w = win['mae_atr'].median() if len(win) else np.nan
    median_mae_l = los['mae_atr'].median() if len(los) else np.nan
    p90_mae_w = np.percentile(win['mae_atr'], 90) if len(win) else np.nan
    print(f"  Median MAE winners:    {median_mae_w:.2f}× ATR  (signal sukses walau experience adverse ini)")
    print(f"  Median MAE losers:     {median_mae_l:.2f}× ATR  (signal gagal, harus stop sebelum kena ini)")
    print(f"  p90 MAE winners:       {p90_mae_w:.2f}× ATR  (90% winners tidak melebihi ini)")
    print(f"  → Sweet spot SL ≈ p90 winners atau median losers (mana yang lebih tinggi)")


if __name__ == '__main__':
    main()
