# main.py — Screener mode (relative ranking + conditional probability)

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import fetch_all_symbols, load_current_universe
from factors      import build_panel, FACTOR_COLS, TS_FACTOR_COLS, detect_regime
from ic_analysis  import cross_sectional_ic, build_composite_score, check_ic_drift

# Sentiment factors: selalu dimasukkan ke composite jika ICIR > 0
SENTIMENT_COLS = ['funding_signal', 'funding_mom', 'funding_extreme',
                  'ls_signal', 'ts_oi_buildup']
from screener     import get_latest_ranking, export_full_ranking
from report       import print_ic_table
import config


def _write_ts_factor_cols(new_cols: dict):
    """Tulis ulang TS_FACTOR_COLS di factors.py."""
    import re
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'factors.py')
    with open(path, 'r') as fh:
        content = fh.read()
    new_block = (
        "TS_FACTOR_COLS = {\n"
        f"    'trend'   : {new_cols['trend']},\n"
        f"    'reversal': {new_cols['reversal']},\n"
        "}"
    )
    updated = re.sub(r'TS_FACTOR_COLS = \{[^}]*\}', new_block, content, flags=re.DOTALL)
    with open(path, 'w') as fh:
        fh.write(updated)


def maybe_refresh_ts_factors(drifted: list):
    if not drifted:
        return
    print("\n  ⚠ IC DRIFT DETECTED:")
    for d in drifted:
        print(f"    {d['factor']:<24} ICIR {d['full_icir']:+.4f} → recent {d['recent_icir']:+.4f} "
              f"(ratio={d['ratio']:.2f})")
    print("  → Re-validasi faktor otomatis (factor_research_wf.py)...")
    try:
        from factor_research_wf import main as wf_validate
        new_cols = wf_validate()
        if new_cols and new_cols.get('trend') is not None:
            import factors as _factors
            _factors.TS_FACTOR_COLS['trend']    = new_cols['trend']
            _factors.TS_FACTOR_COLS['reversal'] = new_cols['reversal']
            _write_ts_factor_cols(new_cols)
            print(f"  ✓ TS_FACTOR_COLS diperbarui dan disimpan ke factors.py")
            print(f"    trend   : {new_cols['trend']}")
            print(f"    reversal: {new_cols['reversal']}")
    except Exception as e:
        print(f"  ✗ Re-validasi gagal: {e}")


def main(force_refresh: bool = False):
    print("\n" + "█"*60)
    print("  CRYPTO PERPETUAL FACTOR SCREENER — 4H INTRADAY")
    print(f"  Timeframe   : {config.TIMEFRAME}")
    print(f"  Forward     : {config.FORWARD_PERIOD} candle (~{config.FORWARD_PERIOD * 4}h)")
    print(f"  Universe    : top {config.TOP_N_SYMBOLS} USDT-M perp by volume")
    print("█"*60)

    # ── STEP 1: Fetch Data ─────────────────────────────────
    print("\n[1/4] Fetching market data...")
    raw_data, btc_df = fetch_all_symbols(force_refresh=True,
                                          force_refresh_universe=force_refresh)

    if len(raw_data) < 5:
        print("  ✗ Tidak cukup data.")
        return

    print(f"  BTC regime data: {len(btc_df)} candles")
    current_regime = detect_regime(btc_df)

    # TREND + BTC di bawah MA → arah trend turun, momentum long berbahaya → pakai reversal
    if current_regime == 'trend' and len(btc_df) >= config.BTC_REGIME_MA:
        btc_close = btc_df['close'].iloc[-1]
        btc_ma    = btc_df['close'].rolling(config.BTC_REGIME_MA).mean().iloc[-1]
        if btc_close < btc_ma:
            print(f"  ⚠ TREND tapi BTC < MA{config.BTC_REGIME_MA} "
                  f"({btc_close:,.0f} < {btc_ma:,.0f}) → override ke REVERSAL")
            current_regime = 'reversal'

    print(f"  Regime         : {current_regime.upper()} "
          f"(vol-based, window={config.REGIME_TREND_WINDOW}c)")

    forward_period = (config.FORWARD_PERIOD_TREND
                      if current_regime == 'trend'
                      else config.FORWARD_PERIOD_REVERSAL)
    print(f"  Forward period : {forward_period} candles (~{forward_period * 4}h) [{current_regime}]")

    # ── STEP 1.5: BTC Directional Gate ─────────────────────
    btc_signal = None
    if config.BTC_GATE_ENABLED:
        print("\n[1.5/4] Computing BTC directional signal...")
        try:
            from btc_research.btc_signal import compute_live_btc_signal
            btc_signal = compute_live_btc_signal(
                train_days   = config.BTC_GATE_TRAIN_DAYS,
                fps          = config.BTC_GATE_FPS,
                ic_threshold = config.BTC_GATE_IC_THRESHOLD,
                long_thr     = config.BTC_GATE_LONG_THR,
                short_thr    = config.BTC_GATE_SHORT_THR,
            )
            forced_tag = ' [FORCED via env]' if btc_signal.get('forced') else ''
            print(f"  BTC composite : {btc_signal['composite']:+.3f}  → MODE: "
                  f"{btc_signal['mode']}{forced_tag}")
            meta = btc_signal.get('factors_meta', {})
            if meta:
                print(f"  Factors used  : {len(meta)} (sign · best_fp · IC train)")
                for f, m in meta.items():
                    sign_c = '+' if m['sign'] > 0 else '-'
                    print(f"    {f:<18} {sign_c}   fp={m['best_fp']:>2}   "
                          f"ic={m['best_ic']:+.3f}")
            if btc_signal.get('btc_close'):
                print(f"  BTC last      : ${btc_signal['btc_close']:,.1f} @ {btc_signal['last_dt']}")
        except Exception as e:
            print(f"  ✗ BTC signal gagal: {e} — fallback gate=BULL")
            btc_signal = {'mode': 'BULL', 'composite': None, 'signs': {}, 'forced': False}

    # ── STEP 2: Build Factors ──────────────────────────────
    print("\n[2/4] Building factors...")
    panel = build_panel(raw_data, btc_df=btc_df, forward_period=forward_period)

    print(f"  Panel shape  : {panel.shape}")
    print(f"  Time range   : {panel['datetime'].min()} → {panel['datetime'].max()}")
    print(f"  Symbols      : {panel['symbol'].nunique()}")

    # Regime summary
    if 'btc_regime' in panel.columns:
        latest_regime = panel.sort_values('datetime').groupby('symbol')['btc_regime'].last()
        bull_pct = (latest_regime >= 0.5).mean()
        print(f"  Current regime: {'BULL 🟢' if bull_pct > 0.5 else 'BEAR 🔴'} "
              f"(BTC {'above' if bull_pct > 0.5 else 'below'} MA{config.BTC_REGIME_MA})")

    # ── STEP 3: IC + Composite Score ──────────────────────
    print("\n[3/4] Computing IC & building composite score...")
    available_factors = [
        col for col in FACTOR_COLS
        if col in panel.columns
        and panel[col].notna().sum() > config.IC_ROLLING_WINDOW * 5
    ]

    ic_summary, ic_ts = cross_sectional_ic(
        panel,
        factor_cols=available_factors,
        date_col='datetime',
        forward_col='forward_1d'
    )

    print_ic_table(ic_summary)

    # IC drift check — faktor yang drifted dibuang dari composite
    drifted = check_ic_drift(ic_ts, ic_summary)
    maybe_refresh_ts_factors(drifted)

    if drifted:
        drifted_names = {d['factor'] for d in drifted}
        ic_summary_composite = ic_summary[~ic_summary['factor'].isin(drifted_names)].copy()
        print(f"  ℹ Composite: exclude {len(drifted_names)} faktor drifted "
              f"({', '.join(drifted_names)})")
    else:
        ic_summary_composite = ic_summary

    # Composite hanya pakai factor ICIR positif & tidak drifted
    panel_scored, weights = build_composite_score(
        panel, ic_summary_composite, sentiment_cols=SENTIMENT_COLS
    )

    # ── STEP 4: Screener Output ────────────────────────────
    print("\n[4/4] Generating screener ranking...")
    current_universe = load_current_universe()
    print(f"  Current universe (liquid sekarang): {len(current_universe)} coins")
    latest_ranking = get_latest_ranking(
        panel_scored, top_n=20, score_col='composite_score',
        universe_symbols=current_universe or None,
        regime=current_regime,
        btc_signal=btc_signal,
    )

    os.makedirs("./output", exist_ok=True)
    export_full_ranking(panel_scored, output_path="./output/screener_ranking.csv")
    ic_summary.to_csv("./output/ic_summary_4h.csv", index=False)
    if not latest_ranking.empty:
        latest_ranking.to_csv("./output/latest_ranking.csv", index=False)

    print("\n" + "█"*60)
    print("  ✓ SELESAI")
    print("  Output:")
    print("    ./output/latest_ranking.csv   ← ranking terbaru")
    print("    ./output/screener_ranking.csv ← history semua candle")
    print("    ./output/ic_summary_4h.csv    ← IC per factor")
    print("█"*60 + "\n")

    return {
        'panel'         : panel_scored,
        'ic_summary'    : ic_summary,
        'latest_ranking': latest_ranking,
        'weights'       : weights
    }


if __name__ == "__main__":
    import sys
    force   = '--refresh' in sys.argv
    results = main(force_refresh=force)
