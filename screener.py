# screener.py — Relative ranking + adaptive TS gate + decision engine

import pandas as pd
import numpy as np
import config


def apply_ts_gate(
    snapshot: pd.DataFrame,
    ts_factor_cols,
    regime: str = 'reversal',
    threshold: float = None,
    min_signals: int = None,
) -> tuple:
    """
    Filter snapshot satu timestamp: coin lolos jika >= min_signals faktor TS
    melebihi |threshold| (z-score absolut).
    ts_factor_cols bisa dict {'trend': [...], 'reversal': [...]} atau list.
    Returns: (passed_df, blocked_df)
    """
    threshold   = threshold   or config.TS_SIGNAL_THRESHOLD
    min_signals = min_signals or config.TS_MIN_SIGNALS

    if isinstance(ts_factor_cols, dict):
        active_cols = ts_factor_cols.get(regime, ts_factor_cols.get('reversal', []))
    else:
        active_cols = ts_factor_cols

    available = [c for c in active_cols if c in snapshot.columns]
    if not available:
        print("  ⚠ TS gate: kolom TS tidak ditemukan, gate dilewati")
        return snapshot.copy(), pd.DataFrame()

    # Kalau faktor aktif <= min_signals, AND logic terlalu ketat — turunkan ke n-1
    effective_min = min(min_signals, max(1, len(available) - 1))
    if effective_min < min_signals:
        print(f"  ℹ TS gate: {len(available)} faktor aktif, min_signals turun {min_signals}→{effective_min}")
    min_signals = effective_min

    snap = snapshot.copy()
    snap['_ts_count']  = (snap[available].abs() >= threshold).sum(axis=1)
    snap['_ts_active'] = snap[available].apply(
        lambda row: [c for c in available if abs(row[c]) >= threshold], axis=1
    )

    passed  = snap[snap['_ts_count'] >= min_signals].copy()
    blocked = snap[snap['_ts_count'] <  min_signals].copy()
    return passed, blocked


def compute_trade_decision(row: pd.Series, regime: str = 'reversal',
                           btc_mode: str = 'BULL') -> tuple:
    """
    Sintesis keputusan LONG/SHORT/WAIT dari 4 sinyal + BTC directional gate.
    btc_mode = 'BULL'  → LONG diizinkan (existing logic)
    btc_mode = 'BEAR'  → LONG diblok; SHORT diizinkan untuk score sangat negatif
    btc_mode = 'NEUTRAL' → semua WAIT
    Returns: (direction: str, conviction: float, notes: list[str])
    """
    score = 0.0
    notes = []

    # 1. Composite score — IC-weighted factor signal
    comp = float(row.get('composite_score', 0) or 0)
    score += np.clip(comp, -1.5, 1.5)
    notes.append(f"comp={comp:+.3f}")

    # 2. Funding rate — contrarian: positif = longs overpay = SHORT pressure
    fr = row.get('funding_rate_raw', np.nan)
    if pd.notna(fr) and fr != 0:
        fr_contrib = -float(fr) / (config.FUNDING_EXTREME_THRESH + 1e-9) * 0.3
        score += np.clip(fr_contrib, -1.0, 1.0)
        notes.append(f"fr={fr:.5f}")

    # 3. L/S ratio — contrarian: crowd long (>1.5) = SHORT; crowd short (<0.7) = LONG
    ls = row.get('ls_ratio_raw', np.nan)
    if pd.notna(ls) and ls > 0:
        ls_contrib = -(float(ls) - 1.0) / 0.5 * 0.3
        score += np.clip(ls_contrib, -1.0, 1.0)
        notes.append(f"ls={ls:.2f}")

    # 4. OI trend — directional confirmation (z-score)
    oi_z = row.get('ts_oi_buildup', np.nan)
    if pd.notna(oi_z):
        oi_contrib = np.sign(comp) * min(abs(float(oi_z)), 1.0) * 0.2
        score += oi_contrib
        notes.append(f"oi_z={oi_z:+.2f}")

    # Di trend regime: sedikit lebih agresif
    if regime == 'trend':
        score *= 1.1

    # Hard filter LONG: funding tidak extreme positif + L/S tidak overcrowded
    fr_val = row.get('funding_rate_raw', np.nan)
    ls_val = row.get('ls_ratio_raw', np.nan)
    long_ok = (
        (pd.isna(fr_val) or float(fr_val) < config.FUNDING_EXTREME_THRESH) and
        (pd.isna(ls_val) or float(ls_val) < config.LS_EXTREME_THRESH)
    )
    # Hard filter SHORT: funding extreme positif OR L/S overcrowded
    short_ok = (
        (pd.notna(fr_val) and float(fr_val) > config.FUNDING_EXTREME_THRESH) or
        (pd.notna(ls_val) and float(ls_val) > config.LS_EXTREME_THRESH)
    )

    # LONG-only mode: SHORT tidak pernah di-emit
    direction = 'LONG' if (score > 0.5 and long_ok) else 'WAIT'

    return direction, round(score, 3), notes


def _btc_mode_badge(mode: str) -> str:
    return {'BULL': '🟢 BULL', 'BEAR': '🔴 BEAR', 'NEUTRAL': '🟡 NEUTRAL'}.get(
        mode, '⚪ N/A'
    )


def get_latest_ranking(
    panel: pd.DataFrame,
    top_n: int = 20,
    score_col: str = 'composite_score',
    universe_symbols: list = None,
    apply_gate: bool = None,
    regime: str = 'reversal',
    btc_signal: dict = None,
) -> pd.DataFrame:
    """Snapshot ranking terbaru (output utama screener)."""
    # Hanya pakai candle yang sudah closed
    candle_minutes = 4 * 60
    now_utc   = pd.Timestamp.utcnow().tz_localize(None)
    closed_dt = panel['datetime'][
        panel['datetime'] + pd.Timedelta(minutes=candle_minutes) <= now_utc
    ].max()
    latest_dt = closed_dt if pd.notna(closed_dt) else panel['datetime'].max()
    latest    = panel[panel['datetime'] == latest_dt].copy()
    latest    = latest.dropna(subset=[score_col])

    if universe_symbols:
        latest = latest[latest['symbol'].isin(universe_symbols)]

    if latest.empty:
        print("  ⚠ Tidak ada data untuk timestamp terbaru")
        return pd.DataFrame()

    # ── Adaptive TS Gate ─────────────────────────────────────
    if apply_gate is None:
        apply_gate = config.TS_GATE_ENABLED

    if apply_gate and not latest.empty:
        from factors import TS_FACTOR_COLS
        latest, blocked = apply_ts_gate(latest, TS_FACTOR_COLS, regime=regime)
        print(f"\n  TS Gate [{regime.upper()} regime] "
              f"(thr={config.TS_SIGNAL_THRESHOLD}, min={config.TS_MIN_SIGNALS})")
        print(f"  Lolos : {len(latest)} coins  |  Blocked: {len(blocked)} coins")
        if not latest.empty:
            for _, row in latest.sort_values(score_col, ascending=False).head(top_n).iterrows():
                sigs = row.get('_ts_active', [])
                print(f"    {row['symbol']:<28} [{', '.join(sigs)}]")
        if latest.empty:
            print("  ⚠ Semua coin blocked — turunkan TS_SIGNAL_THRESHOLD atau TS_MIN_SIGNALS")
            return pd.DataFrame()

    latest = latest.drop(columns=[c for c in ['_ts_count', '_ts_active']
                                   if c in latest.columns])

    latest['rank'] = latest[score_col].rank(ascending=False).astype(int)
    latest = latest.sort_values('rank')

    # ── BTC directional gate ──────────────────────────────────
    # Kalau btc_signal None (gate disabled) → mode 'OFF': izinkan LONG & SHORT bersamaan
    btc_mode = (btc_signal or {}).get('mode', 'OFF')
    btc_comp = (btc_signal or {}).get('composite', None)
    latest['btc_composite'] = btc_comp if btc_comp is not None else np.nan
    latest['btc_mode']      = btc_mode

    # ── Trade Decision ────────────────────────────────────────
    decisions = []
    for _, row in latest.iterrows():
        direction, conviction, _ = compute_trade_decision(row, regime=regime,
                                                          btc_mode=btc_mode)
        decisions.append({'symbol': row['symbol'], 'direction': direction,
                          'conviction': conviction})
    dec_df = pd.DataFrame(decisions)
    latest = latest.merge(dec_df, on='symbol', how='left')

    # Regime label per coin
    if 'btc_regime' in latest.columns:
        latest['_regime_label'] = latest['btc_regime'].apply(
            lambda x: 'BULL' if x >= 0.5 else 'BEAR'
        )
    else:
        latest['_regime_label'] = 'BULL'

    # Print BTC gate banner + ranking
    if btc_signal is not None:
        facs = ', '.join(
            f"{f}{('+' if s>0 else '-')}" for f, s in btc_signal.get('signs', {}).items()
        )
        comp_str = f"{btc_signal['composite']:+.3f}" if btc_signal.get('composite') is not None else 'N/A'
        forced_tag = ' [FORCED]' if btc_signal.get('forced') else ''
        print(f"\n{'━'*100}")
        print(f"  BTC GATE: {_btc_mode_badge(btc_mode)}{forced_tag}   "
              f"composite={comp_str}   "
              f"thresholds: long>+{config.BTC_GATE_LONG_THR}, short<{config.BTC_GATE_SHORT_THR}")
        if facs:
            print(f"  Factors used ({len(btc_signal.get('signs', {}))}): {facs}")
        print(f"{'━'*100}")

    print(f"\n{'='*100}")
    print(f"  SCREENER RANKING  —  {latest_dt}  ({len(latest)} coins lolos gate)")
    print(f"{'='*100}")
    print(f"  {'#':<4} {'Symbol':<26} {'Score':>7}  "
          f"{'Decision':<10}  {'Conv':>5}  "
          f"{'Funding':>10}  {'L/S':>5}  {'OI(M)':>8}  Regime")
    print(f"  {'-'*96}")

    for _, row in latest.head(top_n).iterrows():
        dec  = row.get('direction', '?')
        conv = row.get('conviction', 0)
        fr   = row.get('funding_rate_raw', np.nan)
        ls   = row.get('ls_ratio_raw', np.nan)
        oi   = row.get('oi_raw', np.nan)
        reg  = row.get('_regime_label', '?')

        dec_icon = ('▲ LONG'  if dec == 'LONG'  else
                    '▼ SHORT' if dec == 'SHORT' else
                    '─ WAIT')
        fr_str   = f'{fr*100:+.4f}%' if pd.notna(fr) else '       N/A'
        ls_str   = f'{ls:.2f}'       if pd.notna(ls) else '  N/A'
        oi_str   = f'{oi/1e6:.1f}'   if pd.notna(oi) else '     N/A'

        print(f"  {int(row['rank']):<4} {row['symbol']:<26} "
              f"{row[score_col]:>7.4f}  "
              f"{dec_icon:<10}  "
              f"{conv:>5.2f}  "
              f"{fr_str:>10}  "
              f"{ls_str:>5}  "
              f"{oi_str:>8}  "
              f"{reg}")

    print(f"{'='*100}")
    print(f"  Score    = IC-weighted composite (faktor relatif terhadap semua coin)")
    print(f"  Conv     = composite + funding contrarian + L/S contrarian + OI confirmation")
    print(f"  Funding  = funding rate 8-jam (+ = longs bayar = bearish contrarian)")
    print(f"  L/S      = long/short ratio (>1.5 = crowd long = bearish; <0.7 = crowd short = bullish)")
    print(f"  OI(M)    = open interest juta kontrak")
    print(f"  ▲ LONG  |  ▼ SHORT  |  ─ WAIT")

    latest = latest.drop(columns=['_regime_label'], errors='ignore')
    _print_execution_summary(latest, btc_mode=btc_mode)

    return latest.head(top_n)


def _print_execution_summary(latest: pd.DataFrame, btc_mode: str = 'BULL'):
    """
    Branch eksekusi by BTC mode.
      BULL    : ▲ BELI list (LONG + conv>0.8 + L/S ok)
      BEAR    : ▼ JUAL/SHORT list (SHORT + conv<-0.8 + funding/LS ok)
      NEUTRAL : no execution, PANTAU only
    """
    d  = latest.get('direction',    pd.Series(dtype=str))
    cv = latest.get('conviction',   pd.Series(dtype=float))
    ls = latest.get('ls_ratio_raw', pd.Series(dtype=float))
    fr = latest.get('funding_rate_raw', pd.Series(dtype=float))

    print(f"\n{'█'*60}")
    print(f"  EKSEKUSI — HIGH CONVICTION  [BTC mode: {btc_mode}]")
    print(f"{'█'*60}")

    ls_ok = ls.isna() | (ls < config.LS_EXTREME_THRESH)
    buy = latest[
        (d == 'LONG') & (cv > 0.5) & ls_ok
    ].sort_values('conviction', ascending=False)

    if not buy.empty:
        print(f"\n  ▲ BELI  (LONG + Conv>0.5 + L/S tidak overcrowded)")
        print(f"  {'Symbol':<26} {'Conv':>5}  {'Funding':>10}  {'L/S':>5}  {'OI(M)':>8}")
        print(f"  {'-'*60}")
        for _, r in buy.iterrows():
            ls_v = r.get('ls_ratio_raw', np.nan)
            fr_v = r.get('funding_rate_raw', np.nan)
            oi_v = r.get('oi_raw', np.nan)
            ls_s = f"{ls_v:.2f}"       if pd.notna(ls_v) else '  N/A'
            fr_s = f"{fr_v*100:+.4f}%" if pd.notna(fr_v) else '       N/A'
            oi_s = f"{oi_v/1e6:.1f}"   if pd.notna(oi_v) else '     N/A'
            print(f"  {r['symbol']:<26} {r['conviction']:>5.2f}  "
                  f"{fr_s:>10}  {ls_s:>5}  {oi_s:>8}")
    else:
        print(f"\n  ▲ BELI  — tidak ada kandidat (Conv>0.5 + L/S<{config.LS_EXTREME_THRESH})")

    # PANTAU (WAIT + |conv|>0.3) selalu ditampilkan
    watch = latest[
        (d == 'WAIT') & (cv.abs() > 0.3)
    ].sort_values('conviction', key=lambda s: s.abs(), ascending=False).head(5)
    if not watch.empty:
        syms = ', '.join(
            f"{r['symbol'].replace('/USDT:USDT','')} ({r['conviction']:+.2f})"
            for _, r in watch.iterrows()
        )
        print(f"\n  👁 PANTAU (WAIT + |Conv|>0.3): {syms}")

    print(f"{'█'*60}")


def export_full_ranking(
    panel: pd.DataFrame,
    score_col: str = 'composite_score',
    output_path: str = "./output/screener_ranking.csv"
) -> pd.DataFrame:
    """Export ranking semua candle ke CSV untuk analisis manual."""
    panel = panel.copy()
    panel['rank'] = panel.groupby('datetime')[score_col].rank(ascending=False)
    panel = panel.sort_values(['datetime', 'rank'])
    panel.to_csv(output_path, index=False)
    print(f"  ✓ Full ranking exported → {output_path}")
    return panel
