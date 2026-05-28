# regime_adaptive.py — Pipeline sesuai usulan:
#   1. cari factor cocok untuk regime bull & bear (IC per-regime)
#   2. bobot IC ADAPTIF dengan regime yang sedang terjadi
#   3. relative ranking
#   4. probabilitas (P(up|high score, regime)) sebagai gate opsional
#   5. pick top-5 long
#
# DIBANGUN DENGAN PENGAMAN (kalau tidak → artefak seperti +78% sebelumnya):
#   - universe LIKUID (PIT rolling ~24), bukan lebar tak likuid
#   - SLIPPAGE dimodelkan
#   - WALK-FORWARD: bobot regime & probabilitas dihitung dari DATA MASA LALU saja
#
# Ekspektasi jujur rendah (liquid long-only sudah −49% di tes sebelumnya); ini
# tes bersih apakah adaptasi-regime mengubahnya.

import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

import config
from factor_research  import load_cache, build_panel, FACTORS, ic_series, icir
from backtest_bracket import add_atr, pit_universe
import backtest_neutral as bn

HOLD, WARMUP, REBAL = config.FORWARD_PERIOD, bn.WARMUP, config.FORWARD_PERIOD
N_LONG, L = 5, 3
MIN_ICIR_REG = 0.10          # ambang |ICIR| per-regime untuk masuk composite
PROB_HIGH_Q  = 0.70          # "high score" = top 30%


def regime_weights(panel, regime_val, upto_dt):
    """Signed-ICIR per regime, causal (data <= upto_dt). regime_val 1=bull,0=bear."""
    mask = (panel['datetime'] <= upto_dt) & \
           ((panel['regime'] >= 0.5) if regime_val >= 0.5 else (panel['regime'] < 0.5))
    w = {}
    for c in FACTORS:
        _, ir, _ = icir(ic_series(panel[mask], c))
        if np.isfinite(ir) and abs(ir) >= MIN_ICIR_REG:
            w[c] = ir
    if not w:                                      # fallback: ambil 3 |ICIR| terbesar
        cand = {}
        for c in FACTORS:
            _, ir, _ = icir(ic_series(panel[mask], c))
            cand[c] = ir if np.isfinite(ir) else 0.0
        w = dict(sorted(cand.items(), key=lambda kv: -abs(kv[1]))[:3])
    denom = sum(abs(x) for x in w.values()) + 1e-9
    return {c: x / denom for c, x in w.items()}


def regime_prob(panel, weights, regime_val, upto_dt):
    """P(fwd>0 | composite top-30%) di regime ini, dari data masa lalu."""
    mask = (panel['datetime'] <= upto_dt) & \
           ((panel['regime'] >= 0.5) if regime_val >= 0.5 else (panel['regime'] < 0.5))
    sub = panel[mask].copy()
    if len(sub) < 50:
        return np.nan
    sc = sum(w * sub[c].fillna(0) for c, w in weights.items())
    hi = sc >= sc.quantile(PROB_HIGH_Q)
    fwd = sub.loc[hi, 'fwd'].dropna()
    return (fwd > 0).mean() if len(fwd) >= 20 else np.nan


def score_with(snap, weights):
    return sum(w * snap[c].fillna(0) for c, w in weights.items())


def backtest(panel, raw, atr_map, universe, times, slip=0.0, use_prob_gate=False):
    equity, peak, max_dd = bn.CAPITAL, bn.CAPITAL, 0.0
    rets, fees_t, reasons, n = [], 0.0, {}, 0
    long_rets = []
    wb = wbear = None
    pb = pbear = np.nan
    last_recalc = None
    skipped = 0

    rebal = times[WARMUP: len(times) - HOLD][::REBAL]
    for dt in rebal:
        if equity <= 0:
            break
        snap = panel[panel['datetime'] == dt]
        snap = snap[snap['symbol'].isin(pit_universe(universe, dt))]
        if snap.empty:
            continue
        regime_val = snap['regime'].iloc[0]

        # Recompute bobot+prob bulanan dari data masa lalu (causal)
        if last_recalc is None or (dt - last_recalc).days >= 30:
            cutoff = dt - pd.Timedelta(hours=HOLD * 4)
            wb    = regime_weights(panel, 1, cutoff)
            wbear = regime_weights(panel, 0, cutoff)
            pb    = regime_prob(panel, wb, 1, cutoff)
            pbear = regime_prob(panel, wbear, 0, cutoff)
            last_recalc = dt

        weights = wb if regime_val >= 0.5 else wbear
        prob    = pb if regime_val >= 0.5 else pbear

        # Gate probabilitas: skip kalau P(up|high) historis regime ini <= 0.5
        if use_prob_gate and (not np.isfinite(prob) or prob <= 0.5):
            skipped += 1
            continue

        snap = snap.assign(_s=score_with(snap, weights).values).dropna(subset=['_s'])
        if len(snap) < N_LONG:
            continue
        picks = snap.sort_values('_s', ascending=False).head(N_LONG)['symbol']

        risk_usd, cycle_pnl = bn.RISK_PCT * equity, 0.0
        for sym in picks:
            df, atr = raw[sym], atr_map[sym]
            if dt not in df.index:
                continue
            idx = df.index.get_loc(dt)
            if idx + 1 >= len(df):
                continue
            entry, a = df['close'].iloc[idx], atr.iloc[idx]
            if not np.isfinite(a) or a <= 0:
                continue
            notional = risk_usd / (bn.ATR_MULT * a / entry)
            res = bn.sim_one(df, atr, idx, notional, L, +1, slip=slip)
            if res is None:
                continue
            pnl, reason, fee, _ = res
            cycle_pnl += pnl
            fees_t += fee
            reasons[reason] = reasons.get(reason, 0) + 1
            n += 1
            long_rets.append(pnl / notional)

        r = cycle_pnl / equity
        equity += cycle_pnl
        rets.append(r)
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0)

    rets = np.array(rets)
    sh = rets.mean() / (rets.std() + 1e-9) * np.sqrt(bn.CYCLES_YR) if len(rets) > 2 else float('nan')
    return {'equity': equity, 'ret': equity / bn.CAPITAL - 1, 'sharpe': sh,
            'max_dd': max_dd, 'n': n, 'fees': fees_t, 'skipped': skipped,
            'long_leg': np.mean(long_rets) if long_rets else 0}


def line(label, r):
    return (f"{label:<40} ${r['equity']:>8,.0f} ({r['ret']:>+6.1%})  "
            f"Sh {r['sharpe']:>5.2f}  DD {r['max_dd']:>4.0%}  "
            f"trades={r['n']:<4} long/trade {r['long_leg']:>+.3%}")


def main():
    raw, btc = load_cache()
    universe = pd.read_parquet(f"{config.CACHE_DIR}/_rolling_universe.parquet")
    panel = build_panel(raw, btc)
    atr_map = {s: add_atr(df) for s, df in raw.items()}
    times = sorted(panel['datetime'].unique())

    print("=" * 92)
    print(f"  REGIME-ADAPTIVE long top-{N_LONG} — LIQUID universe, walk-forward, "
          f"risk {bn.RISK_PCT:.0%}, {L}x")
    print(f"  (factor per-regime → bobot adaptif → ranking → prob gate → top-5 long)")
    print("=" * 92)

    for gate in (False, True):
        tag = "PROB-GATE on" if gate else "no gate"
        print(f"\n  [{tag}]")
        for slip in (0.0, 0.002):
            r = backtest(panel, raw, atr_map, universe, times, slip=slip, use_prob_gate=gate)
            extra = f"  (skipped {r['skipped']} cycles)" if gate else ""
            print("  " + line(f"slip {slip:.1%}/side", r) + extra)

    # Benchmark
    eq, ret, sh = __import__('backtest_stress').alt_benchmark(panel, raw, times, universe)
    print(f"\n  BENCHMARK EW-liquid alts (no lev)        ${eq:>8,.0f} ({ret:>+6.1%})  Sh {sh:>5.2f}")
    seg = btc[(btc.index >= pd.Timestamp(times[WARMUP])) & (btc.index <= pd.Timestamp(times[-HOLD]))]
    bh = seg['close'].iloc[-1] / seg['close'].iloc[0] - 1
    print(f"  BTC buy&hold                             ${bn.CAPITAL*(1+bh):>8,.0f} ({bh:>+6.1%})")
    print("=" * 92)


if __name__ == "__main__":
    main()
