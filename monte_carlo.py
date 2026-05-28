# monte_carlo.py — Monte Carlo atas BUY CANDIDATES (top-N by edge).
#
# Bootstrap dari sampel return KONDISIONAL tiap coin (return forward saat coin
# high-score + regime sama — basis edge). Output:
#   1. CI edge per coin (apakah edge solid atau noise N-kecil)
#   2. Distribusi return basket 1-siklus  → P(profit), persentil
#   3. Distribusi equity multi-siklus (compounding) → P(profit), maxDD, ruin
#
# CAVEAT (baca dulu):
#   - Bootstrap i.i.d. → MENGASUMSIKAN masa depan ~ distribusi kondisional
#     historis (abaikan perubahan regime, autокorelasi, picks berubah tiap siklus).
#   - Sampel IN-SAMPLE, N kecil → sentral cenderung optimistis, CI lebar.
#   - MC mengukur ketidakpastian, BUKAN memvalidasi edge.

import warnings, os
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

import config
from factors      import build_panel, FACTOR_COLS
from ic_analysis  import cross_sectional_ic, build_composite_score

CACHE   = config.CACHE_DIR
TOP_N   = 5
M       = 20000           # jumlah trial Monte Carlo
K_CYC   = 100             # siklus per equity-path (≈100 hari)
CAP     = 300.0
FEE     = config.TRANSACTION_COST
HQ      = config.PROB_HIGH_SCORE_PCT
MINS    = config.PROB_MIN_SAMPLES


def load_panel():
    uni = pd.read_parquet(f"{CACHE}/_rolling_universe.parquet")
    btc = pd.read_parquet(f"{CACHE}/_btc_regime.parquet")
    data = {}
    for s in sorted(uni['symbol'].unique()):
        p = f"{CACHE}/{s.replace('/','_').replace(':','_')}_4h.parquet"
        if os.path.exists(p):
            d = pd.read_parquet(p)
            if len(d) >= config.IC_ROLLING_WINDOW * 2:
                data[s] = d
    panel = build_panel(data, btc_df=btc)
    avail = [c for c in FACTOR_COLS if c in panel.columns
             and panel[c].notna().sum() > config.IC_ROLLING_WINDOW * 5]
    ic, _ = cross_sectional_ic(panel, avail, forward_col='forward_1d')
    panel, _ = build_composite_score(panel, ic)
    latest = uni[uni['date'] == uni['date'].max()]['symbol'].tolist()
    return panel, latest


def conditional_returns(panel, sym):
    """Sampel forward_1d saat coin high-score + regime sama sekarang (basis edge)."""
    coin = panel[panel['symbol'] == sym].sort_values('datetime')
    scores = coin['composite_score']
    if scores.notna().sum() < MINS * 2:
        return np.array([]), np.nan
    base = (coin['forward_1d'].dropna() > 0).mean()
    thr = scores.quantile(HQ)
    high = scores >= thr
    cur = coin['btc_regime'].iloc[-1] if 'btc_regime' in coin else 1.0
    reg = (coin['btc_regime'] >= 0.5) if cur >= 0.5 else (coin['btc_regime'] < 0.5)
    r = coin.loc[high & reg, 'forward_1d'].dropna()
    if len(r) < MINS:
        r = coin.loc[high, 'forward_1d'].dropna()
    return r.values, base


def main(slip=0.001):
    panel, latest = load_panel()

    # Buy candidates: top-N by edge (tanpa filter score, sesuai screener)
    rows = []
    for sym in latest:
        r, base = conditional_returns(panel, sym)
        if len(r) >= MINS and np.isfinite(base):
            rows.append((sym, r, (r > 0).mean() - base, base))
    rows.sort(key=lambda x: -x[2])
    picks = rows[:TOP_N]

    print("=" * 78)
    print(f"  MONTE CARLO — buy candidates top-{TOP_N} by edge   "
          f"(M={M:,} trials, cost {2*(FEE+slip):.2%}/cycle)")
    print("=" * 78)

    # ── 1. CI edge per coin (bootstrap) ──
    print(f"\n  [1] EDGE per coin + 90% CI (bootstrap N sampel)")
    print(f"  {'Symbol':<22}{'N':>4}{'edge':>8}{'  90% CI':>20}{'  mean ret':>11}")
    print("  " + "-" * 66)
    pooled = []
    for sym, r, edge, base in picks:
        boot = np.array([(np.random.choice(r, len(r)) > 0).mean() - base
                         for _ in range(2000)])
        lo, hi = np.percentile(boot, [5, 95])
        pooled.append(r)
        flag = "  ⚠ CI lewat 0" if lo <= 0 <= hi else ""
        print(f"  {sym:<22}{len(r):>4}{edge:>+8.1%}"
              f"   [{lo:>+5.1%},{hi:>+6.1%}]{r.mean():>+10.2%}{flag}")

    # ── 2. Distribusi return basket 1-siklus ──
    cost = 2 * (FEE + slip)
    draws = np.column_stack([np.random.choice(r, M) for _, r, _, _ in picks])
    basket = draws.mean(axis=1) - cost            # equal-weight, net cost
    print(f"\n  [2] RETURN BASKET 1-siklus (24h, equal-weight {TOP_N} coin, net cost)")
    print(f"      mean {basket.mean():+.2%}   median {np.median(basket):+.2%}   "
          f"P(profit) {(basket>0).mean():.1%}")
    p5, p25, p75, p95 = np.percentile(basket, [5, 25, 75, 95])
    print(f"      persentil  5%:{p5:+.2%}  25%:{p25:+.2%}  "
          f"75%:{p75:+.2%}  95%:{p95:+.2%}")

    # ── 3. Equity multi-siklus (compounding) ──
    # tiap siklus: basket = rata-rata TOP_N draw independen, dikurangi cost
    bsk = np.stack([
        np.column_stack([np.random.choice(r, M) for _, r, _, _ in picks]).mean(axis=1)
        for _ in range(K_CYC)
    ], axis=1) - cost
    eq = CAP * np.cumprod(1 + bsk, axis=1)
    final = eq[:, -1]
    run_max = np.maximum.accumulate(eq, axis=1)
    maxdd = ((run_max - eq) / run_max).max(axis=1)
    print(f"\n  [3] EQUITY setelah {K_CYC} siklus (mulai ${CAP:.0f}, compounding)")
    print(f"      median ${np.median(final):,.0f}   mean ${final.mean():,.0f}   "
          f"P(profit) {(final>CAP).mean():.1%}")
    fp5, fp25, fp75, fp95 = np.percentile(final, [5, 25, 75, 95])
    print(f"      persentil  5%:${fp5:,.0f}  25%:${fp25:,.0f}  "
          f"75%:${fp75:,.0f}  95%:${fp95:,.0f}")
    print(f"      median maxDD {np.median(maxdd):.1%}   "
          f"P(loss>50%) {(final < CAP*0.5).mean():.1%}")
    print("=" * 78)
    print("  Ingat: in-sample + bootstrap i.i.d. → optimistis & abaikan regime shift.")


if __name__ == "__main__":
    import sys
    s = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001
    main(slip=s)
