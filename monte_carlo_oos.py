# monte_carlo_oos.py — Monte Carlo JUJUR: bootstrap dari per-cycle return
# WALK-FORWARD (out-of-sample) hasil backtest_screener — bukan bucket in-sample.
#
# Cycle returns ini sudah: causal (no look-ahead), universe likuid, kena fee +
# slippage, pakai picks nyata (score>0 implicit via screener? tidak — edge>5%).
# Bootstrap i.i.d. → distribusi equity, P(profit), drawdown, risiko ruin.
#
# CAVEAT: tetap i.i.d. (abaikan autокorelasi/regime shift); pool cycle ~83 saja
# (data 4 bulan). Tapi ini estimasi REALISTIS — beda jauh dari MC bucket in-sample.

import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

import config
import backtest_screener as bs

CAP = bs.CAP


def mc(rets, M=20000, K=None):
    rets = np.asarray(rets)
    if len(rets) < 5:
        return None
    K = K or len(rets)
    draws = np.random.choice(rets, size=(M, K))
    eq = CAP * np.cumprod(1 + draws, axis=1)
    final = eq[:, -1]
    runmax = np.maximum.accumulate(eq, axis=1)
    mdd = ((runmax - eq) / runmax).max(axis=1)
    return final, mdd, K


def report(label, res):
    if res['rets'] is None or len(res['rets']) < 5:
        print(f"  {label}: data kurang"); return
    r = np.asarray(res['rets'])
    out = mc(r)
    if out is None:
        print(f"  {label}: data kurang"); return
    final, mdd, K = out
    p5, p25, med, p75, p95 = np.percentile(final, [5, 25, 50, 75, 95])
    print(f"\n  {label}")
    print(f"    input: {len(r)} cycle returns  (mean {r.mean():+.2%}/cycle, "
          f"std {r.std():.2%})  → backtest aktual {res['ret']:+.1%}")
    print(f"    MC {K} siklus, mulai ${CAP:.0f}:")
    print(f"      median ${med:,.0f}  P(profit) {(final>CAP).mean():.1%}  "
          f"P(loss>50%) {(final<CAP*0.5).mean():.1%}  P(ruin>80%) {(final<CAP*0.2).mean():.1%}")
    print(f"      persentil  5%:${p5:,.0f}  25%:${p25:,.0f}  "
          f"75%:${p75:,.0f}  95%:${p95:,.0f}")
    print(f"      median maxDD {np.median(mdd):.1%}  95%-ile maxDD {np.percentile(mdd,95):.1%}")


def main():
    raw, btc, uni = bs.load()
    panel = bs.build_panel(raw, btc_df=btc)
    atr_map = {s: bs.add_atr(df) for s, df in raw.items()}
    times = sorted(panel['datetime'].unique())

    print("Precompute walk-forward picks (causal)...")
    picks = bs.precompute_picks(panel, raw, uni, times)

    print("=" * 78)
    print(f"  MONTE CARLO JUJUR — TANPA likuidasi (cross-margin / stop selalu eksekusi)")
    print(f"  → leverage IRRELEVANT. {len(picks)} cycle dasar, edge>{bs.EDGE_MIN:.0%}, hold {bs.HOLD}c")
    print("=" * 78)

    for slip in (0.0, 0.0005, 0.001, 0.002):
        print(f"\n  ===== slippage {slip:.2%}/side =====")
        report("A. stop 2×ATR (2% risk)",
               bs.simulate(picks, raw, atr_map, 1, slip, use_stop=True, use_liq=False))
        report("B. hold-to-horizon no-stop",
               bs.simulate(picks, raw, atr_map, 1, slip, use_stop=False,
                           equal_weight=True, use_liq=False))

    print("\n" + "=" * 78)
    print("  Leverage tidak muncul: dgn risk fix + tanpa likuidasi, P&L sama di semua leverage.")
    print("  Penentu nyata: SLIPPAGE (lihat kolom) + edge ~koin-flip + beta.")
    print("=" * 78)


if __name__ == "__main__":
    main()
