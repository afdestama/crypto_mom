# report.py — Charts & output untuk 4H factor model

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import config

sns.set_theme(style="darkgrid", palette="muted")
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_ic_table(ic_summary: pd.DataFrame):
    print("\n" + "="*75)
    print(f"  FACTOR IC — Cross-Sectional Spearman vs Forward {config.FORWARD_PERIOD*4}H Return")
    print("="*75)
    print(f"  {'Factor':<22} {'IC Mean':>9} {'IC Std':>8} {'ICIR':>8} "
          f"{'IC>0%':>7} {'t-stat':>8}  Signal")
    print("-"*75)
    for _, row in ic_summary.iterrows():
        icir = row['icir']
        flag = "✓✓" if abs(icir) > 0.3 else ("✓ " if abs(icir) > 0.15 else "  ")
        dirn = "↑ Momentum " if row['ic_mean'] > 0 else "↓ Contrarian"
        print(f"  {flag} {row['factor']:<20} "
              f"{row['ic_mean']:>9.4f} "
              f"{row['ic_std']:>8.4f} "
              f"{icir:>8.4f} "
              f"{row['ic_positive_%']:>6.1f}% "
              f"{row['t_stat']:>8.3f}  {dirn}")
    print("="*75)
    print("  ✓✓ ICIR > 0.30 (Strong)  |  ✓ ICIR > 0.15 (Moderate)")


def print_metrics(metrics: dict):
    print("\n" + "="*48)
    print("  BACKTEST PERFORMANCE METRICS  (4H Intraday)")
    print("="*48)
    for k, v in metrics.items():
        print(f"  {k:<28} {v}")
    print("="*48)


def plot_full_report(backtest_result: dict,
                     ic_summary: pd.DataFrame,
                     ic_ts: pd.DataFrame):
    port_ret  = backtest_result['portfolio_returns']
    quint_ret = backtest_result['quintile_returns']
    hold_log  = backtest_result['holdings_log']

    cum_port  = (1 + port_ret).cumprod()
    cum_quint = (1 + quint_ret.dropna()).cumprod()

    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor('#0f0f1a')
    fig.suptitle(
        f"Crypto Perpetual Factor Model — 4H Intraday Swing\n"
        f"Forward: {config.FORWARD_PERIOD} candle (~{config.FORWARD_PERIOD*4}h)  |  "
        f"Universe: top {config.TOP_N_SYMBOLS} by volume  |  "
        f"Portfolio: top {config.TOP_N_COINS} coins",
        fontsize=13, fontweight='bold', y=0.99, color='white'
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

    ax_color = '#0f0f1a'

    # ── 1. Cumulative Return ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(ax_color)
    ax1.plot(cum_port.index, cum_port.values,
             color='#00d4aa', linewidth=1.8, label='Portfolio (Net)')
    ax1.fill_between(cum_port.index, 1, cum_port.values,
                     where=cum_port.values >= 1, alpha=0.12, color='#00d4aa')
    ax1.fill_between(cum_port.index, 1, cum_port.values,
                     where=cum_port.values < 1, alpha=0.12, color='#ff4757')
    ax1.axhline(1, color='#888', linewidth=0.7, linestyle='--')
    ax1.set_title("Cumulative Return (4H, Net of Fees)", color='white', fontsize=11)
    ax1.set_ylabel("Growth of $1", color='white')
    ax1.tick_params(colors='white')
    ax1.legend(fontsize=9)

    # ── 2. Drawdown ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(ax_color)
    dd = (cum_port / cum_port.cummax()) - 1
    ax2.fill_between(dd.index, dd.values, 0, color='#ff4757', alpha=0.75)
    ax2.set_title("Drawdown", color='white', fontsize=11)
    ax2.set_ylabel("DD %", color='white')
    ax2.tick_params(colors='white')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    # ── 3. Quintile Returns Bar ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(ax_color)
    if not cum_quint.empty:
        q_final = (cum_quint.iloc[-1] - 1) * 100
        colors  = ['#00d4aa', '#2ecc71', '#f39c12', '#e67e22', '#ff4757']
        bars    = ax3.bar(q_final.index, q_final.values, color=colors, width=0.6)
        ax3.axhline(0, color='white', linewidth=0.5)
        ax3.set_title("Total Return by Quintile\n(Q1 = Highest Score)", color='white', fontsize=10)
        ax3.set_ylabel("Total Return %", color='white')
        ax3.tick_params(colors='white')
        for bar, val in zip(bars, q_final.values):
            ax3.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', va='bottom',
                     fontsize=8, color='white')

    # ── 4. Rolling IC Time-Series ───────────────────────────
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.set_facecolor(ax_color)
    top_factors   = ic_summary.head(5)['factor'].tolist()
    rolling_win   = config.IC_ROLLING_WINDOW
    rolling_ic    = ic_ts[top_factors].rolling(rolling_win).mean()
    palette       = ['#00d4aa','#f39c12','#3498db','#e74c3c','#9b59b6']
    for col, color in zip(top_factors, palette):
        if col in rolling_ic.columns:
            ax4.plot(rolling_ic.index, rolling_ic[col],
                     label=col, linewidth=1.3, color=color)
    ax4.axhline(0, color='#888', linewidth=0.7, linestyle='--')
    ax4.set_title(f"Rolling {rolling_win}-Candle IC (Top 5 Factors)", color='white', fontsize=10)
    ax4.set_ylabel("IC", color='white')
    ax4.tick_params(colors='white')
    ax4.legend(fontsize=8, ncol=3)

    # ── 5. IC Mean Horizontal Bar ───────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_facecolor(ax_color)
    ic_plot = ic_summary.set_index('factor')['ic_mean'].sort_values()
    colors  = ['#00d4aa' if v > 0 else '#ff4757' for v in ic_plot.values]
    ax5.barh(ic_plot.index, ic_plot.values, color=colors, height=0.6)
    ax5.axvline(0, color='white', linewidth=0.5)
    ax5.set_title("IC Mean per Factor", color='white', fontsize=10)
    ax5.set_xlabel("IC Mean", color='white')
    ax5.tick_params(colors='white')

    # ── 6. ICIR Bar ─────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_facecolor(ax_color)
    icir_plot = ic_summary.set_index('factor')['icir'].sort_values()
    colors    = ['#00d4aa' if v > 0 else '#ff4757' for v in icir_plot.values]
    ax6.barh(icir_plot.index, icir_plot.values, color=colors, height=0.6)
    ax6.axvline( 0.2, color='#f1c40f', linewidth=1, linestyle='--', label='±0.2')
    ax6.axvline(-0.2, color='#f1c40f', linewidth=1, linestyle='--')
    ax6.axvline(0,    color='white',   linewidth=0.5)
    ax6.set_title("ICIR per Factor", color='white', fontsize=10)
    ax6.set_xlabel("ICIR", color='white')
    ax6.tick_params(colors='white')
    ax6.legend(fontsize=7)

    # ── 7. Return Distribution ──────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.set_facecolor(ax_color)
    ax7.hist(port_ret.values * 100, bins=50,
             color='#3498db', alpha=0.8, edgecolor='none')
    mu = port_ret.mean() * 100
    ax7.axvline(0,  color='white',   linewidth=1)
    ax7.axvline(mu, color='#f1c40f', linewidth=1.5,
                linestyle='--', label=f'Mean={mu:.3f}%')
    ax7.set_title(f"Return Distribution per {config.FORWARD_PERIOD*4}H", color='white', fontsize=10)
    ax7.set_xlabel("Return (%)", color='white')
    ax7.tick_params(colors='white')
    ax7.legend(fontsize=8)

    # Set all spine colors
    for ax in fig.get_axes():
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')

    path = f"{OUTPUT_DIR}/factor_model_4h_report.png"
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print(f"\n  ✓ Chart saved → {path}")
    return path


def save_results(ic_summary, backtest_result):
    """Save semua hasil ke CSV."""
    ic_summary.to_csv(f"{OUTPUT_DIR}/ic_summary_4h.csv", index=False)
    backtest_result['portfolio_returns'].to_csv(f"{OUTPUT_DIR}/portfolio_returns_4h.csv")
    backtest_result['holdings_log'].to_csv(f"{OUTPUT_DIR}/holdings_log_4h.csv", index=False)
    backtest_result['quintile_returns'].to_csv(f"{OUTPUT_DIR}/quintile_returns_4h.csv")
    print(f"  ✓ Results saved → {OUTPUT_DIR}/")
