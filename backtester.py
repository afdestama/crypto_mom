# backtester.py — 4H Intraday Swing Backtest

import pandas as pd
import numpy as np
import config


def run_backtest(panel: pd.DataFrame,
                 rolling_universe: pd.DataFrame = None,
                 score_col: str = 'composite_score',
                 top_n: int = config.TOP_N_COINS,
                 cost: float = config.TRANSACTION_COST,
                 capital: float = config.INITIAL_CAPITAL) -> dict:
    """
    Long-only 4H backtest:
    - Setiap candle (4 jam), ranking coin by composite_score
    - Hanya trading coin yang masuk rolling_universe pada tanggal tsb
      (anti survivorship bias). Jika rolling_universe=None → semua coin.
    - Long top_n coins equal weight
    - Forward return = FORWARD_PERIOD candle ke depan
    - Deduct transaction cost saat ada turnover

    Returns: dict {portfolio_returns, holdings_log, quintile_returns, metrics}
    """
    panel  = panel.copy().sort_values('datetime')

    # Rolling universe → dict {daily date -> set of symbols}
    universe_by_date = {}
    if rolling_universe is not None and not rolling_universe.empty:
        for d, grp in rolling_universe.groupby('date'):
            universe_by_date[pd.Timestamp(d).normalize()] = set(grp['symbol'])
    universe_dates = sorted(universe_by_date.keys())

    def _universe_for(dt):
        """Universe untuk candle dt = universe daily date terakhir <= dt."""
        if not universe_dates:
            return None  # tanpa filter
        day_date = pd.Timestamp(dt).normalize()
        eligible = [d for d in universe_dates if d <= day_date]
        if not eligible:
            return set()  # belum ada universe untuk tanggal ini
        return universe_by_date[eligible[-1]]

    times  = sorted(panel['datetime'].unique())

    # Buang FORWARD_PERIOD timestamp terakhir (tidak ada forward return)
    times  = times[:-config.FORWARD_PERIOD]

    port_returns  = []
    holdings_log  = []
    prev_holdings = set()

    quintile_rets = {q: [] for q in range(1, 6)}

    for dt in times:
        day = panel[panel['datetime'] == dt].dropna(subset=[score_col, 'forward_1d'])

        # Filter ke rolling universe tanggal ini
        valid_symbols = _universe_for(dt)
        if valid_symbols is not None:
            day = day[day['symbol'].isin(valid_symbols)]

        if len(day) < max(top_n, 5):
            port_returns.append(np.nan)
            continue

        # Ranking
        day = day.sort_values(score_col, ascending=False).reset_index(drop=True)

        # Quintile assignment
        try:
            day['quintile'] = pd.qcut(
                day[score_col].rank(method='first'),
                q=5, labels=[5, 4, 3, 2, 1]
            )
        except Exception:
            day['quintile'] = 3

        # Portfolio: top N
        top_coins = day.head(top_n)
        portfolio = dict(zip(top_coins['symbol'], top_coins['forward_1d']))

        # Transaction cost berdasarkan turnover
        current_set = set(portfolio.keys())
        turnover    = len(current_set - prev_holdings) / top_n if prev_holdings else 1.0
        tc          = turnover * cost

        # Equal-weight return
        avg_ret   = np.mean(list(portfolio.values()))
        net_ret   = avg_ret - tc
        port_returns.append(net_ret)

        holdings_log.append({
            'datetime'    : dt,
            'universe_n'  : len(valid_symbols) if valid_symbols is not None else len(day),
            'symbols'     : ', '.join(sorted(portfolio.keys())),
            'top_scores'  : round(top_coins[score_col].mean(), 4),
            'turnover'    : round(turnover, 3),
            'gross_return': round(avg_ret, 5),
            'tc'          : round(tc, 6),
            'net_return'  : round(net_ret, 5)
        })

        # Quintile returns
        for q in range(1, 6):
            q_ret = day[day['quintile'] == q]['forward_1d']
            quintile_rets[q].append(q_ret.mean() if not q_ret.empty else np.nan)

        prev_holdings = current_set

    # Build series
    port_series = pd.Series(port_returns, index=times, name='portfolio').dropna()

    quintile_df = pd.DataFrame(
        {f'Q{q}': quintile_rets[q] for q in range(1, 6)},
        index=times[:len(quintile_rets[1])]
    )

    metrics = compute_metrics(port_series, capital)

    return {
        'portfolio_returns': port_series,
        'holdings_log'     : pd.DataFrame(holdings_log),
        'quintile_returns' : quintile_df,
        'metrics'          : metrics
    }


def compute_metrics(returns: pd.Series, capital: float = 10_000) -> dict:
    """Semua performance metrics, annualized berdasarkan 4H candle."""
    r     = returns.dropna()
    cum   = (1 + r).cumprod()
    total = cum.iloc[-1] - 1

    # Annualize: 6 candle/hari × 365 hari = 2190 candle/tahun
    candles_per_year = config.CANDLES_PER_DAY * 365
    n_candles        = len(r)
    ann_return       = (1 + total) ** (candles_per_year / n_candles) - 1
    ann_vol          = r.std() * np.sqrt(candles_per_year)
    sharpe           = ann_return / (ann_vol + 1e-9)

    # Drawdown
    rolling_max = cum.cummax()
    drawdown    = (cum - rolling_max) / (rolling_max + 1e-9)
    max_dd      = drawdown.min()
    calmar      = ann_return / (abs(max_dd) + 1e-9)

    # Sortino
    downside_std = r[r < 0].std() * np.sqrt(candles_per_year)
    sortino      = ann_return / (downside_std + 1e-9)

    # Win/Loss
    win_rate      = (r > 0).mean()
    avg_win       = r[r > 0].mean() if (r > 0).any() else 0
    avg_loss      = r[r < 0].mean() if (r < 0).any() else 0
    profit_factor = abs(avg_win / (avg_loss + 1e-9))

    # Holding stats (dalam jam)
    holding_hours = config.FORWARD_PERIOD * 4

    return {
        'Timeframe'           : f"{config.TIMEFRAME} (forward {holding_hours}h)",
        'Total Return'        : f"{total:.2%}",
        'Annualized Return'   : f"{ann_return:.2%}",
        'Annualized Vol'      : f"{ann_vol:.2%}",
        'Sharpe Ratio'        : f"{sharpe:.3f}",
        'Sortino Ratio'       : f"{sortino:.3f}",
        'Calmar Ratio'        : f"{calmar:.3f}",
        'Max Drawdown'        : f"{max_dd:.2%}",
        'Win Rate'            : f"{win_rate:.2%}",
        'Avg Win  (per trade)': f"{avg_win:.4%}",
        'Avg Loss (per trade)': f"{avg_loss:.4%}",
        'Profit Factor'       : f"{profit_factor:.3f}",
        'N Candles Traded'    : n_candles,
        'N Days equiv.'       : round(n_candles / config.CANDLES_PER_DAY),
        'Final Capital ($)'   : f"${capital * (1 + total):,.2f}"
    }
