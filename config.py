# config.py — 4H Intraday Swing Configuration

EXCHANGE = "binanceusdm"

# ── Dynamic Symbol Filter ──────────────────────────────────
SYMBOL_QUOTE        = "USDT"
MIN_VOLUME_USD      = 100_000_000   # Min avg daily volume USD (naik dari 50jt)
TOP_N_SYMBOLS       = 100            # Top N by rolling volume (turun dari 50)
MIN_LISTING_DAYS    = 180           # Coin harus listing minimal 180 hari
EXCLUDE_SYMBOLS     = [
    # Stablecoins
    "USDC/USDT:USDT", "BUSD/USDT:USDT", "TUSD/USDT:USDT",
    "USDP/USDT:USDT", "FDUSD/USDT:USDT", "DAI/USDT:USDT",
    # Leverage tokens
    "BTCUP/USDT:USDT", "BTCDOWN/USDT:USDT",
    "ETHUP/USDT:USDT", "ETHDOWN/USDT:USDT",
    "BNBUP/USDT:USDT", "BNBDOWN/USDT:USDT",
    # Synthetic commodities
    "XAU/USDT:USDT", "XAG/USDT:USDT",
]

# Pattern exclude otomatis (substring match pada symbol)
EXCLUDE_PATTERNS    = [
    "UP/USDT", "DOWN/USDT",     # leverage tokens
    "BULL/USDT", "BEAR/USDT",   # leverage tokens
    "XAU", "XAG", "XPT",        # synthetic commodities
]

# ── Timeframe ──────────────────────────────────────────────
TIMEFRAME           = "4h"          # 4-hour candles
LOOKBACK_DAYS       = 120           # 120 hari × 6 candle = 720 candles
CANDLES_PER_DAY     = 6             # 24h / 4h = 6 candles per day

# Forward return target
# 6 candle = 24 jam = 1 hari ke depan
FORWARD_PERIOD          = 6             # default (dipakai jika regime tidak diketahui)
FORWARD_PERIOD_REVERSAL = 6             # candle target untuk reversal regime (~12h)
FORWARD_PERIOD_TREND    = 6             # candle target untuk trend regime (~24h)

# ── Rolling Windows (dalam satuan CANDLE, bukan hari) ──────
# 30 hari × 6 = 180 candle
LIQUIDITY_WINDOW         = 180
REVERSAL_WINDOW          = 180
LIQ_IMBALANCE_WINDOW     = 180
FUNDING_WINDOW           = 180      # funding update tiap 8h = 2 candle

# ROC windows: 7d=42, 14d=84, 30d=180 candle
ROC_WINDOWS              = [42, 84, 180]

# ── Backtesting ────────────────────────────────────────────
TOP_N_COINS         = 5
REBALANCE_FREQ      = "4h"          # Rebalance setiap 4 jam (per candle)
TRANSACTION_COST    = 0.0005        # 0.05% taker fee Binance perp
INITIAL_CAPITAL     = 10_000

# ── IC Analysis ────────────────────────────────────────────
IC_ROLLING_WINDOW   = 120           # 20 hari × 6 candle
MIN_ICIR            = 0.15           # naik dari 0.10 — hanya factor signifikan
MIN_COINS_PER_PERIOD = 5

# ── Regime Filter (BTC sebagai proxy market regime) ────────
BTC_REGIME_MA       = 50            # MA period deteksi regime (candle)
REGIME_BULL_THRESH  = 0.0           # close > MA(50) = bull

# ── BTC Directional Gate (untuk alt ranking) ───────────────
BTC_GATE_ENABLED        = True
BTC_GATE_LONG_THR       = 1.0       # composite > ini → BULL → boleh LONG alts
BTC_GATE_SHORT_THR      = -1.0      # composite < ini → BEAR → emit SHORT alts
BTC_GATE_TRAIN_DAYS     = 45        # walk-forward train window untuk BTC weights
BTC_GATE_FPS            = [3, 6, 12, 24]   # multi-FP sweep — per factor pilih best |IC|
BTC_GATE_IC_THRESHOLD   = 0.05      # min |IC train| (di best-FP) untuk include factor
PROB_LOOKBACK            = 120      # candle lookback untuk probabilitas
PROB_HIGH_SCORE_PCT      = 0.70    # top 30% skor = "high score" condition (fallback)
PROB_MIN_SAMPLES         = 10      # minimum sample untuk hitung probabilitas
PROB_CALIBRATION_WINDOW  = 540     # 90 hari × 6 candle — window kalibrasi LR

# ── Cache ──────────────────────────────────────────────────
CACHE_DIR           = "./cache"
SYMBOL_CACHE_HOURS  = 6             # Refresh universe tiap 6 jam
DATA_CACHE_HOURS    = 4             # Refresh OHLCV tiap 4 jam (per candle baru)
LIVE_DATA_CACHE_HOURS = 0           # 0 = funding/LS/OI selalu di-refresh setiap run

# ── Time-Series Gate ───────────────────────────────────────
TS_GATE_ENABLED     = True
TS_SIGNAL_THRESHOLD = 0.5           # z-score min agar 1 faktor TS "aktif"
TS_MIN_SIGNALS      = 2             # minimum faktor aktif dari kategori aktif
TS_ZSCORE_WINDOW    = 120           # rolling window z-score per-coin (candle)

# ── Regime Detection (volatility-based) ───────────────────
REGIME_TREND_WINDOW     = 12        # candle untuk hitung trend_strength BTC (= 48 jam)
REGIME_TREND_THRESHOLD  = 1.0       # trend_strength > ini = TREND regime

# ── Trade Decision Engine ──────────────────────────────────
DECISION_EDGE_MIN       = 0.03      # min edge untuk konfirmasi probabilistik
FUNDING_EXTREME_THRESH  = 0.0008    # |funding_rate| > ini = extreme sentiment
LS_EXTREME_THRESH       = 1.5       # L/S > ini = crowd terlalu long (SHORT bias)
LS_LOW_THRESH           = 0.7       # L/S < ini = crowd terlalu short (LONG bias)

# ── IC Drift Monitor ───────────────────────────────────────
IC_DRIFT_RECENT_WINDOW  = 60        # candle terbaru untuk hitung ICIR "sekarang"
IC_DRIFT_THRESHOLD      = 0.50      # ICIR_recent/ICIR_full < 0.50 = drifted
