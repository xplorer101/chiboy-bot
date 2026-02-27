"""
CHIBOY BOT Configuration
==============================
Central configuration for the trading bot including
OANDA (Forex), Binance (Crypto), and CHIBOY strategy parameters.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "src" / "config"

# Bot Settings
BOT_CONFIG = {
    "name": "CHIBOY BOT",
    "version": "1.0.0",
    "run_interval_seconds": 60,
    "analysis_timeframes": ["1W", "1D", "4H", "1H", "15m"],
    "entry_timeframe": "15m",
    "max_candles_per_fetch": 500,
}

# Risk Management
RISK_CONFIG = {
    "max_risk_per_trade": 0.02,      # 2% max risk per trade
    "max_open_trades": 3,
    "max_daily_loss": 0.06,          # 6% max daily loss
    "min_risk_reward": 2.0,          # Minimum 1:2 R:R
    "default_stop_loss_pips": 20,
    "default_take_profit_pips": 40,
}

# Trading Sessions (CHIBOY Kill Zones)
TRADING_SESSIONS = {
    "london": {"start": "02:00", "end": "05:00", "timezone": "UTC"},
    "ny_killzone": {"start": "08:30", "end": "11:30", "timezone": "UTC"},
    "tokyo": {"start": "00:00", "end": "03:00", "timezone": "UTC"},
    "london_ny_overlap": {"start": "13:00", "end": "16:00", "timezone": "UTC"},
}

# OANDA Configuration (Forex)
OANDA_CONFIG = {
    "environment": os.getenv("OANDA_ENV", "practice"),  # 'practice' or 'live'
    "api_key": os.getenv("OANDA_API_KEY", ""),
    "account_id": os.getenv("OANDA_ACCOUNT_ID", ""),
    "stream_endpoint": "stream-fxtrade.oanda.com" if os.getenv("OANDA_ENV") == "live" else "stream-fxpractice.oanda.com",
    "rest_endpoint": "api-fxtrade.oanda.com" if os.getenv("OANDA_ENV") == "live" else "api-fxpractice.oanda.com",
    "forex_pairs": [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
        "AUD_USD", "USD_CAD", "NZD_USD",
        "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY"
    ],
    "granularity_map": {
        "15m": "M15",
        "1H": "H1",
        "4H": "H4",
        "1D": "D",
        "1W": "W"
    }
}

# Binance Configuration (Crypto)
BINANCE_CONFIG = {
    "api_key": os.getenv("BINANCE_API_KEY", ""),
    "secret_key": os.getenv("BINANCE_SECRET_KEY", ""),
    "testnet": os.getenv("BINANCE_TESTNET", "true").lower() == "true",
    "base_url": "https://testnet.binance.vision/api" if os.getenv("BINANCE_TESTNET", "true").lower() == "true" else "https://api.binance.com",
    "crypto_pairs": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
        "XRPUSDT", "ADAUSDT", "DOGEUSDT", "MATICUSDT",
        "AVAXUSDT", "DOTUSDT", "LINKUSDT", "UNIUSDT"
    ],
    "interval_map": {
        "15m": "15m",
        "1H": "1h",
        "4H": "4h",
        "1D": "1d",
        "1W": "1w"
    }
}

# CHIBOY Strategy Parameters
CHIBOY_CONFIG = {
    # Liquidity Detection
    "liquidity_lookback": 50,
    "liquidity_tolerance": 0.001,  # 0.1% tolerance for equal highs/lows
    "liquidity_min_touches": 2,    # Minimum touches to confirm liquidity zone
    
    # Order Blocks
    "order_block_lookback": 10,
    "order_block_min_body_pct": 0.5,  # Body must be 50% of candle range
    
    # Fair Value Gap (FVG)
    "fvg_lookback": 3,
    "fvg_min_gap_pips": 5,
    
    # Trend Detection
    "trend_lookback": 50,
    "ema_fast": 9,
    "ema_slow": 21,
    "ema_mid": 50,
    
    # Structure Break
    "structure_break_bars": 4,
    
    # Market Structure
    "swing_lookback": 5,
    
    # Confirmation
    "require_fvg_confirmation": True,
    "require_order_block_confirmation": True,
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "log_file": str(LOGS_DIR / "trading-bot.log"),
    "console_output": True,
    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
}

# Timeframes for analysis (from high to low)
ANALYSIS_TIMEFRAMES = ["1W", "1D", "4H", "1H", "15m"]

# Trading modes
TRADING_MODE = {
    "dry_run": os.getenv("DRY_RUN", "true").lower() == "true",
    "execute_trades": os.getenv("EXECUTE_TRADES", "false").lower() == "true",
}
