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
    "forex_pairs": os.getenv("OANDA_PAIRS", "EUR_USD,GBP_USD,USD_JPY,USD_CHF,AUD_USD,USD_CAD").split(","),
    "granularity_map": {
        "1m": "M1",
        "5m": "M5",
        "15m": "M15",
        "30m": "M30",
        "1h": "H1",
        "4h": "H4",
        "1D": "D",
        "1W": "W"
    }
}

# Custom spread adjustments to match broker prices
# Subtract this value from the market price
CUSTOM_SPREADS = {
    "XAUUSD": 16.7,  # Adjust XAUUSD to match broker
    "XAGUSD": 0.05,  # Silver adjustment
}

# Binance Configuration (Crypto)
BINANCE_CONFIG = {
    "api_key": os.getenv("BINANCE_API_KEY", ""),
    "secret_key": os.getenv("BINANCE_SECRET_KEY", ""),
    "testnet": os.getenv("BINANCE_TESTNET", "true").lower() == "true",
    "crypto_pairs": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"],
    "base_url": "https://testnet.binance.vision/api" if os.getenv("BINANCE_TESTNET", "true").lower() == "true" else "https://api.binance.com",
}

# Trading Mode
TRADING_MODE = {
    "dry_run": os.getenv("DRY_RUN", "true").lower() == "true",
    "execute_trades": os.getenv("EXECUTE_TRADES", "false").lower() == "true",
}

# CHIBOY Strategy Configuration
CHIBOY_CONFIG = {
    "liquiditylookback": 10,
    "order_block_lookback": 5,
    "min_bos_candles": 2,
    "fibonacci_retracements": [0.382, 0.5, 0.618, 0.786],
    "session_filter": True,
    "atr_multiplier": 1.5,
    "volume_confirmation": False,
}

# Analysis Timeframes (alias for backward compatibility)
ANALYSIS_TIMEFRAMES = BOT_CONFIG["analysis_timeframes"]

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
