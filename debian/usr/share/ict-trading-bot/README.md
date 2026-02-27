# ICT Trading Bot

A comprehensive Python-based trading bot implementing ICT (Inner Circle Trader) concepts for forex and cryptocurrency markets.

## Features

- **Multi-Timeframe Analysis**: Analyzes markets from Weekly → 15min timeframes
- **ICT Strategy Implementation**:
  - Market Structure Analysis (Swing Highs/Lows, BOS, CHoCH)
  - Liquidity Zone Detection (Buy/Sell Side Liquidity)
  - Order Block Identification
  - Fair Value Gap (FVG) Detection
  - Trend Analysis with EMAs
- **Real-Time Data**:
  - OANDA API for Forex pairs
  - Binance API for Crypto pairs
- **Trade Execution**: Support for both demo and live trading

## Requirements

- Python 3.10+
- OANDA Account (for Forex trading)
- Binance Account (for Crypto trading)

## Installation

### 1. Clone and Install Dependencies

```bash
# Using uv (recommended)
cd trading-bot
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys
nano .env
```

#### Getting OANDA API Keys:
1. Log in to your OANDA account
2. Go to Manage API Access
3. Generate your API key and account ID

#### Getting Binance API Keys:
1. Log in to Binance
2. Go to API Management
3. Create new API key (enable futures for testnet)

## Usage

### Command Line
```bash
python main.py
```

### Web Interface (Recommended)
```bash
python webapp.py
```
Then open http://localhost:5000 in your browser

### Other Commands
```bash
# Analyze single pair
python main.py --analyze EUR_USD --type forex
python main.py --analyze BTCUSDT --type crypto

# Analyze all markets
python main.py --all

# Live trading
python main.py --live
```

## Project Structure

```
trading-bot/
├── src/
│   ├── config/
│   │   └── __init__.py       # Configuration settings
│   ├── data/
│   │   ├── oanda_client.py   # OANDA API connector
│   │   └── binance_client.py # Binance API connector
│   ├── analysis/
│   │   ├── ict_analyzer.py   # ICT strategy implementation
│   │   └── multi_timeframe.py # Multi-timeframe analysis
│   └── signals/
│       └── trading_signals.py # Signal generation & execution
├── logs/                      # Trading logs
├── data/                      # Historical data storage
├── main.py                    # Main entry point
├── .env.example              # Environment template
└── pyproject.toml            # Project dependencies
```

## ICT Strategy Components

### 1. Market Structure
- **Swing Highs/Lows**: Key pivot points
- **Break of Structure (BOS)**: Price breaks previous structure
- **Change of Character (CHoCH)**: Structure rejection signals

### 2. Liquidity
- **Buy-Side Liquidity**: Areas above price with clustered buy orders
- **Sell-Side Liquidity**: Areas below price with clustered sell orders
- **Liquidity Sweeps**: Price "hunts" stops before moving

### 3. Order Blocks
- Last bullish/bearish candle before institutional move
- Areas where "smart money" likely entered

### 4. Fair Value Gaps
- Imbalances between supply and demand
- Often act as future support/resistance

## Risk Management

The bot includes built-in risk management:
- Max 2% risk per trade
- Max 3 open trades
- Min 1:2 Risk:Reward ratio
- Max 6% daily loss limit

## Configuration

Edit `src/config/__init__.py` to customize:
- Trading pairs
- Risk parameters
- ICT strategy parameters
- Trading sessions (Kill Zones)

## Disclaimer

⚠️ **IMPORTANT**: This bot is for educational purposes. 
- Always test thoroughly in demo mode first
- Never trade with money you can't afford to lose
- Monitor the bot closely when running live
- The author is not responsible for any losses

## License

MIT License
