# CHIBOY BOT - Forex Trading Bot

## Quick Start
```bash
cd /home/chiboy/.openclaw/workspace/chiboy-bot
uv run python -c "from webapp import app; app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)"
```

## Features
- Real-time forex prices from Yahoo Finance
- ICT-based trading strategy (liquidity sweep → BOS → order block)
- XAUUSD spread adjustment ($17 to match broker)
- Interactive charts with 365 days data
- Timeframes: 1M, 5M, 15M, 1H, 1D, 1W

## Live Prices
- XAUUSD: ~$5,180 (adjusted)
- GBP/USD: ~1.347
- EUR/USD: ~1.179

## Access
- http://192.168.4.205:5000

## Files
- Bot: /home/chiboy/.openclaw/workspace/chiboy-bot/
- Dashboard: /home/chiboy/.openclaw/workspace/chiboy-bot/templates/dashboard.html
- Config: /home/chiboy/.openclaw/workspace/chiboy-bot/src/config/__init__.py
