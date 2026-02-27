"""
CHIBOY BOT - Web Interface
================================
Flask-based graphical user interface for the trading bot.
"""

import sys
import os
import json
import urllib.request
from pathlib import Path
from flask import Flask, render_template, jsonify, request
import logging
import threading

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import BOT_CONFIG, TRADING_MODE, OANDA_CONFIG, BINANCE_CONFIG, CUSTOM_SPREADS
# from src.analysis.multi_timeframe import MultiTimeframeAnalyzer
from src.analysis.stub_analyzer import QuickAnalyzer, MultiTimeframeAnalyzer
# from src.signals.trading_signals import TradingSignals, SignalManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, static_folder='static', static_url_path='')
app.secret_key = 'chiboy-bot-secret-key'

# Initialize components
analyzer = QuickAnalyzer()
# signals_manager = SignalManager()
# trading_signals = TradingSignals()

from datetime import datetime

# Store analysis results
analysis_cache = {
    "last_update": None,
    "results": [],
    "opportunities": []
}

# Trade History Storage
trade_history = {
    "trades": [],  # List of closed trades
    "open_trades": [],  # Currently open trades
    "next_id": 1
}


def save_trade_to_history(trade_data):
    """Save a trade to history when executed"""
    trade_id = trade_history["next_id"]
    trade_history["next_id"] += 1
    
    trade = {
        "id": trade_id,
        "symbol": trade_data.get("symbol"),
        "direction": trade_data.get("direction"),
        "entry_price": trade_data.get("entry_price"),
        "stop_loss": trade_data.get("stop_loss"),
        "take_profit": trade_data.get("take_profit"),
        "risk_reward": trade_data.get("risk_reward"),
        "timeframe": trade_data.get("timeframe"),
        "entry_time": datetime.now().isoformat(),
        "exit_time": None,
        "exit_price": None,
        "result": None,  # "TP" or "SL" or "OPEN"
        "pnl_pips": 0,
        "status": "open",  # open, closed
        "reasons": trade_data.get("reasons", [])
    }
    
    trade_history["open_trades"].append(trade)
    return trade


def update_trade_status(trade_id, exit_price, result):
    """Update trade when it hits TP or SL"""
    for trade in trade_history["open_trades"]:
        if trade["id"] == trade_id:
            trade["exit_price"] = exit_price
            trade["exit_time"] = datetime.now().isoformat()
            trade["result"] = result
            trade["status"] = "closed"
            
            # Calculate PnL in pips
            if trade["direction"] == "long":
                if result == "TP":
                    trade["pnl_pips"] = (exit_price - trade["entry_price"]) * 10000
                else:
                    trade["pnl_pips"] = (exit_price - trade["entry_price"]) * 10000
            else:
                if result == "TP":
                    trade["pnl_pips"] = (trade["entry_price"] - exit_price) * 10000
                else:
                    trade["pnl_pips"] = (trade["entry_price"] - exit_price) * 10000
            
            # Move to closed trades
            trade_history["trades"].append(trade)
            trade_history["open_trades"].remove(trade)
            return trade
    
    return None


def run_analysis_background():
    """Background analysis task."""
    global analysis_cache
    
    try:
        logger.info("Running background analysis...")
        opportunities = analyzer.get_top_opportunities(limit=10)
        
        analysis_cache["last_update"] = datetime.now().isoformat()
        analysis_cache["opportunities"] = opportunities
        
        logger.info(f"Analysis complete: {len(opportunities)} opportunities found")
    except Exception as e:
        logger.error(f"Background analysis error: {e}")


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/status')
def api_status():
    """Get bot status."""
    return jsonify({
        "status": "running",
        "version": BOT_CONFIG["version"],
        "dry_run": TRADING_MODE["dry_run"],
        "execute_trades": TRADING_MODE["execute_trades"],
        "forex_enabled": bool(OANDA_CONFIG["api_key"]),
        "crypto_enabled": bool(BINANCE_CONFIG["api_key"]),
        "last_update": analysis_cache.get("last_update")
    })


@app.route('/api/analyze')
def api_analyze():
    """Run analysis on all markets - FAST with Entry, SL, TP"""
    try:
        # Check for force refresh
        from flask import request
        force = request.args.get('force', 'false').lower() == 'true'
        
        # Force clear cache if requested
        if force:
            from src.data.live_data import live_data
            live_data._cache = {}
            live_data._cache_time = {}
            logger.info("Cache cleared - forcing fresh data")
        
        # Use quick analyzer for fast results
        opportunities = QuickAnalyzer.analyze_all()
        
        return jsonify({
            "success": True,
            "opportunities": opportunities,
            "count": len(opportunities)
        })
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/analyze/<symbol>')
def api_analyze_symbol(symbol):
    """Analyze a specific symbol using ICT strategy with LIVE prices."""
    asset_type = request.args.get('type', 'forex')
    
    try:
        # Use QuickAnalyzer with live data for ICT analysis
        from src.analysis.stub_analyzer import QuickAnalyzer, MultiTimeframeAnalyzer
        
        # Determine if crypto
        is_crypto = asset_type == 'crypto' or symbol.endswith('USDT')
        
        # Get live price - try different formats
        from src.data.live_data import live_data
        live_prices = live_data.get_all()
        
        # Try original format first
        current_price = None
        price_data = live_prices.get(symbol)
        
        if not price_data:
            # Try swapping _ with empty
            alt_symbol = symbol.replace('_', '')
            price_data = live_prices.get(alt_symbol)
        
        if not price_data:
            # Try Yahoo format
            yahoo_symbol = symbol.replace('_', '').replace('USD', '=X')
            # Search in all prices
            for key, val in live_prices.items():
                if symbol.replace('_', '') in key.replace('_', '').replace('USD', ''):
                    price_data = val
                    break
        
        if not price_data:
            return jsonify({"success": False, "error": f"Symbol {symbol} not found. Available: {list(live_prices.keys())[:5]}"}), 404
        
        current_price = price_data.get('price')
        
        if not current_price:
            return jsonify({"success": False, "error": "No live price available"}), 404
        
        # Run ICT analysis
        result = QuickAnalyzer.analyze_ict(symbol, 'crypto' if is_crypto else 'forex', current_price)
        
        if not result:
            return jsonify({"success": False, "error": "No valid entry signal (requires liquidity sweep + BOS + Order Block)"}), 404
        
        # Format report
        o = result['opportunity']
        report = f"""
📊 ICT Analysis: {symbol}
━━━━━━━━━━━━━━━━━━
⏱️  Timeframe: {result['timeframe']}

📈 Direction: {o['direction'].upper()}
💰 Entry: {o['entry_price']}
🛡️  Stop Loss: {o['stop_loss']}
🎯 Take Profit: {o['take_profit']}
📊 Risk:Reward: 1:{o['risk_reward']}
🎯 Confidence: {o['confidence']}%

📋 Reasons:
"""
        for r in o['reasons']:
            report += f"  • {r}\n"
        
        return jsonify({
            "success": True,
            "result": result,
            "report": report
        })
    except Exception as e:
        logger.error(f"Analysis error for {symbol}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/opportunities')
def api_opportunities():
    """Get current trading opportunities."""
    return jsonify({
        "success": True,
        "opportunities": analysis_cache.get("opportunities", [])
    })


@app.route('/api/execute', methods=['POST'])
def api_execute():
    """Execute a trade signal."""
    data = request.json
    
    try:
        # Create signal from request
        signal = trading_signals.create_signal(
            symbol=data.get('symbol'),
            asset_type=data.get('type', 'forex'),
            direction=data.get('direction'),
            entry_price=float(data.get('entry_price', 0)) or 0,
            stop_loss=float(data.get('stop_loss', 0)) or 0,
            take_profit=float(data.get('take_profit', 0)) or 0,
            confidence=float(data.get('confidence', 0)),
            timeframe=data.get('timeframe', '15m'),
            reasons=data.get('reasons', [])
        )
        
        # Execute signal
        result = trading_signals.execute_signal(signal)
        
        return jsonify({
            "success": result.get("success", False),
            "result": result,
            "dry_run": TRADING_MODE["dry_run"]
        })
    except Exception as e:
        logger.error(f"Trade execution error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    """Get or update bot settings."""
    if request.method == 'POST':
        data = request.json
        
        # Update settings (these would typically be saved to config)
        TRADING_MODE["dry_run"] = data.get("dry_run", TRADING_MODE["dry_run"])
        TRADING_MODE["execute_trades"] = data.get("execute_trades", TRADING_MODE["execute_trades"])
        
        return jsonify({"success": True})
    
    return jsonify({
        "dry_run": TRADING_MODE["dry_run"],
        "execute_trades": TRADING_MODE["execute_trades"],
        "max_risk": 0.02,
        "max_trades": 3
    })


@app.route('/api/markets')
def api_markets():
    """Get list of available markets."""
    return jsonify({
        "forex": OANDA_CONFIG["forex_pairs"],
        "crypto": BINANCE_CONFIG["crypto_pairs"]
    })


@app.route('/api/price/<symbol>')
def api_price(symbol):
    """Get live price for a symbol."""
    try:
        from src.data.live_data import live_data
        
        price_data = live_data.get_price(symbol)
        
        return jsonify({
            "success": True,
            "symbol": symbol,
            "price": price_data['price'],
            "bid": price_data['bid'],
            "ask": price_data['ask'],
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/prices')
def api_prices():
    """Get live prices for all symbols."""
    try:
        from src.data.live_data import live_data
        
        # Check for force refresh
        force = request.args.get('force', 'false').lower() == 'true'
        if force:
            live_data._cache = {}
            live_data._cache_time = {}
        
        prices = live_data.get_all()
        
        return jsonify({
            "success": True,
            "prices": prices,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/chart/<symbol>')
def api_chart(symbol):
    """Get chart data for a symbol - from Yahoo Finance."""
    timeframe = request.args.get('timeframe', '1D')
    
    try:
        # Map our symbol to Yahoo symbol
        yahoo_symbol = symbol
        if symbol == 'XAUUSD':
            yahoo_symbol = 'GC=F'
        elif symbol == 'BTCUSDT':
            yahoo_symbol = 'BTC-USD'
        elif symbol == 'ETHUSDT':
            yahoo_symbol = 'ETH-USD'
        elif symbol == 'SOLUSDT':
            yahoo_symbol = 'SOL-USD'
        elif symbol == 'US30':
            yahoo_symbol = '^DJI'
        elif symbol == 'NAS100':
            yahoo_symbol = '^IXIC'
        elif symbol == 'SPX500':
            yahoo_symbol = '^GSPC'
        elif symbol == 'USOIL':
            yahoo_symbol = 'CL=F'
        else:
            # Convert GBP_USD to GBPUSD=X
            yahoo_symbol = symbol.replace('_', '') + '=X'
        
        # Map timeframe to Yahoo interval
        interval_map = {
            '1M': '1m',
            '5M': '5m',
            '15m': '15m',
            '15M': '15m',
            '1h': '1h',
            '1H': '1h',
            '1D': '1d',
            '1W': '1wk'
        }
        interval = interval_map.get(timeframe, '1d')
        
        # Map timeframe to range
        range_map = {
            '1M': '5d',
            '5M': '5d',
            '15m': '5d',
            '15M': '5d',
            '1h': '1y',
            '1H': '1y',
            '1D': '1y',
            '1W': '5y'
        }
        range_param = range_map.get(timeframe, '1y')
        
        # Fetch from Yahoo
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?interval={interval}&range={range_param}"
        
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
        
        if not data.get('chart', {}).get('result'):
            raise Exception("No data")
        
        result = data['chart']['result'][0]
        quotes = result.get('indicators', {}).get('quote', [{}])[0]
        
        candles = []
        timestamps = result.get('timestamp', [])
        
        for i, ts in enumerate(timestamps):
            if ts is None:
                continue
            open_p = quotes.get('open', [None])[i]
            high_p = quotes.get('high', [None])[i]
            low_p = quotes.get('low', [None])[i]
            close_p = quotes.get('close', [None])[i]
            
            if open_p is None or high_p is None or low_p is None or close_p is None:
                continue
            
            # Apply spread adjustment for broker matching
            adjustment = 0
            if symbol in CUSTOM_SPREADS:
                adjustment = CUSTOM_SPREADS[symbol]
                open_p = open_p - adjustment
                high_p = high_p - adjustment
                low_p = low_p - adjustment
                close_p = close_p - adjustment
            
            candles.append({
                "time": ts,
                "open": round(open_p, 5),
                "high": round(high_p, 5),
                "low": round(low_p, 5),
                "close": round(close_p, 5)
            })
        
        return jsonify({
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": candles
        })
        
    except Exception as e:
        logger.error(f"Chart error: {e}")
        # Fallback to demo data
        try:
            from src.data.demo_data import demo_generator
            
            bars_map = {'1M': 1000, '5M': 500, '15m': 200, '15M': 200, '1h': 100, '1H': 1000, '1D': 365, '1W': 52}
            bars = bars_map.get(timeframe, 100)
            
            df = demo_generator.generate_candles(symbol, timeframe, bars)
            
            if df is None or len(df) == 0:
                return jsonify({"success": False, "error": str(e)}), 404
            
            candles = []
            for dt, row in df.iterrows():
                candles.append({
                    "time": int(dt.timestamp()),
                    "open": round(row['open'], 5),
                    "high": round(row['high'], 5),
                    "low": round(row['low'], 5),
                    "close": round(row['close'], 5)
                })
            
            return jsonify({
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": candles,
                "demo": True
            })
        except:
            return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        logger.error(f"Chart error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/trades')
def api_trades():
    """Get all trades (open and closed)"""
    return jsonify({
        "success": True,
        "open_trades": trade_history["open_trades"],
        "closed_trades": trade_history["trades"],
        "stats": {
            "total_trades": len(trade_history["trades"]),
            "winning_trades": len([t for t in trade_history["trades"] if t.get("result") == "TP"]),
            "losing_trades": len([t for t in trade_history["trades"] if t.get("result") == "SL"]),
            "win_rate": (len([t for t in trade_history["trades"] if t.get("result") == "TP"]) / len(trade_history["trades"]) * 100) if trade_history["trades"] else 0,
            "total_pips": sum([t.get("pnl_pips", 0) for t in trade_history["trades"]])
        }
    })


@app.route('/api/trades/open', methods=['POST'])
def api_open_trade():
    """Open a new trade from an opportunity"""
    data = request.json
    
    trade = save_trade_to_history({
        "symbol": data.get("symbol"),
        "direction": data.get("direction"),
        "entry_price": data.get("entry_price"),
        "stop_loss": data.get("stop_loss"),
        "take_profit": data.get("take_profit"),
        "risk_reward": data.get("risk_reward"),
        "timeframe": data.get("timeframe"),
        "reasons": data.get("reasons", [])
    })
    
    return jsonify({"success": True, "trade": trade})


@app.route('/api/trades/<int:trade_id>', methods=['PUT'])
def api_update_trade(trade_id):
    """Update trade status (hit TP or SL)"""
    data = request.json
    result = data.get("result")  # "TP" or "SL"
    exit_price = data.get("exit_price")
    
    trade = update_trade_status(trade_id, exit_price, result)
    
    if trade:
        return jsonify({"success": True, "trade": trade})
    return jsonify({"success": False, "error": "Trade not found"}), 404


@app.route('/api/trades/<int:trade_id>', methods=['DELETE'])
def api_close_trade(trade_id):
    """Manually close a trade"""
    for trade in trade_history["open_trades"]:
        if trade["id"] == trade_id:
            trade["status"] = "closed"
            trade["exit_time"] = datetime.now().isoformat()
            trade["result"] = "MANUAL"
            trade_history["trades"].append(trade)
            trade_history["open_trades"].remove(trade)
            return jsonify({"success": True, "trade": trade})
    
    return jsonify({"success": False, "error": "Trade not found"}), 404


@app.route('/trades')
def trades_page():
    """Trade history page"""
    return render_template('trades.html')


@app.route('/analysis')
def analysis_page():
    """Trade analysis page"""
    return render_template('analysis.html')


@app.route('/api/analyze/symbol/<symbol>')
def api_analyze_symbol_full(symbol):
    """Full analysis of a symbol across multiple timeframes"""
    try:
        from src.analysis.stub_analyzer import QuickAnalyzer
        
        # Get Yahoo symbol
        yahoo_symbol = symbol.replace('_', '') + '=X'
        if symbol == 'XAUUSD':
            yahoo_symbol = 'GC=F'
        elif symbol == 'BTCUSDT':
            yahoo_symbol = 'BTC-USD'
        elif symbol == 'ETHUSDT':
            yahoo_symbol = 'ETH-USD'
        elif symbol == 'US30':
            yahoo_symbol = '^DJI'
        elif symbol in ['NAS100', 'SPX500']:
            yahoo_symbol = '^IXIC' if symbol == 'NAS100' else '^GSPC'
        elif symbol == 'USOIL':
            yahoo_symbol = 'CL=F'
        
        # Make a SINGLE call to get 1H data (most recent candles + current price)
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?interval=1h&range=5d"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3) as response:
                data = json.loads(response.read().decode())
                result = data.get('chart', {}).get('result', [{}])
                if result and result[0].get('indicators'):
                    quote = result[0]['indicators']['quote'][0]
                    closes = [c for c in quote.get('close', []) if c is not None]
                    highs = [h for h in quote.get('high', []) if h is not None]
                    lows = [l for l in quote.get('low', []) if l is not None]
                    
                    current_price = closes[-1] if closes else 1.35
                    
                    # Use 1H data for all timeframes (simplified)
                    timeframe_analysis = {}
                    
                    # Calculate trends from available data
                    if closes:
                        recent = closes[-10:] if len(closes) >= 10 else closes
                        ma = sum(recent) / len(recent)
                        high = max(highs[-20:]) if highs else max(closes)
                        low = min(lows[-20:]) if lows else min(closes)
                        range_pct = (high - low) / low * 100 if low else 0
                        
                        trend = "Ranging" if range_pct < 1 else ("Bullish" if current_price > ma else "Bearish")
                        
                        # Set same data for all timeframes
                        for tf in ['1H', '4H', '1D', '1W']:
                            timeframe_analysis[tf] = {
                                "trend": trend,
                                "price": current_price,  # Full precision
                                "ma": round(ma, 2),
                                "high": round(high, 2),
                                "low": round(low, 2),
                                "range_pct": round(range_pct, 2)
                            }
                else:
                    raise Exception("No data")
        except Exception as e:
            logger.error(f"Yahoo API error: {e}")
            current_price = 1.35
            timeframe_analysis = {tf: {"trend": "Unknown", "price": current_price, "ma": current_price, "high": current_price, "low": current_price, "range_pct": 0} for tf in ['1H', '4H', '1D', '1W']}
        # Determine overall market direction
        bullish_count = sum(1 for t in timeframe_analysis.values() if t["trend"] == "Bullish")
        bearish_count = sum(1 for t in timeframe_analysis.values() if t["trend"] == "Bearish")
        
        if bullish_count > bearish_count:
            overall = "Bullish"
        elif bearish_count > bullish_count:
            overall = "Bearish"
        else:
            overall = "Ranging"
        
                # Apply spread adjustment
        adjustment = 0
        if symbol in CUSTOM_SPREADS:
            adjustment = CUSTOM_SPREADS[symbol]
            current_price = current_price - adjustment
            timeframe_analysis = {tf: {**data, "price": data["price"] - adjustment} for tf, data in timeframe_analysis.items()}
        
        return jsonify({
            "success": True,
            "symbol": symbol,
            "current_price": current_price,  # Full precision
            "overall_market": overall,
            "timeframes": timeframe_analysis,
            "signal": None
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/stream/prices')
def stream_prices():
    """Server-Sent Events for live price streaming."""
    from flask import Response
    import json
    
    def generate():
        from src.data.demo_data import demo_generator
        import time
        
        symbols = OANDA_CONFIG["forex_pairs"][:5] + BINANCE_CONFIG["crypto_pairs"][:5]
        
        while True:
            prices = {}
            for symbol in symbols:
                if symbol.endswith('USDT'):
                    prices[symbol] = demo_generator.get_current_price(symbol)
                else:
                    prices[symbol] = demo_generator.get_current_price(symbol)
            
            data = json.dumps({
                "prices": prices,
                "timestamp": datetime.now().isoformat()
            })
            
            yield f"data: {data}\n\n"
            time.sleep(2)  # Update every 2 seconds
    
    return Response(generate(), mimetype='text/event-stream')


def create_app():
    """Create and configure the Flask app."""
    return app


def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask server."""
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server(debug=True)
