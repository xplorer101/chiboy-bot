"""
CHIBOY BOT - Web Interface
================================
Flask-based graphical user interface for the trading bot.
"""

import sys
import os
from pathlib import Path
from flask import Flask, render_template, jsonify, request
import logging
import threading

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import BOT_CONFIG, TRADING_MODE, OANDA_CONFIG, BINANCE_CONFIG
from src.analysis.multi_timeframe import MultiTimeframeAnalyzer
from src.signals.trading_signals import TradingSignals, SignalManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = 'chiboy-bot-secret-key'

# Initialize components
analyzer = MultiTimeframeAnalyzer()
signals_manager = SignalManager()
trading_signals = TradingSignals()

from datetime import datetime

# Store analysis results
analysis_cache = {
    "last_update": None,
    "results": [],
    "opportunities": []
}


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
    """Run analysis on all markets."""
    try:
        opportunities = analyzer.get_top_opportunities(limit=10)
        
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
    """Analyze a specific symbol."""
    asset_type = request.args.get('type', 'forex')
    
    try:
        if asset_type == 'crypto':
            result = analyzer.analyze_crypto_pair(symbol)
        else:
            result = analyzer.analyze_forex_pair(symbol)
        
        report = analyzer.format_analysis_report(result)
        
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


def create_app():
    """Create and configure the Flask app."""
    return app


def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask server."""
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server(debug=True)
