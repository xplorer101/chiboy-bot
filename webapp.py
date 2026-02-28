"""
CHIBOY BOT - Web Interface
================================
Flask-based graphical user interface for the trading bot.
ICT Trading Strategy Implementation
"""

import sys
import os
import json
import urllib.request
from pathlib import Path
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session
from functools import wraps
import logging
import threading

# ICT Analysis Functions
def detect_fvg(highs, lows, closes, index=-1):
    """
    Detect Fair Value Gap
    Bearish FVG: current high < previous low (gap down)
    Bullish FVG: current low > previous high (gap up)
    """
    if index < -2 or len(closes) < 3:
        return None
    
    idx = index if index >= 0 else len(closes) + index
    
    if idx < 2 or idx >= len(closes):
        return None
    
    # For bullish: low[index-1] > high[index-2] (gap up)
    # For bearish: high[index-1] < low[index-2] (gap down)
    
    prev_low = lows[idx - 1] if idx > 0 else lows[0]
    prev_high = highs[idx - 1] if idx > 0 else highs[0]
    curr_low = lows[idx]
    curr_high = highs[idx]
    
    # Bullish FVG (gap up)
    if curr_low > prev_high:
        return {
            "type": "bullish",
            "top": curr_low,
            "bottom": prev_high,
            "mid": (curr_low + prev_high) / 2
        }
    # Bearish FVG (gap down)
    elif curr_high < prev_low:
        return {
            "type": "bearish",
            "top": prev_low,
            "bottom": curr_high,
            "mid": (prev_low + curr_high) / 2
        }
    
    return None


def detect_order_blocks(highs, lows, closes, lookback=10):
    """
    Detect Order Blocks
    Bullish OB: Bearish candle that broke structure, followed by upward move
    Bearish OB: Bullish candle that broke structure, followed by downward move
    """
    order_blocks = []
    
    for i in range(lookback, len(closes) - 1):
        # Check if this is a break of structure candle
        prev_high = max(highs[i-lookback:i])
        prev_low = min(lows[i-lookback:i])
        
        # Bullish OB: price broke above previous high then pulled back
        if highs[i] > prev_high and closes[i] > opens[i] if i < len(opens) else closes[i] > (highs[i] + lows[i]) / 2:
            # Check if there's a retracement after
            if i + 3 < len(closes):
                retracement_low = min(lows[i:i+5])
                if retracement_low > lows[i]:  # Pullback stayed above OB
                    order_blocks.append({
                        "type": "bullish",
                        "index": i,
                        "high": highs[i],
                        "low": lows[i],
                        "close": closes[i],
                        "price": lows[i]  # Entry at OB low
                    })
        
        # Bearish OB: price broke below previous low then pulled back
        if lows[i] < prev_low and closes[i] < opens[i] if i < len(opens) else closes[i] < (highs[i] + lows[i]) / 2:
            if i + 3 < len(closes):
                retracement_high = max(highs[i:i+5])
                if retracement_high < highs[i]:  # Pullback stayed below OB
                    order_blocks.append({
                        "type": "bearish",
                        "index": i,
                        "high": highs[i],
                        "low": lows[i],
                        "close": closes[i],
                        "price": highs[i]  # Entry at OB high
                    })
    
    return order_blocks[-5:]  # Return last 5 order blocks


def detect_liquidity_zones(highs, lows, lookback=20):
    """
    Detect Liquidity Zones (sweep areas)
    - Recent highs (liquidity above)
    - Recent lows (liquidity below)
    """
    recent_highs = sorted(set(highs[-lookback:]), reverse=True)[:3]
    recent_lows = sorted(set(lows[-lookback:]))[:3]
    
    return {
        "above": [{"price": h, "type": "liquidity"} for h in recent_highs],
        "below": [{"price": l, "type": "liquidity"} for l in recent_lows]
    }


def detect_market_structure(closes, highs, lows):
    """
    Detect Market Structure (Swing Highs/Lows and BOS)
    """
    structures = []
    
    for i in range(5, len(closes) - 1):
        # Swing high
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1]:
            structures.append({"type": "swing_high", "price": highs[i], "index": i})
        # Swing low
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1]:
            structures.append({"type": "swing_low", "price": lows[i], "index": i})
    
    # Check for recent BOS (Break of Structure)
    recent_swings = structures[-6:] if len(structures) >= 6 else structures
    bos = None
    
    if len(recent_swings) >= 2:
        # Bullish BOS: price broke above recent swing high
        swing_highs = [s for s in recent_swings if s["type"] == "swing_high"]
        if swing_highs and closes[-1] > swing_highs[-1]["price"]:
            bos = {"type": "bullish", "broken_level": swing_highs[-1]["price"]}
        
        # Bearish BOS: price broke below recent swing low
        swing_lows = [s for s in recent_swings if s["type"] == "swing_low"]
        if swing_lows and closes[-1] < swing_lows[-1]["price"]:
            bos = {"type": "bearish", "broken_level": swing_lows[-1]["price"]}
    
    return {
        "structures": structures[-10:],  # Last 10 swing points
        "bos": bos
    }


def analyze_ict_setup(closes, opens, highs, lows):
    """
    Complete ICT Setup Analysis
    Returns trading setup based on ICT methodology
    """
    analysis = {
        "fvgs": [],
        "order_blocks": [],
        "liquidity": {},
        "structure": {},
        "signal": None,
        "entry_price": None,
        "stop_loss": None,
        "take_profit": None,
        "risk_reward": None
    }
    
    # Get FVGs
    for i in [-1, -3, -5]:
        fvg = detect_fvg(highs, lows, closes, i)
        if fvg:
            analysis["fvgs"].append(fvg)
    
    # Get liquidity zones
    analysis["liquidity"] = detect_liquidity_zones(highs, lows)
    
    # Get market structure
    analysis["structure"] = detect_market_structure(closes, highs, lows)
    
    # Determine signal based on structure
    bos = analysis["structure"].get("bos")
    current_price = closes[-1] if closes else 0
    
    if bos:
        if bos["type"] == "bullish":
            # Look for bullish entry
            analysis["signal"] = "BUY"
            # Entry at recent low or FVG
            if analysis["fvgs"] and analysis["fvgs"][0]["type"] == "bullish":
                analysis["entry_price"] = round(analysis["fvgs"][0]["bottom"], 5)
            else:
                analysis["entry_price"] = round(min(lows[-5:]), 5)
            analysis["stop_loss"] = round(min(lows[-10:]) * 0.998, 5)  # 0.2% stop
            # Target: 3R
            risk = analysis["entry_price"] - analysis["stop_loss"]
            analysis["take_profit"] = round(analysis["entry_price"] + (risk * 3), 5)
            analysis["risk_reward"] = "1:3"
            
        elif bos["type"] == "bearish":
            analysis["signal"] = "SELL"
            if analysis["fvgs"] and analysis["fvgs"][0]["type"] == "bearish":
                analysis["entry_price"] = round(analysis["fvgs"][0]["top"], 5)
            else:
                analysis["entry_price"] = round(max(highs[-5:]), 5)
            analysis["stop_loss"] = round(max(highs[-10:]) * 1.002, 5)
            risk = analysis["stop_loss"] - analysis["entry_price"]
            analysis["take_profit"] = round(analysis["entry_price"] - (risk * 3), 5)
            analysis["risk_reward"] = "1:3"
    
    return analysis

# Import database module
import database

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

from datetime import datetime, timedelta
import time as time_module
import re


# Login required decorator - disabled
def login_required(f):
    return f


# Auth routes - disabled
@app.route('/login')
def login_page():
    """Login page - disabled"""
    return redirect('/')

@app.route('/register')
def register_page():
    """Registration page"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        if not username or not password:
            flash('Please enter username and password', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        
        if len(password) < 4:
            flash('Password must be at least 4 characters', 'error')
            return render_template('register.html')
        
        user_id = database.create_user(username, password)
        
        if user_id:
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login_page'))
        else:
            flash('Username already exists', 'error')
    
    return render_template('register.html')


@app.route('/logout')
def logout():
    """Logout route"""
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login_page'))


@app.route('/signals')
def signals_page():
    """Signal history page"""
    return render_template('signals.html')


@app.route('/api/signals', methods=['GET'])
def api_signals():
    """Get signal history"""
    
    symbol = request.args.get('symbol', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    
    signals = database.get_signals(
        user_id=user_id,
        symbol=symbol if symbol else None,
        start_date=start_date if start_date else None,
        end_date=end_date if end_date else None,
        limit=100
    )
    
    signals_list = []
    for s in signals:
        signals_list.append({
            'id': s['id'],
            'symbol': s['symbol'],
            'timeframe': s['timeframe'],
            'direction': s['direction'],
            'entry_price': s['entry_price'],
            'sl': s['sl'],
            'tp': s['tp'],
            'confidence': s['confidence'],
            'reasons': s['reasons'],
            'created_at': s['created_at']
        })
    
    return jsonify({
        'success': True,
        'signals': signals_list,
        'count': len(signals_list)
    })


@app.route('/api/signals', methods=['POST'])
def api_save_signal():
    """Save a new signal"""
    data = request.json
    
    signal_id = database.save_signal(
        user_id=user_id,
        symbol=data.get('symbol'),
        timeframe=data.get('timeframe', '15m'),
        direction=data.get('direction'),
        entry_price=float(data.get('entry_price', 0)),
        sl=float(data.get('sl', 0)),
        tp=float(data.get('tp', 0)),
        confidence=float(data.get('confidence', 0)),
        reasons=data.get('reasons', '')
    )
    
    return jsonify({
        'success': True,
        'signal_id': signal_id
    })

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

# Economic Calendar Cache
economic_cache = {
    "data": [],
    "last_update": None
}

# Market News Cache
news_cache = {
    "data": [],
    "last_update": None
}

CACHE_DURATION = 3600  # 1 hour in seconds


def fetch_economic_calendar():
    """Fetch economic calendar - uses demo data with realistic events"""
    global economic_cache
    
    # Check if cache is still valid
    if economic_cache["last_update"]:
        elapsed = (datetime.now() - datetime.fromisoformat(economic_cache["last_update"])).total_seconds()
        if elapsed < CACHE_DURATION:
            return economic_cache["data"]
    
    # Use demo data with realistic upcoming economic events
    events = get_demo_calendar_data()
    
    economic_cache["data"] = events
    economic_cache["last_update"] = datetime.now().isoformat()
    
    return events


def get_demo_calendar_data():
    """Get demo economic calendar data with realistic upcoming events"""
    today = datetime.now()
    
    # Generate events for today and next 6 days
    events = []
    
    # Today's events (Feb 28) - Typical US session events
    events.append({"date": today.strftime("%Y-%m-%d"), "time": "08:30", "currency": "GBP", "event": "BRC Shop Price Index", "impact": "low", "forecast": "0.5%", "actual": "-"})
    events.append({"date": today.strftime("%Y-%m-%d"), "time": "09:00", "currency": "EUR", "event": "German GfK Consumer Confidence", "impact": "medium", "forecast": "-21.5", "actual": "-"})
    events.append({"date": today.strftime("%Y-%m-%d"), "time": "10:00", "currency": "EUR", "event": "ECB Economic Bulletin", "impact": "medium", "forecast": "-", "actual": "-"})
    events.append({"date": today.strftime("%Y-%m-%d"), "time": "13:30", "currency": "USD", "event": "Core PCE Price Index", "impact": "high", "forecast": "2.8%", "actual": "-"})
    events.append({"date": today.strftime("%Y-%m-%d"), "time": "13:30", "currency": "USD", "event": "Personal Spending", "impact": "medium", "forecast": "0.2%", "actual": "-"})
    events.append({"date": today.strftime("%Y-%m-%d"), "time": "13:30", "currency": "USD", "event": "Personal Income", "impact": "medium", "forecast": "0.3%", "actual": "-"})
    events.append({"date": today.strftime("%Y-%m-%d"), "time": "15:00", "currency": "USD", "event": "Pending Home Sales", "impact": "medium", "forecast": "0.5%", "actual": "-"})
    events.append({"date": today.strftime("%Y-%m-%d"), "time": "15:00", "currency": "USD", "event": "Michigan Consumer Sentiment", "impact": "high", "forecast": "64.5", "actual": "-"})
    
    # Tomorrow
    tomorrow = today + timedelta(days=1)
    events.append({"date": tomorrow.strftime("%Y-%m-%d"), "time": "00:30", "currency": "AUD", "event": "CPI (YoY)", "impact": "high", "forecast": "2.4%", "actual": "-"})
    events.append({"date": tomorrow.strftime("%Y-%m-%d"), "time": "01:30", "currency": "AUD", "event": "RBA Consumer Inflation Expectations", "impact": "medium", "forecast": "4.0%", "actual": "-"})
    events.append({"date": tomorrow.strftime("%Y-%m-%d"), "time": "08:30", "currency": "GBP", "event": "Mortgage Approvals", "impact": "medium", "forecast": "42.5K", "actual": "-"})
    events.append({"date": tomorrow.strftime("%Y-%m-%d"), "time": "09:00", "currency": "EUR", "event": "ECB President Speech", "impact": "high", "forecast": "-", "actual": "-"})
    events.append({"date": tomorrow.strftime("%Y-%m-%d"), "time": "09:30", "currency": "GBP", "event": "GDP (QoQ)", "impact": "high", "forecast": "0.1%", "actual": "-"})
    events.append({"date": tomorrow.strftime("%Y-%m-%d"), "time": "09:30", "currency": "GBP", "event": "GDP (YoY)", "impact": "high", "forecast": "0.5%", "actual": "-"})
    events.append({"date": tomorrow.strftime("%Y-%m-%d"), "time": "13:30", "currency": "CAD", "event": "GDP (MoM)", "impact": "high", "forecast": "0.1%", "actual": "-"})
    events.append({"date": tomorrow.strftime("%Y-%m-%d"), "time": "13:30", "currency": "USD", "event": "Chicago PMI", "impact": "medium", "forecast": "48.0", "actual": "-"})
    
    # Day 2
    day2 = today + timedelta(days=2)
    events.append({"date": day2.strftime("%Y-%m-%d"), "time": "01:30", "currency": "AUD", "event": "Retail Sales (MoM)", "impact": "high", "forecast": "0.3%", "actual": "-"})
    events.append({"date": day2.strftime("%Y-%m-%d"), "time": "04:30", "currency": "JPY", "event": "Tokyo CPI (YoY)", "impact": "medium", "forecast": "2.6%", "actual": "-"})
    events.append({"date": day2.strftime("%Y-%m-%d"), "time": "08:30", "currency": "GBP", "event": "Manufacturing PMI", "impact": "high", "forecast": "49.2", "actual": "-"})
    events.append({"date": day2.strftime("%Y-%m-%d"), "time": "09:30", "currency": "GBP", "event": "Construction PMI", "impact": "medium", "forecast": "51.0", "actual": "-"})
    events.append({"date": day2.strftime("%Y-%m-%d"), "time": "10:00", "currency": "EUR", "event": "CPI (YoY)", "impact": "high", "forecast": "2.4%", "actual": "-"})
    events.append({"date": day2.strftime("%Y-%m-%d"), "time": "13:45", "currency": "USD", "event": "ISM Manufacturing PMI", "impact": "high", "forecast": "48.5", "actual": "-"})
    events.append({"date": day2.strftime("%Y-%m-%d"), "time": "14:00", "currency": "USD", "event": "JOLTS Job Openings", "impact": "medium", "forecast": "8.7M", "actual": "-"})
    
    # Day 3
    day3 = today + timedelta(days=3)
    events.append({"date": day3.strftime("%Y-%m-%d"), "time": "01:30", "currency": "AUD", "event": "Trade Balance", "impact": "medium", "forecast": "5.8B", "actual": "-"})
    events.append({"date": day3.strftime("%Y-%m-%d"), "time": "08:30", "currency": "GBP", "event": "Services PMI", "impact": "high", "forecast": "51.0", "actual": "-"})
    events.append({"date": day3.strftime("%Y-%m-%d"), "time": "09:00", "currency": "EUR", "event": "PPI (MoM)", "impact": "medium", "forecast": "0.2%", "actual": "-"})
    events.append({"date": day3.strftime("%Y-%m-%d"), "time": "10:00", "currency": "EUR", "event": "Retail Sales (MoM)", "impact": "medium", "forecast": "0.1%", "actual": "-"})
    events.append({"date": day3.strftime("%Y-%m-%d"), "time": "13:15", "currency": "USD", "event": "ADP Non-Farm Employment", "impact": "medium", "forecast": "150K", "actual": "-"})
    events.append({"date": day3.strftime("%Y-%m-%d"), "time": "14:00", "currency": "USD", "event": "ISM Services PMI", "impact": "high", "forecast": "52.5", "actual": "-"})
    
    # Day 4
    day4 = today + timedelta(days=4)
    events.append({"date": day4.strftime("%Y-%m-%d"), "time": "00:30", "currency": "AUD", "event": "RBA Interest Rate Decision", "impact": "high", "forecast": "4.35%", "actual": "-"})
    events.append({"date": day4.strftime("%Y-%m-%d"), "time": "08:30", "currency": "GBP", "event": "Bank of England Interest Rate", "impact": "high", "forecast": "5.25%", "actual": "-"})
    events.append({"date": day4.strftime("%Y-%m-%d"), "time": "09:00", "currency": "EUR", "event": "ECB Interest Rate Decision", "impact": "high", "forecast": "4.50%", "actual": "-"})
    events.append({"date": day4.strftime("%Y-%m-%d"), "time": "13:30", "currency": "USD", "event": "Initial Jobless Claims", "impact": "medium", "forecast": "215K", "actual": "-"})
    events.append({"date": day4.strftime("%Y-%m-%d"), "time": "13:30", "currency": "USD", "event": "Non-Farm Productivity", "impact": "medium", "forecast": "1.5%", "actual": "-"})
    
    # Day 5 - NFP Friday March 6th
    day5 = today + timedelta(days=5)
    day6_nfp = datetime(2026, 3, 6)  # Friday March 6th - NFP day
    events.append({"date": day5.strftime("%Y-%m-%d"), "time": "08:30", "currency": "GBP", "event": "Halifax House Prices", "impact": "medium", "forecast": "0.3%", "actual": "-"})
    events.append({"date": day6_nfp.strftime("%Y-%m-%d"), "time": "13:30", "currency": "USD", "event": "NFP Non-Farm Payrolls", "impact": "high", "forecast": "180K", "actual": "-"})
    events.append({"date": day6_nfp.strftime("%Y-%m-%d"), "time": "13:30", "currency": "USD", "event": "Unemployment Rate", "impact": "high", "forecast": "4.0%", "actual": "-"})
    events.append({"date": day6_nfp.strftime("%Y-%m-%d"), "time": "13:30", "currency": "USD", "event": "Average Hourly Earnings", "impact": "high", "forecast": "0.3%", "actual": "-"})
    events.append({"date": day5.strftime("%Y-%m-%d"), "time": "15:00", "currency": "USD", "event": "Factory Orders", "impact": "medium", "forecast": "-0.2%", "actual": "-"})
    
    # Day 6
    day6 = today + timedelta(days=6)
    events.append({"date": day6.strftime("%Y-%m-%d"), "time": "01:30", "currency": "AUD", "event": "RBA Monetary Policy Statement", "impact": "high", "forecast": "-", "actual": "-"})
    events.append({"date": day6.strftime("%Y-%m-%d"), "time": "09:00", "currency": "EUR", "event": "German Factory Orders", "impact": "medium", "forecast": "0.5%", "actual": "-"})
    events.append({"date": day6.strftime("%Y-%m-%d"), "time": "13:30", "currency": "USD", "event": "Consumer Credit", "impact": "medium", "forecast": "15.0B", "actual": "-"})
    
    economic_cache["data"] = events
    economic_cache["last_update"] = datetime.now().isoformat()
    return events


def fetch_market_news():
    """Fetch market news from Yahoo Finance"""
    global news_cache
    
    # Check if cache is still valid
    if news_cache["last_update"]:
        elapsed = (datetime.now() - datetime.fromisoformat(news_cache["last_update"])).total_seconds()
        if elapsed < CACHE_DURATION:
            return news_cache["data"]
    
    try:
        # Fetch from Yahoo Finance
        url = "https://news.yahoo.com/topic/business-news/"
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8')
        
        news_items = []
        
        # Parse news items
        pattern = r'<h3[^>]*class="[^"]*title[^"]*"[^>]*>(.*?)</h3>'
        titles = re.findall(pattern, html, re.DOTALL)
        
        for title in titles[:20]:
            # Clean up the title
            clean_title = re.sub(r'<[^>]+>', '', title).strip()
            if clean_title and len(clean_title) > 10:
                news_items.append({
                    "title": clean_title,
                    "source": "Yahoo Finance",
                    "timestamp": datetime.now().isoformat()
                })
        
        # Alternative parsing
        if not news_items:
            pattern2 = r'"title":"([^"]+)"'
            titles2 = re.findall(pattern2, html)
            for t in titles2[:20]:
                if len(t) > 10 and 'business' in t.lower() or 'market' in t.lower() or 'stock' in t.lower() or 'economy' in t.lower():
                    news_items.append({
                        "title": t,
                        "source": "Yahoo Finance",
                        "timestamp": datetime.now().isoformat()
                    })
        
        if not news_items:
            news_items = get_demo_news_data()
        
        news_cache["data"] = news_items
        news_cache["last_update"] = datetime.now().isoformat()
        
        return news_items
        
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return get_demo_news_data()


def get_demo_news_data():
    """Get demo market news data"""
    news = [
        {"title": "Fed signals potential rate cuts in 2024", "source": "Reuters", "timestamp": datetime.now().isoformat()},
        {"title": "US Dollar weakens as inflation data cools", "source": "Bloomberg", "timestamp": datetime.now().isoformat()},
        {"title": "European markets open mixed amid rate concerns", "source": "CNBC", "timestamp": datetime.now().isoformat()},
        {"title": "Oil prices rise on supply outlook", "source": "Reuters", "timestamp": datetime.now().isoformat()},
        {"title": "Tech stocks rally on strong earnings", "source": "Yahoo Finance", "timestamp": datetime.now().isoformat()},
        {"title": "Gold reaches new highs as safe haven", "source": "Kitco News", "timestamp": datetime.now().isoformat()},
        {"title": "Bitcoin surges past resistance levels", "source": "CoinDesk", "timestamp": datetime.now().isoformat()},
        {"title": "US Treasury yields fall on dovish Fed comments", "source": "MarketWatch", "timestamp": datetime.now().isoformat()},
    ]
    news_cache["data"] = news
    news_cache["last_update"] = datetime.now().isoformat()
    return news


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
        elif symbol == 'SOLUSDT':
            yahoo_symbol = 'SOL-USD'
        elif symbol == 'US30':
            yahoo_symbol = '^DJI'
        elif symbol in ['NAS100', 'SPX500']:
            yahoo_symbol = '^IXIC' if symbol == 'NAS100' else '^GSPC'
        elif symbol == 'USOIL':
            yahoo_symbol = 'CL=F'
        
        # Fetch data for multiple timeframes
        timeframe_data = {}
        
        # Define intervals to fetch
        intervals = {
            '1H': '1h',
            '4H': '4h',
            '1D': '1d',
            '1W': '1wk'
        }
        
        for tf_name, interval in intervals.items():
            try:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?interval={interval}&range=30d"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=3) as response:
                    data = json.loads(response.read().decode())
                    result = data.get('chart', {}).get('result', [{}])
                    if result and result[0].get('indicators'):
                        quote = result[0]['indicators']['quote'][0]
                        tf_closes = [c for c in quote.get('close', []) if c is not None]
                        tf_highs = [h for h in quote.get('high', []) if h is not None]
                        tf_lows = [l for l in quote.get('low', []) if l is not None]
                        tf_opens = [o for o in quote.get('open', []) if o is not None]
                        
                        if tf_closes:
                            tf_price = tf_closes[-1]
                            tf_ma = sum(tf_closes[-10:]) / min(10, len(tf_closes))
                            tf_high = max(tf_highs[-20:]) if tf_highs else tf_price
                            tf_low = min(tf_lows[-20:]) if tf_lows else tf_price
                            tf_range = (tf_high - tf_low) / tf_low * 100 if tf_low else 0
                            tf_trend = "Ranging" if tf_range < 1 else ("Bullish" if tf_price > tf_ma else "Bearish")
                            
                            timeframe_data[tf_name] = {
                                "trend": tf_trend,
                                "price": tf_price,
                                "ma": round(tf_ma, 5),
                                "high": round(tf_high, 5),
                                "low": round(tf_low, 5),
                                "range_pct": round(tf_range, 2),
                                "closes": tf_closes,
                                "highs": tf_highs,
                                "lows": tf_lows,
                                "opens": tf_opens
                            }
            except Exception as e:
                logger.info(f"Failed to fetch {tf_name} data: {e}")
                continue
        
        # Set current price from 1H data
        if '1H' in timeframe_data:
            current_price = timeframe_data['1H'].get('price', 1.35)
        else:
            current_price = 1.35
        
        # Use 1H data for current_price and ICT
        timeframe_analysis = {}
        
        # Calculate 1H analysis from timeframe_data
        if '1H' in timeframe_data:
            tf_data = timeframe_data['1H']
            timeframe_analysis['1H'] = {
                "trend": tf_data["trend"],
                "price": tf_data["price"],
                "ma": tf_data["ma"],
                "high": tf_data["high"],
                "low": tf_data["low"],
                "range_pct": tf_data["range_pct"]
            }
        
        # Merge with other timeframes
        for tf in ['4H', '1D', '1W']:
            if tf in timeframe_data:
                tf_data = timeframe_data[tf]
                timeframe_analysis[tf] = {
                    "trend": tf_data["trend"],
                    "price": tf_data["price"],
                    "ma": tf_data["ma"],
                    "high": tf_data["high"],
                    "low": tf_data["low"],
                    "range_pct": tf_data["range_pct"]
                }
            else:
                # Fallback to 1H data
                timeframe_analysis[tf] = timeframe_analysis.get('1H', {"trend": "Unknown", "price": current_price, "ma": current_price, "high": current_price, "low": current_price, "range_pct": 0})
        
        # Ensure 1H exists
        if '1H' not in timeframe_analysis:
            timeframe_analysis['1H'] = {"trend": "Unknown", "price": current_price, "ma": current_price, "high": current_price, "low": current_price, "range_pct": 0}
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
        
        # ICT Analysis - use 1H data
        if '1H' in timeframe_data and len(timeframe_data['1H'].get('closes', [])) > 10:
            tf_data = timeframe_data['1H']
            ict_analysis = analyze_ict_setup(tf_data.get('closes', []), tf_data.get('opens', []), tf_data.get('highs', []), tf_data.get('lows', []))
        else:
            ict_analysis = {
                "signal": None, "entry_price": None, "stop_loss": None, "take_profit": None, "risk_reward": None,
                "fvgs": [], "order_blocks": [], "liquidity": {}, "structure": {}
            }
        
        return jsonify({
            "success": True,
            "symbol": symbol,
            "current_price": current_price,  # Full precision
            "overall_market": overall,
            "timeframes": timeframe_analysis,
            "signal": ict_analysis.get("signal"),
            "ict": {
                "signal": ict_analysis.get("signal"),
                "entry_price": ict_analysis.get("entry_price"),
                "stop_loss": ict_analysis.get("stop_loss"),
                "take_profit": ict_analysis.get("take_profit"),
                "risk_reward": ict_analysis.get("risk_reward"),
                "fvgs": ict_analysis.get("fvgs", []),
                "liquidity": ict_analysis.get("liquidity", {}),
                "structure": ict_analysis.get("structure", {})
            }
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/sentiment')
def sentiment_page():
    """Market sentiment page"""
    return render_template('sentiment.html')


@app.route('/api/sentiment')
def api_sentiment():
    """Get currency strength and sentiment data from Yahoo Finance"""
    import urllib.request
    import json
    from datetime import datetime
    
    currency_strength = {}
    sentiment_data = []
    
    # Major pairs for calculating currency strength
    pairs = [
        ('EURUSD=X', 'EUR', 'USD'),
        ('GBPUSD=X', 'GBP', 'USD'),
        ('USDJPY=X', 'USD', 'JPY'),
        ('USDCAD=X', 'USD', 'CAD'),
        ('USDCHF=X', 'USD', 'CHF'),
        ('AUDUSD=X', 'AUD', 'USD'),
    ]
    
    for yahoo, base, quote in pairs:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo}?interval=1d&range=5d"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                result = data.get('chart', {}).get('result', [])
                if result and result[0].get('indicators', {}).get('quote'):
                    closes = result[0]['indicators']['quote'][0].get('close', [])
                    if len(closes) >= 2 and closes[-1] and closes[-2]:
                        change = ((closes[-1] - closes[-2]) / closes[-2]) * 100
                        if base not in currency_strength:
                            currency_strength[base] = {'strength': 50, 'pairs': 0}
                        if quote not in currency_strength:
                            currency_strength[quote] = {'strength': 50, 'pairs': 0}
                        currency_strength[base]['strength'] += change * 2
                        currency_strength[quote]['strength'] -= change * 2
                        currency_strength[base]['pairs'] += 1
                        currency_strength[quote]['pairs'] += 1
        except:
            continue
    
    # Fallback if no data
    if not currency_strength:
        for c in ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF']:
            currency_strength[c] = {'strength': 50, 'pairs': 1}
    
    # Normalize
    values = [v['strength'] for v in currency_strength.values()]
    if values:
        min_val, max_val = min(values), max(values)
    else:
        min_val, max_val = 0, 1
    
    for currency in currency_strength:
        raw = currency_strength[currency]['strength']
        normalized = ((raw - min_val) / (max_val - min_val) * 100) if max_val != min_val else 50
        currency_strength[currency]['strength'] = round(normalized, 1)
        currency_strength[currency]['change'] = round(raw - 50, 1)
    
    # Symbol sentiment - Forex, Gold, and Crypto
    symbol_pairs = [
        ('GC=F', 'XAUUSD'), 
        ('EURUSD=X', 'EURUSD'), 
        ('GBPUSD=X', 'GBPUSD'), 
        ('USDJPY=X', 'USDJPY'),
        ('BTC-USD', 'BTCUSDT'),
        ('ETH-USD', 'ETHUSDT'),
        ('SOL-USD', 'SOLUSDT'),
    ]
    for yahoo, sym in symbol_pairs:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo}?interval=1d&range=5d"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                result = data.get('chart', {}).get('result', [])
                if result and result[0].get('indicators', {}).get('quote'):
                    closes = result[0]['indicators']['quote'][0].get('close', [])
                    if len(closes) >= 2 and closes[-1] and closes[-2]:
                        change = ((closes[-1] - closes[-2]) / closes[-2]) * 100
                        sentiment_data.append({
                            'symbol': sym,
                            'long': round(50 + change * 5, 1),
                            'short': round(50 - change * 5, 1),
                            'bullish': change > 0,
                            'change': round(change, 2)
                        })
        except:
            continue
    
    if not sentiment_data:
        for sym in ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
            sentiment_data.append({'symbol': sym, 'long': 50, 'short': 50, 'bullish': True, 'change': 0})
    
    return jsonify({
        'success': True,
        'timestamp': datetime.now().isoformat(),
        'currencies': currency_strength,
        'symbols': sentiment_data,
        'overall_sentiment': 'Bullish' if sum(s['long'] for s in sentiment_data) / len(sentiment_data) > 50 else 'Bearish'
    })


@app.route('/api/heatmap')
def api_heatmap():
    from flask import jsonify
    try:
        response = api_sentiment()
        data = response.get_json()
        currencies = data.get('currencies', {})
        heatmap_data = []
        for currency, info in currencies.items():
            strength = info.get('strength', 50)
            color = '#3fb950' if strength >= 60 else '#f0b429' if strength >= 40 else '#f85149'
            heatmap_data.append({
                'currency': currency,
                'strength': strength,
                'color': color,
                'change': info.get('change', 0)
            })
        heatmap_data.sort(key=lambda x: x['strength'], reverse=True)
        return jsonify({'success': True, 'data': heatmap_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    run_server(debug=True)
if __name__ == '__main__':
    run_server(debug=True)

@app.route('/calendar')
def calendar_page():
    """Economic calendar page"""
    return render_template('calendar.html')

@app.route('/api/economic-calendar')
def economic_calendar():
    """Get economic calendar events from Forex Factory"""
    data = fetch_economic_calendar()
    if isinstance(data, list):
        return jsonify({'success': True, 'events': data, 'date': datetime.now().strftime('%Y-%m-%d')})
    return jsonify(data)

