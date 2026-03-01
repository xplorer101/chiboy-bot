"""
CHIBOY BOT - Web Interface
================================
Flask-based graphical user interface for the trading bot.
ICT Trading Strategy Implementation - Professional Edition
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
from datetime import datetime, timedelta

# ICT Analysis Functions - Professional Implementation
def detect_swing_points(highs, lows, lookback=5):
    """Detect swing highs and lows"""
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(highs) - lookback):
        # Swing high: higher than both neighbors
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append({"price": highs[i], "index": i})
        # Swing low: lower than both neighbors
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append({"price": lows[i], "index": i})
    
    return {"swing_highs": swing_highs, "swing_lows": swing_lows}


def detect_equal_levels(prices, tolerance=0.001):
    """Detect equal highs or lows within tolerance"""
    equal_levels = []
    for i in range(len(prices)):
        for j in range(i+1, len(prices)):
            if abs(prices[i] - prices[j]) / prices[i] < tolerance:
                level = (prices[i] + prices[j]) / 2
                if level not in [e["price"] for e in equal_levels]:
                    equal_levels.append({"price": level, "count": 2})
    return equal_levels


def detect_fvg(highs, lows, closes, index=-1):
    """
    Detect Fair Value Gap
    Bullish FVG: current low > previous high (gap up)
    Bearish FVG: current high < previous low (gap down)
    """
    if index < -2 or len(closes) < 3:
        return None
    
    idx = index if index >= 0 else len(closes) + index
    
    if idx < 2 or idx >= len(closes):
        return None
    
    prev_low = lows[idx - 1]
    prev_high = highs[idx - 1]
    curr_low = lows[idx]
    curr_high = highs[idx]
    
    # Bullish FVG (gap up)
    if curr_low > prev_high:
        return {
            "type": "bullish",
            "top": curr_low,
            "bottom": prev_high,
            "mid": (curr_low + prev_high) / 2,
            "index": idx
        }
    # Bearish FVG (gap down)
    elif curr_high < prev_low:
        return {
            "type": "bearish",
            "top": prev_low,
            "bottom": curr_high,
            "mid": (prev_low + curr_high) / 2,
            "index": idx
        }
    
    return None


def detect_order_block(highs, lows, closes, opens, direction="bullish"):
    """
    Detect valid Order Block
    Bullish OB: Bearish candle that broke structure up, then price pulled back
    Bearish OB: Bullish candle that broke structure down, then price pulled back
    """
    order_blocks = []
    
    for i in range(10, len(closes) - 5):
        is_bearish_candle = closes[i] < opens[i] if i < len(opens) else closes[i] < (highs[i] + lows[i]) / 2
        is_bullish_candle = closes[i] > opens[i] if i < len(opens) else closes[i] > (highs[i] + lows[i]) / 2
        
        # Get recent swing points
        recent_highs = highs[max(0, i-15):i]
        recent_lows = lows[max(0, i-15):i]
        prev_swing_high = max(recent_highs) if recent_highs else highs[i]
        prev_swing_low = min(recent_lows) if recent_lows else lows[i]
        
        if direction == "bullish":
            # Bullish OB: Bearish candle broke above previous swing high, then pulled back
            if is_bearish_candle and highs[i] > prev_swing_high:
                # Check if price pulled back (retraced)
                retracement_low = min(lows[i+1:i+5]) if i+5 <= len(lows) else lows[i]
                if retracement_low > lows[i]:  # Pullback stays above OB
                    order_blocks.append({
                        "type": "bullish",
                        "index": i,
                        "high": highs[i],
                        "low": lows[i],
                        "close": closes[i],
                        "entry_price": lows[i],  # Entry at low of OB
                        "stop_loss": lows[i] - (highs[i] - lows[i]),  # 1:1 ATR stop
                        "timeframe": "1H"
                    })
        else:
            # Bearish OB: Bullish candle broke below previous swing low, then pulled back
            if is_bullish_candle and lows[i] < prev_swing_low:
                retracement_high = max(highs[i+1:i+5]) if i+5 <= len(highs) else highs[i]
                if retracement_high < highs[i]:  # Pullback stays below OB
                    order_blocks.append({
                        "type": "bearish",
                        "index": i,
                        "high": highs[i],
                        "low": lows[i],
                        "close": closes[i],
                        "entry_price": highs[i],  # Entry at high of OB
                        "stop_loss": highs[i] + (highs[i] - lows[i]),  # 1:1 ATR stop
                        "timeframe": "1H"
                    })
    
    return order_blocks[-3:] if order_blocks else []


def detect_liquidity_zones(highs, lows, closes):
    """
    Detect all liquidity zones:
    - Equal highs (4H, Daily, Weekly)
    - Equal lows (4H, Daily, Weekly)
    - Previous day high/low
    - Previous week high/low
    """
    liquidity = {
        "equal_highs": [],
        "equal_lows": [],
        "prev_day_high": None,
        "prev_day_low": None,
        "prev_week_high": None,
        "prev_week_low": None,
        "swing_highs": [],
        "swing_lows": []
    }
    
    if len(highs) < 24:  # Need enough data
        return liquidity
    
    # Previous day high/low (last 24 candles = ~4H on 1H chart for forex)
    # Assuming 1H candles, 24 = 1 day
    liquidity["prev_day_high"] = max(highs[-24:])
    liquidity["prev_day_low"] = min(lows[-24:])
    
    # Previous week high/low (last 120 candles = ~5 days)
    if len(highs) >= 120:
        liquidity["prev_week_high"] = max(highs[-120:-24])
        liquidity["prev_week_low"] = min(lows[-120:-24])
    
    # Equal highs (within 0.1% tolerance)
    recent_highs = highs[-20:]
    recent_lows = lows[-20:]
    
    for i in range(len(recent_highs)):
        for j in range(i+1, len(recent_highs)):
            if abs(recent_highs[i] - recent_highs[j]) / recent_highs[i] < 0.001:
                level = (recent_highs[i] + recent_highs[j]) / 2
                if not any(abs(l["price"] - level) / level < 0.001 for l in liquidity["equal_highs"]):
                    liquidity["equal_highs"].append({"price": level, "type": "equal_high"})
            
            if abs(recent_lows[i] - recent_lows[j]) / recent_lows[i] < 0.001:
                level = (recent_lows[i] + recent_lows[j]) / 2
                if not any(abs(l["price"] - level) / level < 0.001 for l in liquidity["equal_lows"]):
                    liquidity["equal_lows"].append({"price": level, "type": "equal_low"})
    
    # Recent swing highs/lows
    swings = detect_swing_points(highs, lows)
    liquidity["swing_highs"] = [{"price": h["price"], "type": "swing_high"} for h in swings["swing_highs"][-5:]]
    liquidity["swing_lows"] = [{"price": l["price"], "type": "swing_low"} for l in swings["swing_lows"][-5:]]
    
    return liquidity


def detect_market_structure(closes, highs, lows):
    """
    Detect Market Structure: Swing points, CHoCH, BOS
    """
    swings = detect_swing_points(highs, lows)
    
    # Find most recent swing points
    swing_highs = swings["swing_highs"]
    swing_lows = swings["swing_lows"]
    
    if not swing_highs or not swing_lows:
        return {"structures": [], "bos": None, "trend": "Ranging"}
    
    # Determine current structure
    last_high = swing_highs[-1]["price"] if swing_highs else None
    last_low = swing_lows[-1]["price"] if swing_lows else None
    
    # Check for BOS (Break of Structure)
    bos = None
    current_price = closes[-1]
    
    # Bullish BOS: price broke above last swing high
    if last_high and current_price > last_high:
        bos = {"type": "bullish", "broken_level": last_high, "current_price": current_price}
    
    # Bearish BOS: price broke below last swing low
    elif last_low and current_price < last_low:
        bos = {"type": "bearish", "broken_level": last_low, "current_price": current_price}
    
    # Determine trend
    if last_high and last_low:
        if last_high > (swing_highs[-2]["price"] if len(swing_highs) > 1 else last_high):
            trend = "Bullish"
        elif last_low < (swing_lows[-2]["price"] if len(swing_lows) > 1 else last_low):
            trend = "Bearish"
        else:
            trend = "Ranging"
    else:
        trend = "Unknown"
    
    return {
        "structures": {
            "swing_highs": swing_highs[-5:],
            "swing_lows": swing_lows[-5:]
        },
        "bos": bos,
        "trend": trend,
        "last_swing_high": last_high,
        "last_swing_low": last_low
    }


def check_liquidity_sweep(current_price, liquidity, direction="bullish"):
    """
    Check if liquidity has been swept
    For bullish: check if previous lows were swept
    For bearish: check if previous highs were swept
    """
    swept = False
    swept_level = None
    
    if direction == "bullish":
        # Check if price swept below recent lows (liquidity grab)
        all_lows = []
        if liquidity.get("equal_lows"):
            all_lows.extend([l["price"] for l in liquidity["equal_lows"]])
        if liquidity.get("swing_lows"):
            all_lows.extend([l["price"] for l in liquidity["swing_lows"]])
        if liquidity.get("prev_day_low"):
            all_lows.append(liquidity["prev_day_low"])
        
        # Check if any were recently swept (price went below and came back)
        for low in all_lows:
            if current_price < low:  # Price swept below
                swept = True
                swept_level = low
                break
    else:
        # Check if price swept above recent highs
        all_highs = []
        if liquidity.get("equal_highs"):
            all_highs.extend([h["price"] for h in liquidity["equal_highs"]])
        if liquidity.get("swing_highs"):
            all_highs.extend([h["price"] for h in liquidity["swing_highs"]])
        if liquidity.get("prev_day_high"):
            all_highs.append(liquidity["prev_day_high"])
        
        for high in all_highs:
            if current_price > high:  # Price swept above
                swept = True
                swept_level = high
                break
    
    return {"swept": swept, "level": swept_level}


def _calculate_pips(price1: float, price2: float) -> float:
    """Calculate distance in pips between two prices"""
    # For forex pairs (not JPY): 1 pip = 0.0001
    # For JPY pairs: 1 pip = 0.01
    # For crypto: varies, use 0.0001 as base
    
    # Check if JPY pair (price around 100-200)
    is_jpy = price1 > 50 or price2 > 50
    
    if is_jpy:
        return abs(price1 - price2) * 100  # JPY pips
    else:
        return abs(price1 - price2) * 10000  # Standard pips


def _filter_ob_by_distance(obs: list, current_price: float, min_pips: int = 30, max_pips: int = 50) -> list:
    """Filter order blocks to only those within 30-50 pips of current price"""
    filtered = []
    for ob in obs:
        entry = ob.get("entry_price", 0)
        if entry <= 0:
            continue
        pips = _calculate_pips(entry, current_price)
        if min_pips <= pips <= max_pips:
            ob["distance_pips"] = round(pips, 1)
            filtered.append(ob)
    # Sort by distance - prefer closer OBs
    filtered.sort(key=lambda x: x.get("distance_pips", 999))
    return filtered


def _find_swing_for_stop_loss(closes: list, highs: list, lows: list, direction: str, current_price: float) -> float:
    """Find proper stop loss based on recent swing low/high"""
    if len(highs) < 10 or len(lows) < 10:
        return 0
    
    # Get recent swing points
    swing_highs = []
    swing_lows = []
    
    for i in range(5, len(highs) - 5):
        # Swing high
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append(highs[i])
        # Swing low
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append(lows[i])
    
    if direction == "bullish":
        # For long: SL goes below recent swing low
        if swing_lows:
            recent_swing_low = min(swing_lows[-3:])  # Use last 3 swing lows
            # Ensure SL is below current price
            if recent_swing_low < current_price:
                return recent_swing_low
        # Fallback: use recent low
        return min(lows[-10:])
    else:
        # For short: SL goes above recent swing high
        if swing_highs:
            recent_swing_high = max(swing_highs[-3:])
            if recent_swing_high > current_price:
                return recent_swing_high
        # Fallback: use recent high
        return max(highs[-10:])


def _check_mss_requirement(closes: list, highs: list, lows: list, direction: str) -> dict:
    """
    Check for Market Structure Shift (MSS/BOS) requirement
    Returns dict with 'found' and details
    """
    if len(closes) < 20:
        return {"found": False, "reason": "Insufficient data"}
    
    current_price = closes[-1]
    
    # Find swing points
    swing_highs = []
    swing_lows = []
    
    for i in range(5, len(highs) - 5):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append({"price": highs[i], "index": i})
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append({"price": lows[i], "index": i})
    
    if not swing_highs or not swing_lows:
        return {"found": False, "reason": "No clear swing points"}
    
    last_swing_high = swing_highs[-1]["price"] if swing_highs else None
    last_swing_low = swing_lows[-1]["price"] if swing_lows else None
    
    if direction == "bullish":
        # Bullish MSS: Price breaking ABOVE last swing high
        if last_swing_high and current_price > last_swing_high:
            return {
                "found": True,
                "type": "Bullish MSS/BOS",
                "broken_level": last_swing_high,
                "current_price": current_price,
                "strength": round((current_price - last_swing_high) / last_swing_high * 10000, 1)
            }
    else:
        # Bearish MSS: Price breaking BELOW last swing low
        if last_swing_low and current_price < last_swing_low:
            return {
                "found": True,
                "type": "Bearish MSS/BOS",
                "broken_level": last_swing_low,
                "current_price": current_price,
                "strength": round((last_swing_low - current_price) / last_swing_low * 10000, 1)
            }
    
    return {"found": False, "reason": f"No MSS - price not broken {direction}"}


def analyze_ict_setup_full(timeframe_data, htf_data=None):
    """
    Complete ICT Setup Analysis with all requirements:
    1. Only select order blocks within 30-50 pips of current price
    2. Prefer discount entries (below current price for longs) over premium
    3. Add FVG as backup entry if no good OB is found
    4. Add Market Structure Shift (MSS) requirement before showing any signal
    5. Calculate proper stop loss based on recent swing low/high
    """
    
    # Use 1H data as primary
    tf = timeframe_data.get("1H", {})
    closes = tf.get("closes", [])
    highs = tf.get("highs", [])
    lows = tf.get("lows", [])
    opens = tf.get("opens", [])
    
    if len(closes) < 20:
        return {"signal": None, "reason": "Insufficient data", "reasons": ["Insufficient price data"]}
    
    current_price = closes[-1]
    
    # Get structure info
    structure = detect_market_structure(closes, highs, lows)
    liquidity = detect_liquidity_zones(highs, lows, closes)
    
    # Detect FVGs
    fvgs = []
    for i in [-1, -3, -5]:
        fvg = detect_fvg(highs, lows, closes, i)
        if fvg:
            fvgs.append(fvg)
    
    # Detect Order Blocks
    bullish_obs = detect_order_block(highs, lows, closes, opens, "bullish")
    bearish_obs = detect_order_block(highs, lows, closes, opens, "bearish")
    
    # Filter OBs to only those within 30-50 pips of current price
    bullish_obs_filtered = _filter_ob_by_distance(bullish_obs, current_price, 30, 50)
    bearish_obs_filtered = _filter_ob_by_distance(bearish_obs, current_price, 30, 50)
    
    # Get structure info
    last_high = structure.get("last_swing_high")
    last_low = structure.get("last_swing_low")
    htf_trend = structure.get("trend", "Ranging")
    
    # Initialize reasons list
    reasons = []
    trade_direction = None
    entry_price = None
    stop_loss = None
    take_profit = None
    
    # Check liquidity sweep status
    liquidity_swept_bullish = check_liquidity_sweep(current_price, liquidity, "bullish")
    liquidity_swept_bearish = check_liquidity_sweep(current_price, liquidity, "bearish")
    
    # Check MSS requirement FIRST - this is mandatory
    mss_bullish = _check_mss_requirement(closes, highs, lows, "bullish")
    mss_bearish = _check_mss_requirement(closes, highs, lows, "bearish")
    
    # BULLISH SETUP Requirements:
    # 1. MSS confirmed (price broke above last swing high) - MANDATORY
    # 2. Liquidity swept (prev lows taken) OR approaching liquidity
    # 3. Valid Bullish OB available within 30-50 pips (discount preferred)
    
    # Check if we have MSS for bullish
    has_bullish_mss = mss_bullish.get("found", False)
    
    if has_bullish_mss:
        reasons.append(f"✓ MSS: {mss_bullish['type']} - broke {mss_bullish.get('broken_level', 0):.5f}")
        
        # Check liquidity
        if liquidity_swept_bullish["swept"]:
            reasons.append("✓ Liquidity swept (lows taken)")
        elif last_low:
            reasons.append(f"📍 Near liquidity zone at {last_low:.5f}")
        
        # Look for bullish OB - DISCOUNT ONLY (below current price)
        # Sort filtered OBs to prefer discount (below price)
        sorted_bullish = sorted(
            bullish_obs_filtered, 
            key=lambda x: (x.get("entry_price", 0) > current_price, abs(x.get("entry_price", 0) - current_price))
        )
        
        for ob in sorted_bullish:
            ob_entry = ob.get("entry_price", 0)
            
            # FOR BULLISH: MUST be discount (below or at current price)
            if ob_entry <= current_price * 1.0005:  # Allow tiny premium (0.05%)
                # Calculate SL: use swing low OR OB's built-in SL (whichever is tighter but below entry)
                swing_sl = _find_swing_for_stop_loss(closes, highs, lows, "bullish", current_price)
                ob_calculated_sl = ob.get("stop_loss", 0)  # This is below entry
                
                # For long: SL must be below entry
                if swing_sl > 0 and swing_sl < ob_entry:
                    stop_loss = swing_sl
                elif ob_calculated_sl > 0 and ob_calculated_sl < ob_entry:
                    stop_loss = ob_calculated_sl
                else:
                    # Fallback: use swing low or OB low minus buffer
                    stop_loss = round(min(swing_sl if swing_sl > 0 else ob_entry, ob_entry - 0.001), 5)
                
                entry_price = round(ob_entry, 5)
                risk = entry_price - stop_loss
                
                # Only proceed if valid risk
                if risk > 0:
                    take_profit = round(entry_price + (risk * 3), 5)
                    trade_direction = "BUY"
                    reasons.append(f"✓ Bullish OB (discount) at {entry_price:.5f} ({ob.get('distance_pips', '?')} pips)")
                    reasons.append(f"✓ Entry at Order Block (discount)")
                    break
    
    # BEARISH SETUP Requirements
    # Check if we have MSS for bearish
    has_bearish_mss = mss_bearish.get("found", False)
    
    if not trade_direction and has_bearish_mss:
        reasons.append(f"✓ MSS: {mss_bearish['type']} - broke {mss_bearish.get('broken_level', 0):.5f}")
        
        # Check liquidity
        if liquidity_swept_bearish["swept"]:
            reasons.append("✓ Liquidity swept (highs taken)")
        elif last_high:
            reasons.append(f"📍 Near liquidity zone at {last_high:.5f}")
        
        # Look for bearish OB - PREMIUM ONLY (above current price)
        sorted_bearish = sorted(
            bearish_obs_filtered,
            key=lambda x: (x.get("entry_price", 0) < current_price, abs(x.get("entry_price", 0) - current_price))
        )
        
        for ob in sorted_bearish:
            ob_entry = ob.get("entry_price", 0)
            
            # FOR BEARISH: MUST be premium (above or at current price)
            if ob_entry >= current_price * 0.9995:  # Allow tiny discount
                # For short: SL must be above entry
                swing_sl = _find_swing_for_stop_loss(closes, highs, lows, "bearish", current_price)
                ob_calculated_sl = ob.get("stop_loss", 0)  # This is above entry
                
                if swing_sl > 0 and swing_sl > ob_entry:
                    stop_loss = swing_sl
                elif ob_calculated_sl > 0 and ob_calculated_sl > ob_entry:
                    stop_loss = ob_calculated_sl
                else:
                    # Fallback
                    stop_loss = round(max(swing_sl if swing_sl > 0 else ob_entry, ob_entry + 0.001), 5)
                
                entry_price = round(ob_entry, 5)
                risk = stop_loss - entry_price
                
                if risk > 0:
                    take_profit = round(entry_price - (risk * 3), 5)
                    trade_direction = "SELL"
                    reasons.append(f"✓ Bearish OB (premium) at {entry_price:.5f} ({ob.get('distance_pips', '?')} pips)")
                    reasons.append(f"✓ Entry at Order Block (premium)")
                    break
    
    # FVG BACKUP: If no good OB found but MSS exists, try FVG
    if not trade_direction:
        if has_bullish_mss:
            # Look for bullish FVG within 30-50 pips
            for fvg in fvgs:
                if fvg.get("type") != "bullish":
                    continue
                fvg_mid = fvg.get("mid", 0)
                fvg_pips = _calculate_pips(fvg_mid, current_price)
                
                if 30 <= fvg_pips <= 50 and fvg_mid <= current_price:
                    # Use FVG as entry point
                    entry_price = round(fvg_mid, 5)
                    # SL below the FVG bottom
                    fvg_bottom = fvg.get("bottom", 0)
                    swing_sl = _find_swing_for_stop_loss(closes, highs, lows, "bullish", current_price)
                    stop_loss = min(swing_sl, fvg_bottom) if swing_sl > 0 else round(fvg_bottom, 5)
                    
                    risk = entry_price - stop_loss
                    if risk > 0:
                        take_profit = round(entry_price + (risk * 3), 5)
                        trade_direction = "BUY"
                        reasons.append(f"✓ FVG POI (backup) at {entry_price:.5f} ({fvg_pips:.1f} pips)")
                        break
        
        elif has_bearish_mss:
            # Look for bearish FVG within 30-50 pips
            for fvg in fvgs:
                if fvg.get("type") != "bearish":
                    continue
                fvg_mid = fvg.get("mid", 0)
                fvg_pips = _calculate_pips(fvg_mid, current_price)
                
                if 30 <= fvg_pips <= 50 and fvg_mid >= current_price:
                    entry_price = round(fvg_mid, 5)
                    fvg_top = fvg.get("top", 0)
                    swing_sl = _find_swing_for_stop_loss(closes, highs, lows, "bearish", current_price)
                    stop_loss = max(swing_sl, fvg_top) if swing_sl > 0 else round(fvg_top, 5)
                    
                    risk = stop_loss - entry_price
                    if risk > 0:
                        take_profit = round(entry_price - (risk * 3), 5)
                        trade_direction = "SELL"
                        reasons.append(f"✓ FVG POI (backup) at {entry_price:.5f} ({fvg_pips:.1f} pips)")
                        break
    
    # If still no signal - add helpful message
    if not trade_direction:
        if not has_bullish_mss and not has_bearish_mss:
            reasons.append("✗ No MSS - price hasn't broken swing point yet")
        if not bullish_obs_filtered and not bearish_obs_filtered:
            reasons.append("✗ No valid OB within 30-50 pips")
        if has_bullish_mss and not bullish_obs_filtered:
            reasons.append("✗ No discount OB found (try FVG backup)")
        if has_bearish_mss and not bearish_obs_filtered:
            reasons.append("✗ No premium OB found (try FVG backup)")
    
    # Calculate R:R
    risk_reward = None
    if entry_price and stop_loss and take_profit:
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        if risk > 0:
            risk_reward = round(reward / risk, 1)
    
    return {
        "signal": trade_direction,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward": f"1:{risk_reward}" if risk_reward else None,
        "reasons": reasons,
        "structure": structure,
        "mss_check": {
            "bullish": mss_bullish,
            "bearish": mss_bearish
        },
        "liquidity": {
            "equal_highs": liquidity.get("equal_highs", []),
            "equal_lows": liquidity.get("equal_lows", []),
            "prev_day_high": liquidity.get("prev_day_high"),
            "prev_day_low": liquidity.get("prev_day_low"),
            "prev_week_high": liquidity.get("prev_week_high"),
            "prev_week_low": liquidity.get("prev_week_low"),
            "swept_bullish": liquidity_swept_bullish,
            "swept_bearish": liquidity_swept_bearish
        },
        "order_blocks": {
            "bullish": bullish_obs_filtered,
            "bearish": bearish_obs_filtered
        },
        "fvgs": fvgs,
        "htf_poi": htf_data if htf_data else None
    }

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


@app.route('/api/notifications/preferences', methods=['GET'])
def api_notifications_preferences():
    """Get notification preferences (placeholder)."""
    return jsonify({
        "success": True,
        "preferences": {
            "enabled": False,
            "sound": True,
            "desktop": True
        }
    })

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


@app.route('/opportunities')
def opportunities_page():
    """Trading opportunities page"""
    return render_template('opportunities.html')


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
            ict_analysis = analyze_ict_setup_full(timeframe_data)
        else:
            ict_analysis = {
                "signal": None, "entry_price": None, "stop_loss": None, "take_profit": None, "risk_reward": None,
                "fvgs": [], "order_blocks": {"bullish": [], "bearish": []}, "liquidity": {}, "structure": {}
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
                "reasons": ict_analysis.get("reasons", []),
                "order_blocks": ict_analysis.get("order_blocks", {}),
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


def run_server(debug=False, host='0.0.0.0', port=5000):
    """Run the Flask server"""
    app.run(host=host, port=port, debug=debug)


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


"""
Backtesting Routes
==================
"""

@app.route('/backtest')
def backtest_page():
    """Backtesting page"""
    return render_template('backtest.html')


@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    """Run backtest with given parameters"""
    data = request.json
    
    symbol = data.get('symbol', 'EURUSD')
    timeframe = data.get('timeframe', '1H')
    strategy = data.get('strategy', 'ICT')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    # Parse dates
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.now() - timedelta(days=90)
        end = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
        days = (end - start).days
    except:
        days = 90
        start = datetime.now() - timedelta(days=90)
        end = datetime.now()
    
    # Map timeframe
    tf_map = {'1H': '1h', '4H': '4h', '1D': '1d'}
    yahoo_tf = tf_map.get(timeframe, '1h')
    
    # Get Yahoo Finance symbol
    yahoo_symbol = symbol
    if symbol == 'XAUUSD':
        yahoo_symbol = 'GC=F'
    elif symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        yahoo_symbol = symbol.replace('USDT', '-USD')
    else:
        yahoo_symbol = symbol.replace('_', '') + '=X'
    
    try:
        # Fetch historical data from Yahoo
        import urllib.request
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?interval={yahoo_tf}&range={days}d"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            ydata = json.loads(response.read().decode())
        
        result = ydata.get('chart', {}).get('result', [])
        if not result:
            raise Exception("No data returned")
        
        quote = result[0]['indicators']['quote'][0]
        timestamps = result[0].get('timestamp', [])
        
        # Build candles list
        candles = []
        for i, ts in enumerate(timestamps):
            if ts is None:
                continue
            o = quote.get('open', [None])[i]
            h = quote.get('high', [None])[i]
            l = quote.get('low', [None])[i]
            c = quote.get('close', [None])[i]
            if all(v is not None for v in [o, h, l, c]):
                candles.append({
                    'time': ts,
                    'open': o, 'high': h, 'low': l, 'close': c
                })
        
        if len(candles) < 50:
            return jsonify({'success': False, 'error': 'Insufficient data for backtest'}), 400
        
        # Run strategy-based backtest
        backtest_result = run_strategy_backtest(candles, strategy, symbol)
        
        return jsonify({
            'success': True,
            'result': backtest_result,
            'candles': candles,
            'params': {
                'symbol': symbol,
                'timeframe': timeframe,
                'strategy': strategy,
                'start_date': start.strftime('%Y-%m-%d'),
                'end_date': end.strftime('%Y-%m-%d')
            }
        })
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def run_strategy_backtest(candles, strategy, symbol):
    """Run strategy-based backtest on historical data"""
    import math
    
    # Helper functions to replace numpy
    def mean(arr):
        return sum(arr) / len(arr) if arr else 0
    
    def std(arr):
        if len(arr) < 2:
            return 0
        m = mean(arr)
        variance = sum((x - m) ** 2 for x in arr) / len(arr)
        return math.sqrt(variance)
    
    # Initial settings
    initial_balance = 10000
    balance = initial_balance
    trades = []
    position = None
    
    # Extract price arrays
    opens = [c['open'] for c in candles]
    highs = [c['high'] for c in candles]
    lows = [c['low'] for c in candles]
    closes = [c['close'] for c in candles]
    times = [c['time'] for c in candles]
    
    # Strategy parameters
    if strategy == 'ICT':
        lookback = 5
        rr_ratio = 3
    elif strategy == 'MA Cross':
        fast_ma = 9
        slow_ma = 21
    elif strategy == 'RSI':
        rsi_period = 14
        rsi_oversold = 30
        rsi_overbought = 70
    
    for i in range(50, len(candles) - 10):
        if position is None:
            signal = None
            
            if strategy == 'ICT':
                current_price = closes[i]
                swing_low = min(lows[i-lookback:i])
                swing_high = max(highs[i-lookback:i])
                
                if abs(current_price - swing_low) / current_price < 0.003:
                    if closes[i] > opens[i] and closes[i] > opens[i-1]:
                        signal = {'direction': 'long', 'entry': current_price}
                elif abs(current_price - swing_high) / current_price < 0.003:
                    if closes[i] < opens[i] and closes[i] < opens[i-1]:
                        signal = {'direction': 'short', 'entry': current_price}
            
            elif strategy == 'MA Cross':
                fast = mean(closes[i-fast_ma+1:i+1])
                slow = mean(closes[i-slow_ma+1:i+1])
                prev_fast = mean(closes[i-fast_ma:i])
                prev_slow = mean(closes[i-slow_ma:i-1])
                
                if prev_fast <= prev_slow and fast > slow:
                    signal = {'direction': 'long', 'entry': closes[i]}
                elif prev_fast >= prev_slow and fast < slow:
                    signal = {'direction': 'short', 'entry': closes[i]}
            
            elif strategy == 'RSI':
                rsi_vals = closes[i-rsi_period+1:i+1]
                if len(rsi_vals) >= rsi_period:
                    deltas = [rsi_vals[j+1] - rsi_vals[j] for j in range(len(rsi_vals)-1)]
                    gains = [d for d in deltas if d > 0]
                    losses = [abs(d) for d in deltas if d < 0]
                    avg_gain = mean(gains) if gains else 0
                    avg_loss = mean(losses) if losses else 0.0001
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    if rsi < rsi_oversold:
                        signal = {'direction': 'long', 'entry': closes[i]}
                    elif rsi > rsi_overbought:
                        signal = {'direction': 'short', 'entry': closes[i]}
            
            if signal:
                risk_pct = 0.02
                risk_amount = balance * risk_pct
                
                # Calculate ATR-like stop
                atr = std(closes[i-14:i]) if i > 14 else 0.001 * closes[i]
                stop_distance = max(atr, closes[i] * 0.005)
                
                if signal['direction'] == 'long':
                    sl = signal['entry'] - stop_distance
                    tp = signal['entry'] + (stop_distance * rr_ratio if strategy == 'ICT' else stop_distance * 2)
                else:
                    sl = signal['entry'] + stop_distance
                    tp = signal['entry'] - (stop_distance * rr_ratio if strategy == 'ICT' else stop_distance * 2)
                
                position = {
                    'direction': signal['direction'],
                    'entry': signal['entry'],
                    'sl': sl,
                    'tp': tp,
                    'entry_time': times[i],
                    'size': risk_amount / stop_distance,
                    'reason': strategy
                }
        
        else:
            current_price = closes[i]
            exit_reason = None
            
            if position['direction'] == 'long':
                if current_price <= position['sl']:
                    exit_price = position['sl']
                    exit_reason = 'SL'
                elif current_price >= position['tp']:
                    exit_price = position['tp']
                    exit_reason = 'TP'
            else:
                if current_price >= position['sl']:
                    exit_price = position['sl']
                    exit_reason = 'SL'
                elif current_price <= position['tp']:
                    exit_price = position['tp']
                    exit_reason = 'TP'
            
            if exit_reason:
                pnl = (exit_price - position['entry']) * position['size'] if position['direction'] == 'long' else (position['entry'] - exit_price) * position['size']
                balance += pnl
                
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': times[i],
                    'direction': position['direction'],
                    'entry': round(position['entry'], 5),
                    'exit': round(exit_price, 5),
                    'sl': round(position['sl'], 5),
                    'tp': round(position['tp'], 5),
                    'pnl': round(pnl, 2),
                    'pnl_pct': round(pnl / initial_balance * 100, 2),
                    'result': exit_reason
                })
                position = None
    
    # Close open position
    if position:
        final_price = closes[-1]
        pnl = (final_price - position['entry']) * position['size'] if position['direction'] == 'long' else (position['entry'] - final_price) * position['size']
        balance += pnl
        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': times[-1],
            'direction': position['direction'],
            'entry': round(position['entry'], 5),
            'exit': round(final_price, 5),
            'sl': round(position['sl'], 5),
            'tp': round(position['tp'], 5),
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl / initial_balance * 100, 2),
            'result': 'CLOSE'
        })
    
    # Calculate metrics
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = mean([abs(t['pnl']) for t in losses]) if losses else 0
    
    # Max drawdown
    running_balance = initial_balance
    peak = initial_balance
    max_dd = 0
    for t in trades:
        running_balance += t['pnl']
        if running_balance > peak:
            peak = running_balance
        dd = (peak - running_balance) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    
    avg_rr = avg_win / avg_loss if avg_loss != 0 else 0
    
    return {
        'symbol': symbol,
        'initial_balance': initial_balance,
        'final_balance': round(balance, 2),
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': round(win_rate, 2),
        'total_pnl': round(balance - initial_balance, 2),
        'total_pnl_pct': round((balance - initial_balance) / initial_balance * 100, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'avg_rr': round(avg_rr, 2),
        'trades': trades[-50:]
    }


@app.route('/api/backtest/history')
def api_backtest_history():
    """Get backtest history"""
    results = database.get_backtest_results(limit=20)
    history = []
    for r in results:
        history.append({
            'id': r['id'],
            'symbol': r['symbol'],
            'timeframe': r['timeframe'],
            'strategy': r['strategy'],
            'start_date': r['start_date'],
            'end_date': r['end_date'],
            'total_trades': r['total_trades'],
            'win_rate': r['win_rate'],
            'total_pnl': r['total_pnl'],
            'total_pnl_pct': r['total_pnl_pct'],
            'created_at': r['created_at']
        })
    return jsonify({'success': True, 'history': history})


@app.route('/api/backtest/save', methods=['POST'])
def api_backtest_save():
    """Save backtest result"""
    data = request.json
    result = data.get('result', {})
    params = data.get('params', {})
    
    try:
        result_id = database.save_backtest_result(
            user_id=None,
            symbol=params.get('symbol'),
            timeframe=params.get('timeframe'),
            strategy=params.get('strategy'),
            start_date=params.get('start_date'),
            end_date=params.get('end_date'),
            initial_balance=result.get('initial_balance'),
            final_balance=result.get('final_balance'),
            total_trades=result.get('total_trades'),
            wins=result.get('wins'),
            losses=result.get('losses'),
            win_rate=result.get('win_rate'),
            total_pnl=result.get('total_pnl'),
            total_pnl_pct=result.get('total_pnl_pct'),
            avg_win=result.get('avg_win'),
            avg_loss=result.get('avg_loss'),
            max_drawdown_pct=result.get('max_drawdown_pct'),
            sharpe_ratio=result.get('avg_rr'),
            trades_json=result.get('trades', [])
        )
        return jsonify({'success': True, 'id': result_id})
    except Exception as e:
        logger.error(f"Save backtest error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== RISK MANAGEMENT ROUTES ====================

# Default account settings (can be moved to config)
DEFAULT_ACCOUNT_BALANCE = 10000.0
DEFAULT_RISK_PERCENT = 2.0
DEFAULT_MAX_DAILY_RISK = 6.0


@app.route('/risk')
def risk_page():
    """Risk Management Dashboard Page"""
    return render_template('risk.html')


@app.route('/api/risk/positions')
def api_risk_positions():
    """Get current open positions with risk metrics"""
    try:
        positions = []
        
        # Get open trades from trade_history
        for trade in trade_history["open_trades"]:
            # Calculate current P&L based on live price
            current_price = trade.get("current_price", trade.get("entry_price", 0))
            entry_price = trade.get("entry_price", 0)
            
            if trade.get("direction") == "long" or trade.get("direction") == "BUY":
                pnl = (current_price - entry_price) * 10000 if entry_price > 0 else 0  # pips
                pnl_percent = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
            else:
                pnl = (entry_price - current_price) * 10000 if entry_price > 0 else 0
                pnl_percent = ((entry_price - current_price) / entry_price * 100) if entry_price > 0 else 0
            
            # Calculate risk
            stop_loss = trade.get("stop_loss", 0)
            entry = trade.get("entry_price", 0)
            if entry > 0 and stop_loss > 0:
                risk_pips = abs(entry - stop_loss) * 10000
            else:
                risk_pips = 0
            
            positions.append({
                "id": trade.get("id"),
                "symbol": trade.get("symbol"),
                "direction": trade.get("direction"),
                "entry_price": entry_price,
                "current_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": trade.get("take_profit"),
                "pnl_pips": round(pnl, 2),
                "pnl_percent": round(pnl_percent, 2),
                "risk_pips": round(risk_pips, 2),
                "status": trade.get("status", "open"),
                "entry_time": trade.get("entry_time"),
                "timeframe": trade.get("timeframe")
            })
        
        return jsonify({
            "success": True,
            "positions": positions,
            "count": len(positions)
        })
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/risk/summary')
def api_risk_summary():
    """Get risk summary including total exposure, P&L, and account metrics"""
    try:
        # Calculate from trade history
        open_trades = trade_history["open_trades"]
        closed_trades = trade_history["trades"]
        
        # Calculate total unrealized P&L
        total_unrealized_pnl = 0.0
        total_exposure = 0.0
        
        for trade in open_trades:
            current_price = trade.get("current_price", trade.get("entry_price", 0))
            entry_price = trade.get("entry_price", 0)
            
            if trade.get("direction") == "long" or trade.get("direction") == "BUY":
                pnl = (current_price - entry_price) * 10000 if entry_price > 0 else 0
            else:
                pnl = (entry_price - current_price) * 10000 if entry_price > 0 else 0
            
            total_unrealized_pnl += pnl
            
            # Calculate exposure (notional value)
            # Assuming 1 standard lot = 100,000 units
            lot_size = trade.get("lot_size", 1.0)
            exposure = lot_size * 100000 * entry_price
            total_exposure += exposure
        
        # Calculate realized P&L from closed trades
        total_realized_pnl = sum(t.get("pnl_pips", 0) for t in closed_trades)
        
        # Calculate win rate
        winning_trades = len([t for t in closed_trades if t.get("result") == "TP"])
        total_closed = len(closed_trades)
        win_rate = (winning_trades / total_closed * 100) if total_closed > 0 else 0
        
        # Get account balance (using default or from config)
        account_balance = DEFAULT_ACCOUNT_BALANCE
        
        # Calculate margin (simplified - assume 100:1 leverage)
        used_margin = total_exposure / 100 if total_exposure > 0 else 0
        available_margin = account_balance - used_margin
        
        # Calculate drawdown (from peak balance)
        peak_balance = account_balance + total_realized_pnl + total_unrealized_pnl
        current_balance = account_balance + total_realized_pnl + total_unrealized_pnl
        drawdown = ((peak_balance - current_balance) / peak_balance * 100) if peak_balance > 0 else 0
        
        # Calculate risk metrics
        risk_per_trade = DEFAULT_RISK_PERCENT
        max_daily_risk = DEFAULT_MAX_DAILY_RISK
        
        # Calculate today's risk
        today = datetime.now().strftime("%Y-%m-%d")
        today_trades = [t for t in closed_trades if t.get("exit_time", "").startswith(today)]
        daily_risk_used = sum(abs(t.get("pnl_pips", 0)) for t in today_trades if t.get("result") == "SL") / (account_balance / 10000) if account_balance > 0 else 0
        
        return jsonify({
            "success": True,
            "summary": {
                "account_balance": account_balance,
                "total_open_positions": len(open_trades),
                "total_unrealized_pnl": round(total_unrealized_pnl, 2),
                "total_realized_pnl": round(total_realized_pnl, 2),
                "total_exposure": round(total_exposure, 2),
                "available_margin": round(available_margin, 2),
                "used_margin": round(used_margin, 2),
                "win_rate": round(win_rate, 2),
                "total_trades": total_closed,
                "winning_trades": winning_trades,
                "losing_trades": total_closed - winning_trades,
                "current_drawdown": round(drawdown, 2),
                "risk_per_trade": risk_per_trade,
                "max_daily_risk": max_daily_risk,
                "daily_risk_used": round(daily_risk_used, 2),
                "max_daily_risk_remaining": round(max_daily_risk - daily_risk_used, 2)
            }
        })
    except Exception as e:
        logger.error(f"Error fetching risk summary: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/risk/calculator', methods=['POST'])
def api_risk_calculator():
    """Position size calculator"""
    try:
        data = request.json
        
        account_balance = float(data.get("account_balance", DEFAULT_ACCOUNT_BALANCE))
        risk_percent = float(data.get("risk_percent", DEFAULT_RISK_PERCENT))
        stop_loss_pips = float(data.get("stop_loss_pips", 20))
        entry_price = float(data.get("entry_price", 0))
        stop_loss = float(data.get("stop_loss", 0))
        
        # Calculate risk amount in account currency
        risk_amount = account_balance * (risk_percent / 100)
        
        # Calculate pip value (simplified - assume USD base)
        # For direct pairs (EURUSD, GBPUSD, etc.): 1 pip = $10 per standard lot
        # For USDJPY: 1 pip = ¥1000 per standard lot, need to convert
        # This is simplified - real calculation depends on account currency
        
        pip_value = 10  # Default for direct pairs
        
        # Calculate lot size
        # Lot size = Risk amount / (Stop loss pips * pip value)
        lot_size = risk_amount / (stop_loss_pips * pip_value) if stop_loss_pips > 0 and pip_value > 0 else 0
        
        # Convert to more readable units
        # Standard lot = 100,000 units
        # Mini lot = 10,000 units
        # Micro lot = 1,000 units
        standard_lots = lot_size
        mini_lots = lot_size * 10
        micro_lots = lot_size * 100
        
        # Calculate position value
        position_value = lot_size * 100000 * entry_price if entry_price > 0 else 0
        
        # Calculate margin required (assuming 100:1 leverage)
        margin_required = position_value / 100 if position_value > 0 else 0
        
        # Calculate potential profit/loss for 1:2 R:R
        potential_r2 = risk_amount * 2
        
        return jsonify({
            "success": True,
            "calculation": {
                "account_balance": account_balance,
                "risk_percent": risk_percent,
                "risk_amount": round(risk_amount, 2),
                "stop_loss_pips": stop_loss_pips,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "lot_size": round(lot_size, 2),
                "standard_lots": round(standard_lots, 2),
                "mini_lots": round(mini_lots, 2),
                "micro_lots": round(micro_lots, 2),
                "position_value": round(position_value, 2),
                "margin_required": round(margin_required, 2),
                "potential_profit_1_2_rr": round(potential_r2, 2),
                "risk_reward_ratio": "1:2"
            }
        })
    except Exception as e:
        logger.error(f"Error in position calculator: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

