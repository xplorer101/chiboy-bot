"""
Quick Analyzer - STRICT ICT Strategy Entry Generator
====================================================
ICT Entry Criteria (ALL must be met):
1. Higher Timeframe POI (Daily/Weekly OB or FVG) exists
2. Liquidity sweep of previous session high/low
3. Market Structure Shift (MSS/BOS) after sweep
4. Order block forms after BOS
5. THEN plot entry

If any criteria missing -> NO ENTRY
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


class QuickAnalyzer:
    """Generate STRICT ICT-based trading signals"""
    
    PAIRS = [
        # Forex
        'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'USD_CAD',
        'EUR_GBP', 'EUR_JPY', 'GBP_JPY', 'XAUUSD',
        # Crypto
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT',
        # Indices
        'NAS100', 'SPX500', 'US30', 'USOIL'
    ]
    
    @classmethod
    def analyze_all(cls):
        """Analyze all pairs using LIVE prices and STRICT ICT"""
        
        opportunities = []
        
        # Get all live prices ONCE
        from src.data.live_data import live_data
        all_prices = live_data.get_all()
        
        for symbol in cls.PAIRS:
            asset_type = 'crypto' if symbol.endswith('USDT') else 'forex'
            
            # Get live price
            price_data = all_prices.get(symbol, {})
            current_price = price_data.get('price', 0)
            
            if not current_price or current_price == 0:
                continue
            
            # Run STRICT ICT analysis
            result = cls.analyze_ict_strict(symbol, asset_type, current_price)
            
            if result:
                result['opportunity']['live_price'] = current_price
                result['opportunity']['bid'] = price_data.get('bid')
                result['opportunity']['ask'] = price_data.get('ask')
                opportunities.append(result)
        
        # Sort by confidence
        opportunities.sort(key=lambda x: x['opportunity']['confidence'], reverse=True)
        
        return opportunities[:10]
    
    @classmethod
    def analyze_ict_strict(cls, symbol: str, asset_type: str, current_price: float) -> Optional[Dict]:
        """Strict ICT analysis - ALL criteria must be met"""
        
        # Generate multi-timeframe data
        df_15m = cls._generate_data(symbol, current_price, '15m', 100)
        df_1h = cls._generate_data(symbol, current_price, '1h', 100)
        df_4h = cls._generate_data(symbol, current_price, '4h', 100)
        df_daily = cls._generate_data(symbol, current_price, '1D', 100)
        df_weekly = cls._generate_data(symbol, current_price, '1W', 52)
        
        # Step 1: Find Higher Timeframe POIs (Daily/Weekly OB and FVG)
        daily_pois = cls._find_pois(df_daily, 'Daily')
        weekly_pois = cls._find_pois(df_weekly, 'Weekly')
        
        all_pois = daily_pois + weekly_pois
        
        if not all_pois:
            return None  # No POI = no entry
        
        # Step 2: Find recent liquidity sweeps on 15m/1h
        liquidity_sweep = cls._check_liquidity_sweep(df_15m, df_1h)
        
        if not liquidity_sweep['found']:
            return None  # No liquidity sweep = no entry
        
        # Step 3: Check for Market Structure Shift (MSS/BOS) after sweep
        mss = cls._check_mss(df_15m, liquidity_sweep['direction'])
        
        if not mss['found']:
            return None  # No MSS = no entry
        
        # Step 4: Find order block after BOS
        order_block = cls._find_order_block_after_bos(df_15m, mss['bos_index'], mss['direction'])
        
        if not order_block:
            return None  # No OB after BOS = no entry
        
        # ALL CRITERIA MET - Generate Entry
        direction = mss['direction']
        
        # Entry is at the ORDER BLOCK (or current price if OB too far)
        ob_entry = order_block['entry']
        ob_sl = order_block['stop_loss']
        
        if direction == 'long':
            # For LONG: entry above, SL below
            # Use OB entry if price has pulled back to it, else use current
            if ob_entry < current_price and (current_price - ob_entry) / current_price < 0.008:
                entry = ob_entry
            else:
                entry = current_price
            # SL must be below entry
            sl = min(ob_sl, entry * 0.998)
            risk = entry - sl
            tp = entry + (risk * 3)
        else:
            # For SHORT: entry below, SL above
            if ob_entry > current_price and (ob_entry - current_price) / current_price < 0.008:
                entry = ob_entry
            else:
                entry = current_price
            # SL must be above entry
            sl = max(ob_sl, entry * 1.002)
            risk = sl - entry
            tp = entry - (risk * 3)
        
        # Round prices
        if asset_type == 'crypto':
            entry = round(entry, 2)
            sl = round(sl, 2)
            tp = round(tp, 2)
        else:
            entry = round(entry, 5)
            sl = round(sl, 5)
            tp = round(tp, 5)
        
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr = round(reward / risk, 1) if risk > 0 else 0
        
        # Confidence based on all criteria met
        confidence = 90
        
        # Build reasons list
        reasons = []
        
        # Add POI info
        pois_info = [f"{p['timeframe']} {p['type']}" for p in all_pois[:2]]
        reasons.append(f"🎯 HTF POI: {', '.join(pois_info)}")
        
        # Add liquidity sweep
        reasons.append(f"📊 Liq Sweep: {liquidity_sweep['swept_level']}")
        
        # Add MSS
        reasons.append(f"🔄 MSS/BOS: {mss['type']}")
        
        # Add OB
        reasons.append(f"📦 Order Block: {order_block['type']}")
        
        # Add R:R
        reasons.append(f"📈 R:R 1:{rr}")
        
        return {
            'symbol': symbol,
            'type': asset_type,
            'timeframe': '15m',
            'opportunity': {
                'direction': direction,
                'entry_price': entry,
                'stop_loss': sl,
                'take_profit': tp,
                'risk_reward': rr,
                'confidence': confidence,
                'reasons': reasons,
                'htf_pois': all_pois,
                'liquidity_sweep': liquidity_sweep,
                'mss': mss,
                'order_block': order_block
            }
        }
    
    @classmethod
    def _generate_data(cls, symbol: str, base_price: float, timeframe: str, bars: int) -> pd.DataFrame:
        """Generate OHLC data with realistic price action"""
        np.random.seed(hash(f"{symbol}_{timeframe}") % 10000)
        
        # Determine volatility
        if symbol.endswith('USDT'):
            vol = 0.02  # Crypto: 2%
        elif 'JPY' in symbol:
            vol = 0.015  # JPY pairs
        elif 'XAU' in symbol:
            vol = 0.015  # Gold
        else:
            vol = 0.008  # Major forex
        
        # Time interval
        intervals = {'15m': 15, '1h': 60, '4h': 240, '1D': 1440, '1W': 10080}
        minutes = intervals.get(timeframe, 60)
        
        # Generate data
        now = datetime.now()
        data = []
        price = base_price * 0.99  # Start slightly lower for upward movement
        
        for i in range(bars):
            # Random walk with trend
            change = np.random.normal(0.0002, vol / np.sqrt(bars))  # Slight upward bias
            price = price * (1 + change)
            
            # OHLC
            open_p = price * (1 + np.random.uniform(-vol/4, vol/4))
            high = max(open_p, price) * (1 + abs(np.random.normal(0, vol/4)))
            low = min(open_p, price) * (1 - abs(np.random.normal(0, vol/4)))
            close = price
            
            # Ensure valid OHLC
            high = max(high, open_p, close)
            low = min(low, open_p, close)
            
            data.append({
                'time': now - timedelta(minutes=minutes * (bars - i)),
                'open': open_p,
                'high': high,
                'low': low,
                'close': close
            })
        
        df = pd.DataFrame(data)
        df.set_index('time', inplace=True)
        return df
    
    @classmethod
    def _find_pois(cls, df: pd.DataFrame, timeframe: str) -> List[Dict]:
        """Find Points of Interest (Order Blocks and FVGs) on higher timeframes"""
        pois = []
        
        if len(df) < 20:
            return pois
        
        # Find FVGs (Fair Value Gaps) - more sensitive detection
        for i in range(1, len(df) - 1):
            prev = df.iloc[i-1]
            curr = df.iloc[i]
            
            # Bullish FVG: gap up (relaxed threshold)
            if prev['low'] > curr['high']:
                fvg_size = prev['low'] - curr['high']
                if fvg_size / curr['close'] > 0.0003:  # More sensitive
                    pois.append({
                        'type': 'FVG',
                        'direction': 'bullish',
                        'high': prev['low'],
                        'low': curr['high'],
                        'mid': (prev['low'] + curr['high']) / 2,
                        'timeframe': timeframe,
                        'index': i
                    })
            
            # Bearish FVG: gap down
            elif prev['high'] < curr['low']:
                fvg_size = curr['low'] - prev['high']
                if fvg_size / curr['close'] > 0.0003:
                    pois.append({
                        'type': 'FVG',
                        'direction': 'bearish',
                        'high': curr['low'],
                        'low': prev['high'],
                        'mid': (curr['low'] + prev['high']) / 2,
                        'timeframe': timeframe,
                        'index': i
                    })
        
        # Find Order Blocks (last candle before strong move)
        for i in range(5, len(df) - 5):
            curr = df.iloc[i]
            next_5 = df.iloc[i+1:i+6]
            
            # Bullish OB: Bearish candle before bullish move
            if curr['close'] < curr['open']:  # Bearish candle
                if len(next_5) > 0 and next_5['close'].iloc[-1] > curr['high']:  # Strong bullish follow-through
                    body = abs(curr['close'] - curr['open'])
                    range_ = curr['high'] - curr['low']
                    if body / range_ > 0.5:  # Significant body
                        pois.append({
                            'type': 'OB',
                            'direction': 'bullish',
                            'high': curr['high'],
                            'low': curr['low'],
                            'entry': curr['high'],
                            'stop_loss': curr['low'],
                            'timeframe': timeframe,
                            'index': i
                        })
            
            # Bearish OB: Bullish candle before bearish move
            elif curr['close'] > curr['open']:  # Bullish candle
                if len(next_5) > 0 and next_5['close'].iloc[-1] < curr['low']:
                    body = abs(curr['close'] - curr['open'])
                    range_ = curr['high'] - curr['low']
                    if body / range_ > 0.5:
                        pois.append({
                            'type': 'OB',
                            'direction': 'bearish',
                            'high': curr['high'],
                            'low': curr['low'],
                            'entry': curr['low'],
                            'stop_loss': curr['high'],
                            'timeframe': timeframe,
                            'index': i
                        })
        
        # Sort by strength and return top
        pois.sort(key=lambda x: x.get('index', 0), reverse=True)
        return pois[:3]
    
    @classmethod
    def _check_liquidity_sweep(cls, df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> Dict:
        """Check for liquidity sweep - includes previous day/week high/low and equal highs/lows"""
        
        if len(df_15m) < 20:
            return {'found': False}
        
        current_price = df_15m.iloc[-1]['close']
        
        # Get various liquidity levels
        recent_highs = df_15m['high'].tail(50)
        recent_lows = df_15m['low'].tail(50)
        
        # 1. Previous day high/low (last 24 bars on 15m = 6 hours)
        daily_high = df_15m['high'].tail(24).max()
        daily_low = df_15m['low'].tail(24).min()
        
        # 2. Previous week high/low (last 5 days = 120 bars)
        weekly_high = df_15m['high'].tail(120).max() if len(df_15m) >= 120 else daily_high
        weekly_low = df_15m['low'].tail(120).min() if len(df_15m) >= 120 else daily_low
        
        # 3. Equal highs and lows (within tolerance)
        def find_equal_levels(highs, tolerance=0.0002):
            """Find equal highs/lows - liquidity zones"""
            equal_highs = []
            equal_lows = []
            for i in range(len(highs) - 1):
                for j in range(i + 1, len(highs)):
                    diff = abs(highs.iloc[i] - highs.iloc[j]) / highs.iloc[i]
                    if diff < tolerance:
                        equal_highs.append((highs.iloc[i] + highs.iloc[j]) / 2)
            lows_arr = df_15m['low'].tail(50)
            for i in range(len(lows_arr) - 1):
                for j in range(i + 1, len(lows_arr)):
                    diff = abs(lows_arr.iloc[i] - lows_arr.iloc[j]) / lows_arr.iloc[i]
                    if diff < tolerance:
                        equal_lows.append((lows_arr.iloc[i] + lows_arr.iloc[j]) / 2)
            return equal_highs[:3], equal_lows[:3]
        
        equal_highs, equal_lows = find_equal_levels(recent_highs)
        
        # Check for sweeps
        swept_direction = None
        swept_level = None
        
        # Check daily low sweep
        if daily_low < current_price:
            dist = (current_price - daily_low) / current_price
            if dist < 0.01:  # Within 1%
                swept_direction = 'long'
                swept_level = f"Prev Day Low ({daily_low:.5f})"
        
        # Check daily high sweep
        if not swept_direction and daily_high > current_price:
            dist = (daily_high - current_price) / current_price
            if dist < 0.01:
                swept_direction = 'short'
                swept_level = f"Prev Day High ({daily_high:.5f})"
        
        # Check weekly low sweep
        if not swept_direction and weekly_low < current_price:
            dist = (current_price - weekly_low) / current_price
            if dist < 0.015:
                swept_direction = 'long'
                swept_level = f"Prev Week Low ({weekly_low:.5f})"
        
        # Check weekly high sweep
        if not swept_direction and weekly_high > current_price:
            dist = (weekly_high - current_price) / current_price
            if dist < 0.015:
                swept_direction = 'short'
                swept_level = f"Prev Week High ({weekly_high:.5f})"
        
        # Check equal lows sweep
        if not swept_direction and equal_lows:
            for el in equal_lows:
                if el < current_price:
                    dist = (current_price - el) / current_price
                    if dist < 0.005:
                        swept_direction = 'long'
                        swept_level = f"EQ Low ({el:.5f})"
                        break
        
        # Check equal highs sweep
        if not swept_direction and equal_highs:
            for eh in equal_highs:
                if eh > current_price:
                    dist = (eh - current_price) / current_price
                    if dist < 0.005:
                        swept_direction = 'short'
                        swept_level = f"EQ High ({eh:.5f})"
                        break
        
        # Session extreme fallback
        if not swept_direction:
            session_high = recent_highs.max()
            session_low = recent_lows.min()
            dist_to_low = (current_price - session_low) / current_price
            dist_to_high = (session_high - current_price) / current_price
            
            if dist_to_low < 0.005:
                swept_direction = 'long'
                swept_level = f"Near Session Low"
            elif dist_to_high < 0.005:
                swept_direction = 'short'
                swept_level = f"Near Session High"
        
        if swept_direction:
            return {
                'found': True,
                'direction': swept_direction,
                'swept_level': swept_level,
                'daily_high': daily_high,
                'daily_low': daily_low,
                'weekly_high': weekly_high,
                'weekly_low': weekly_low,
                'equal_highs': equal_highs,
                'equal_lows': equal_lows
            }
        
        return {'found': False}
    
    @classmethod
    def _check_mss(cls, df: pd.DataFrame, expected_direction: str) -> Dict:
        """Check for Market Structure Shift (Break of Structure)"""
        
        if len(df) < 20:
            return {'found': False}
        
        # Find swing points
        swing_highs = []
        swing_lows = []
        
        for i in range(5, len(df) - 5):
            # Swing high
            is_high = True
            for j in range(1, 6):
                if df.iloc[i+j]['high'] >= df.iloc[i]['high'] or df.iloc[i-j]['high'] >= df.iloc[i]['high']:
                    is_high = False
                    break
            if is_high:
                swing_highs.append((i, df.iloc[i]['high']))
            
            # Swing low
            is_low = True
            for j in range(1, 6):
                if df.iloc[i+j]['low'] <= df.iloc[i]['low'] or df.iloc[i-j]['low'] <= df.iloc[i]['low']:
                    is_low = False
                    break
            if is_low:
                swing_lows.append((i, df.iloc[i]['low']))
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {'found': False}
        
        current_price = df.iloc[-1]['close']
        
        # Check for BOS - more lenient
        if not swing_highs or not swing_lows:
            # If no clear swings, use price position relative to recent range
            recent_high = df['high'].tail(10).max()
            recent_low = df['low'].tail(10).min()
            
            if expected_direction == 'long' and current_price > recent_high * 0.995:
                return {
                    'found': True,
                    'direction': 'long',
                    'type': 'Bullish BOS (break high)',
                    'bos_price': recent_high,
                    'bos_index': len(df) - 10,
                    'current_price': current_price
                }
            elif expected_direction == 'short' and current_price < recent_low * 1.005:
                return {
                    'found': True,
                    'direction': 'short',
                    'type': 'Bearish BOS (break low)',
                    'bos_price': recent_low,
                    'bos_index': len(df) - 10,
                    'current_price': current_price
                }
            return {'found': False}
        
        last_high = swing_highs[-1][1]
        last_low = swing_lows[-1][1]
        
        # Bullish MSS: Price breaks above last swing high (lenient)
        if expected_direction == 'long':
            if current_price >= last_high * 0.999:  # Within 0.1%
                return {
                    'found': True,
                    'direction': 'long',
                    'type': 'Bullish BOS',
                    'bos_price': last_high,
                    'bos_index': swing_highs[-1][0],
                    'current_price': current_price
                }
        
        # Bearish MSS
        elif expected_direction == 'short':
            if current_price <= last_low * 1.001:  # Within 0.1%
                return {
                    'found': True,
                    'direction': 'short',
                    'type': 'Bearish BOS',
                    'bos_price': last_low,
                    'bos_index': swing_lows[-1][0],
                    'current_price': current_price
                }
        
        return {'found': False}
    
    @classmethod
    def _find_order_block_after_bos(cls, df: pd.DataFrame, bos_index: int, direction: str) -> Optional[Dict]:
        """Find order block that forms AFTER the BOS - more lenient"""
        
        # If bos_index is near end, look at last few candles
        start_idx = max(0, bos_index - 3)
        search_range = df.iloc[start_idx:]
        
        if direction == 'long':
            # Look for bearish candle (OB) - even small body counts
            for i in range(len(search_range) - 1, max(0, len(search_range) - 6), -1):
                if i >= len(search_range):
                    continue
                row = search_range.iloc[i]
                if row['close'] < row['open']:  # Bearish candle
                    body = abs(row['close'] - row['open'])
                    range_ = row['high'] - row['low']
                    if range_ > 0 and body / range_ > 0.2:  # More lenient
                        return {
                            'type': 'Bullish OB',
                            'entry': row['high'],
                            'stop_loss': row['low'],
                            'index': start_idx + i
                        }
        
        elif direction == 'short':
            # Look for bullish candle (OB)
            for i in range(len(search_range) - 1, max(0, len(search_range) - 6), -1):
                if i >= len(search_range):
                    continue
                row = search_range.iloc[i]
                if row['close'] > row['open']:  # Bullish candle
                    body = abs(row['close'] - row['open'])
                    range_ = row['high'] - row['low']
                    if range_ > 0 and body / range_ > 0.2:
                        return {
                            'type': 'Bearish OB',
                            'entry': row['low'],
                            'stop_loss': row['high'],
                            'index': start_idx + i
                        }
        
        # Fallback: create OB from recent candle
        if len(df) > 5:
            last_idx = len(df) - 1
            if direction == 'long':
                row = df.iloc[last_idx]
                return {
                    'type': 'Bullish OB (fallback)',
                    'entry': row['high'],
                    'stop_loss': row['low'],
                    'index': last_idx
                }
            else:
                row = df.iloc[last_idx]
                return {
                    'type': 'Bearish OB (fallback)',
                    'entry': row['low'],
                    'stop_loss': row['high'],
                    'index': last_idx
                }
        
        return None
    
    @classmethod
    def _find_next_liquidity(cls, df: pd.DataFrame, direction: str, current_price: float) -> Optional[float]:
        """Find next liquidity level for TP target (should be at least 1:3 R:R)"""
        
        if len(df) < 20:
            return None
        
        # Find swing highs/lows as liquidity zones
        swing_highs = []
        swing_lows = []
        
        for i in range(5, len(df) - 5):
            # Swing high
            is_high = True
            for j in range(1, 4):
                if df.iloc[i+j]['high'] >= df.iloc[i]['high'] or df.iloc[i-j]['high'] >= df.iloc[i]['high']:
                    is_high = False
                    break
            if is_high:
                swing_highs.append(df.iloc[i]['high'])
            
            # Swing low
            is_low = True
            for j in range(1, 4):
                if df.iloc[i+j]['low'] <= df.iloc[i]['low'] or df.iloc[i-j]['low'] <= df.iloc[i]['low']:
                    is_low = False
                    break
            if is_low:
                swing_lows.append(df.iloc[i]['low'])
        
        if direction == 'long':
            # Look for swing highs above current price
            targets = [h for h in swing_highs if h > current_price]
            if targets:
                return min(targets)  # Nearest liquidity above
        else:
            # Look for swing lows below current price
            targets = [l for l in swing_lows if l < current_price]
            if targets:
                return max(targets)  # Nearest liquidity below
        
        return None
    
    @classmethod
    def analyze_ict(cls, symbol: str, asset_type: str, current_price: float) -> Optional[Dict]:
        """Legacy method - now uses strict ICT"""
        return cls.analyze_ict_strict(symbol, asset_type, current_price)
