"""
CHIBOY Market Analyzer
==================
Implements CHIBOY (Inner Circle Trader) concepts for market analysis.
Analyzes market structure, liquidity, order blocks, and fair value gaps.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

from ..config import CHIBOY_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class SwingPoint:
    """Represents a swing high or low."""
    index: int
    time: pd.Timestamp
    price: float
    type: str  # 'high' or 'low'
    strength: int


@dataclass
class LiquidityZone:
    """Represents a liquidity zone."""
    price: float
    type: str  # 'buy_side' or 'sell_side'
    touches: int
    strength: float


@dataclass
class OrderBlock:
    """Represents an order block."""
    start_index: int
    end_index: int
    type: str  # 'bullish' or 'bearish'
    entry_price: float
    stop_loss: float
    timeframe: str


@dataclass
class FairValueGap:
    """Represents a Fair Value Gap."""
    start_index: int
    end_index: int
    high: float
    low: float
    type: str  # 'bullish' or 'bearish'
    filled: bool = False


@dataclass
class MarketStructure:
    """Represents market structure."""
    trend: str  # 'uptrend', 'downtrend', 'ranging'
    swing_highs: List[SwingPoint]
    swing_lows: List[SwingPoint]
    current_structure: str  # 'bullish', 'bearish', 'neutral'


class CHIBOYAnalyzer:
    """CHIBOY trading strategy analyzer."""
    
    def __init__(self):
        self.config = CHIBOY_CONFIG
    
    # ==================== SWING POINT DETECTION ====================
    
    def find_swing_points(self, df: pd.DataFrame, lookback: int = None) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """
        Find swing highs and swing lows in the price data.
        
        Args:
            df: OHLCV DataFrame
            lookback: Number of bars to look back
            
        Returns:
            Tuple of (swing_highs, swing_lows)
        """
        lookback = lookback or self.config["swing_lookback"]
        
        swing_highs = []
        swing_lows = []
        
        for i in range(lookback, len(df) - lookback):
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            
            # Check for swing high
            is_high = True
            for j in range(1, lookback + 1):
                if df.iloc[i+j]['high'] >= current_high or df.iloc[i-j]['high'] >= current_high:
                    is_high = False
                    break
            
            if is_high:
                strength = self._calculate_swing_strength(df, i, 'high')
                swing_highs.append(SwingPoint(
                    index=i,
                    time=df.index[i],
                    price=current_high,
                    type='high',
                    strength=strength
                ))
            
            # Check for swing low
            is_low = True
            for j in range(1, lookback + 1):
                if df.iloc[i+j]['low'] <= current_low or df.iloc[i-j]['low'] <= current_low:
                    is_low = False
                    break
            
            if is_low:
                strength = self._calculate_swing_strength(df, i, 'low')
                swing_lows.append(SwingPoint(
                    index=i,
                    time=df.index[i],
                    price=current_low,
                    type='low',
                    strength=strength
                ))
        
        return swing_highs, swing_lows
    
    def _calculate_swing_strength(self, df: pd.DataFrame, index: int, swing_type: str) -> int:
        """Calculate how many times a swing point was tested."""
        price = df.iloc[index][swing_type]
        tolerance = self.config["liquidity_tolerance"]
        
        strength = 1
        # Look for nearby touches
        for i in range(max(0, index - 20), min(len(df), index + 20)):
            if i != index:
                check_price = df.iloc[i][swing_type]
                if abs(check_price - price) / price < tolerance:
                    strength += 1
        
        return strength
    
    # ==================== MARKET STRUCTURE ====================
    
    def analyze_structure(self, df: pd.DataFrame) -> MarketStructure:
        """Analyze overall market structure."""
        swing_highs, swing_lows = self.find_swing_points(df)
        
        # Determine trend
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            last_high = swing_highs[-1].price
            prev_high = swing_highs[-2].price
            last_low = swing_lows[-1].price
            prev_low = swing_lows[-2].price
            
            if last_high > prev_high and last_low > prev_low:
                trend = "uptrend"
                current_structure = "bullish"
            elif last_high < prev_high and last_low < prev_low:
                trend = "downtrend"
                current_structure = "bearish"
            else:
                trend = "ranging"
                current_structure = "neutral"
        else:
            trend = "unknown"
            current_structure = "neutral"
        
        return MarketStructure(
            trend=trend,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            current_structure=current_structure
        )
    
    # ==================== LIQUIDITY ZONES ====================
    
    def find_liquidity_zones(self, df: pd.DataFrame) -> List[LiquidityZone]:
        """
        Find liquidity zones (areas with clustered stop orders).
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            List of LiquidityZone objects
        """
        zones = []
        tolerance = self.config["liquidity_tolerance"]
        lookback = self.config["liquidity_lookback"]
        
        recent_highs = df['high'].tail(lookback)
        recent_lows = df['low'].tail(lookback)
        
        # Find equal highs (buy-side liquidity)
        for i in range(len(recent_highs) - 1):
            for j in range(i + 1, len(recent_highs)):
                h1, h2 = recent_highs.iloc[i], recent_highs.iloc[j]
                if h1 > 0 and abs(h1 - h2) / h1 < tolerance:
                    # Found equal highs - potential liquidity zone
                    price = (h1 + h2) / 2
                    touches = 2
                    
                    # Check for additional touches
                    for k in range(len(recent_highs)):
                        if k != i and k != j:
                            if recent_highs.iloc[k] > 0 and abs(recent_highs.iloc[k] - price) / price < tolerance:
                                touches += 1
                    
                    zones.append(LiquidityZone(
                        price=price,
                        type='buy_side',
                        touches=touches,
                        strength=touches / 5.0  # Normalize strength
                    ))
        
        # Find equal lows (sell-side liquidity)
        for i in range(len(recent_lows) - 1):
            for j in range(i + 1, len(recent_lows)):
                l1, l2 = recent_lows.iloc[i], recent_lows.iloc[j]
                if l1 > 0 and abs(l1 - l2) / l1 < tolerance:
                    price = (l1 + l2) / 2
                    touches = 2
                    
                    for k in range(len(recent_lows)):
                        if k != i and k != j:
                            if recent_lows.iloc[k] > 0 and abs(recent_lows.iloc[k] - price) / price < tolerance:
                                touches += 1
                    
                    zones.append(LiquidityZone(
                        price=price,
                        type='sell_side',
                        touches=touches,
                        strength=touches / 5.0
                    ))
        
        # Also check daily/weekly highs and lows
        if len(df) >= 24:
            daily_high = df['high'].tail(24).max()
            daily_low = df['low'].tail(24).min()
            
            # Add as liquidity zones
            if daily_high > 0:
                zones.append(LiquidityZone(price=daily_high, type='buy_side', touches=1, strength=0.5))
            if daily_low > 0:
                zones.append(LiquidityZone(price=daily_low, type='sell_side', touches=1, strength=0.5))
        
        # Sort by strength
        zones.sort(key=lambda x: x.strength, reverse=True)
        
        return zones
    
    def check_liquidity_sweep(
        self, 
        df: pd.DataFrame, 
        liquidity_zones: List[LiquidityZone]
    ) -> Dict:
        """
        Check if price has swept any liquidity zones.
        
        Returns:
            Dict with sweep information
        """
        if not liquidity_zones:
            return {"swept": False, "zone": None, "direction": None}
        
        current_price = df.iloc[-1]['close']
        
        for zone in liquidity_zones:
            if zone.type == "buy_side":
                # For buy-side liquidity (above price), check if price went above and came back
                if current_price < zone.price:
                    # Check if price recently went above
                    recent_highs = df['high'].tail(10)
                    if (recent_highs > zone.price).any():
                        return {
                            "swept": True,
                            "zone": zone,
                            "direction": "bullish",
                            "sweep_price": zone.price
                        }
            
            elif zone.type == "sell_side":
                # For sell-side liquidity (below price), check if price went below and came back
                if current_price > zone.price:
                    recent_lows = df['low'].tail(10)
                    if (recent_lows < zone.price).any():
                        return {
                            "swept": True,
                            "zone": zone,
                            "direction": "bearish",
                            "sweep_price": zone.price
                        }
        
        return {"swept": False, "zone": None, "direction": None}
    
    # ==================== ORDER BLOCKS ====================
    
    def find_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """
        Find order blocks (last bullish/bearish candle before a strong move).
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            List of OrderBlock objects
        """
        order_blocks = []
        lookback = self.config["order_block_lookback"]
        
        for i in range(lookback, len(df) - 1):
            # Check for bullish order block (before bullish move)
            # Look for strong bearish candle followed by bullish momentum
            current_candle = df.iloc[i]
            next_candles = df.iloc[i+1:min(i+5, len(df))]
            
            # Strong bearish candle
            if current_candle['close'] < current_candle['open']:
                body_size = abs(current_candle['close'] - current_candle['open'])
                range_size = current_candle['high'] - current_candle['low']
                
                if body_size / range_size > self.config["order_block_min_body_pct"] / 100:
                    # Check for bullish continuation
                    if len(next_candles) > 0 and next_candles['close'].iloc[-1] > current_candle['high']:
                        # Calculate entry and stop loss
                        entry = current_candle['high']
                        stop_loss = current_candle['low']
                        
                        order_blocks.append(OrderBlock(
                            start_index=i,
                            end_index=i,
                            type='bullish',
                            entry_price=entry,
                            stop_loss=stop_loss,
                            timeframe='unknown'
                        ))
            
            # Check for bearish order block
            if current_candle['close'] > current_candle['open']:
                body_size = abs(current_candle['close'] - current_candle['open'])
                range_size = current_candle['high'] - current_candle['low']
                
                if body_size / range_size > self.config["order_block_min_body_pct"] / 100:
                    if len(next_candles) > 0 and next_candles['close'].iloc[-1] < current_candle['low']:
                        entry = current_candle['low']
                        stop_loss = current_candle['high']
                        
                        order_blocks.append(OrderBlock(
                            start_index=i,
                            end_index=i,
                            type='bearish',
                            entry_price=entry,
                            stop_loss=stop_loss,
                            timeframe='unknown'
                        ))
        
        return order_blocks[-10:]  # Return last 10 order blocks
    
    # ==================== FAIR VALUE GAPS ====================
    
    def find_fair_value_gaps(self, df: pd.DataFrame) -> List[FairValueGap]:
        """
        Find Fair Value Gaps (imbalances between supply and demand).
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            List of FairValueGap objects
        """
        fvgs = []
        lookback = self.config["fvg_lookback"]
        
        for i in range(1, len(df) - 1):
            prev_candle = df.iloc[i-1]
            current_candle = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            # Bullish FVG: Previous low > Current high > Next low
            # (Gap up - no trading between)
            if prev_candle['low'] > current_candle['high']:
                gap_size = prev_candle['low'] - current_candle['high']
                avg_price = (prev_candle['low'] + current_candle['high']) / 2
                
                # Check if gap is significant
                if gap_size > self.config["fvg_min_gap_pips"] * df.iloc[i]['close'] / 10000:
                    fvgs.append(FairValueGap(
                        start_index=i-1,
                        end_index=i,
                        high=prev_candle['low'],
                        low=current_candle['high'],
                        type='bullish'
                    ))
            
            # Bearish FVG: Previous high < Current low < Next high
            elif prev_candle['high'] < current_candle['low']:
                gap_size = current_candle['low'] - prev_candle['high']
                avg_price = (prev_candle['high'] + current_candle['low']) / 2
                
                if gap_size > self.config["fvg_min_gap_pips"] * df.iloc[i]['close'] / 10000:
                    fvgs.append(FairValueGap(
                        start_index=i-1,
                        end_index=i,
                        high=current_candle['low'],
                        low=prev_candle['high'],
                        type='bearish'
                    ))
        
        return fvgs[-10:]  # Return last 10 FVGs
    
    # ==================== TREND ANALYSIS ====================
    
    def analyze_trend(self, df: pd.DataFrame) -> Dict:
        """Analyze trend using EMAs."""
        close = df['close']
        
        # Calculate EMAs
        ema_9 = close.ewm(span=9, adjust=False).mean()
        ema_21 = close.ewm(span=21, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()
        
        current_ema_9 = ema_9.iloc[-1]
        current_ema_21 = ema_21.iloc[-1]
        current_ema_50 = ema_50.iloc[-1]
        
        # Determine trend
        if current_ema_9 > current_ema_21 > current_ema_50:
            trend = "strong_uptrend"
        elif current_ema_9 < current_ema_21 < current_ema_50:
            trend = "strong_downtrend"
        elif current_ema_9 > current_ema_21:
            trend = "uptrend"
        elif current_ema_9 < current_ema_21:
            trend = "downtrend"
        else:
            trend = "neutral"
        
        return {
            "trend": trend,
            "ema_9": current_ema_9,
            "ema_21": current_ema_21,
            "ema_50": current_ema_50,
            "price_vs_ema9": (close.iloc[-1] - current_ema_9) / current_ema_9 * 100,
            "price_vs_ema21": (close.iloc[-1] - current_ema_21) / current_ema_21 * 100
        }
    
    # ==================== BREAK OF STRUCTURE ====================
    
    def check_break_of_structure(self, df: pd.DataFrame, structure: MarketStructure) -> Dict:
        """Check for break of structure (BOS)."""
        if len(df) < 10:
            return {"bos": False, "direction": None}
        
        recent_bars = df.tail(self.config["structure_break_bars"])
        
        # Get last swing points
        if structure.swing_highs and structure.swing_lows:
            last_high = structure.swing_highs[-1].price
            last_low = structure.swing_lows[-1].price
            
            # Check for bullish BOS (break above last high with momentum)
            if df.iloc[-1]['close'] > last_high:
                # Check if there's momentum (multiple candles closing higher)
                closes = recent_bars['close'].values
                if all(closes[i] < closes[i+1] for i in range(len(closes)-1)):
                    return {
                        "bos": True,
                        "direction": "bullish",
                        "break_price": last_high,
                        "current_price": df.iloc[-1]['close']
                    }
            
            # Check for bearish BOS
            if df.iloc[-1]['close'] < last_low:
                closes = recent_bars['close'].values
                if all(closes[i] > closes[i+1] for i in range(len(closes)-1)):
                    return {
                        "bos": True,
                        "direction": "bearish",
                        "break_price": last_low,
                        "current_price": df.iloc[-1]['close']
                    }
        
        return {"bos": False, "direction": None}
    
    # ==================== COMPLETE ANALYSIS ====================
    
    def analyze(self, df: pd.DataFrame, timeframe: str = "unknown") -> Dict:
        """
        Perform complete CHIBOY analysis on the data.
        
        Args:
            df: OHLCV DataFrame
            timeframe: Timeframe being analyzed
            
        Returns:
            Dict with complete analysis results
        """
        if df.empty or len(df) < 50:
            return {"error": "Insufficient data for analysis"}
        
        # Run all analyses
        structure = self.analyze_structure(df)
        liquidity_zones = self.find_liquidity_zones(df)
        order_blocks = self.find_order_blocks(df)
        fvgs = self.find_fair_value_gaps(df)
        trend = self.analyze_trend(df)
        liquidity_sweep = self.check_liquidity_sweep(df, liquidity_zones)
        bos = self.check_break_of_structure(df, structure)
        
        # Determine trading opportunity
        opportunity = self._determine_opportunity(
            structure=structure,
            trend=trend,
            liquidity_sweep=liquidity_sweep,
            order_blocks=order_blocks,
            fvgs=fvgs,
            bos=bos
        )
        
        return {
            "timeframe": timeframe,
            "current_price": df.iloc[-1]['close'],
            "structure": {
                "trend": structure.trend,
                "current_structure": structure.current_structure,
                "swing_highs_count": len(structure.swing_highs),
                "swing_lows_count": len(structure.swing_lows),
                "last_swing_high": structure.swing_highs[-1].price if structure.swing_highs else None,
                "last_swing_low": structure.swing_lows[-1].price if structure.swing_lows else None
            },
            "trend": trend,
            "liquidity_zones": [
                {"price": z.price, "type": z.type, "touches": z.touches, "strength": z.strength}
                for z in liquidity_zones[:5]
            ],
            "liquidity_sweep": liquidity_sweep,
            "order_blocks": [
                {"entry": ob.entry_price, "stop_loss": ob.stop_loss, "type": ob.type}
                for ob in order_blocks[-3:]
            ],
            "fair_value_gaps": [
                {"high": fvg.high, "low": fvg.low, "type": fvg.type}
                for fvg in fvgs[-3:]
            ],
            "break_of_structure": bos,
            "opportunity": opportunity
        }
    
    def _determine_opportunity(
        self,
        structure: MarketStructure,
        trend: Dict,
        liquidity_sweep: Dict,
        order_blocks: List[OrderBlock],
        fvgs: List[FairValueGap],
        bos: Dict
    ) -> Dict:
        """Determine if there's a trading opportunity based on CHIBOY criteria."""
        
        opportunity = {
            "exists": False,
            "direction": None,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "risk_reward": None,
            "confidence": 0,
            "reasons": []
        }
        
        # Count bullish/bearish signals
        bullish_signals = 0
        bearish_signals = 0
        
        # Check trend
        if trend["trend"] in ["strong_uptrend", "uptrend"]:
            bullish_signals += 1
            opportunity["reasons"].append(f"Trend: {trend['trend']}")
        elif trend["trend"] in ["strong_downtrend", "downtrend"]:
            bearish_signals += 1
            opportunity["reasons"].append(f"Trend: {trend['trend']}")
        
        # Check structure
        if structure.current_structure == "bullish":
            bullish_signals += 1
            opportunity["reasons"].append("Bullish structure")
        elif structure.current_structure == "bearish":
            bearish_signals += 1
            opportunity["reasons"].append("Bearish structure")
        
        # Check liquidity sweep
        if liquidity_sweep["swept"]:
            if liquidity_sweep["direction"] == "bullish":
                bullish_signals += 2
                opportunity["reasons"].append("Buy-side liquidity swept")
            else:
                bearish_signals += 2
                opportunity["reasons"].append("Sell-side liquidity swept")
        
        # Check order blocks
        if order_blocks:
            last_ob = order_blocks[-1]
            if last_ob.type == "bullish":
                bullish_signals += 1
                opportunity["reasons"].append("Bullish order block found")
            else:
                bearish_signals += 1
                opportunity["reasons"].append("Bearish order block found")
        
        # Check FVG
        if fvgs:
            last_fvg = fvgs[-1]
            if last_fvg.type == "bullish":
                bullish_signals += 1
                opportunity["reasons"].append("Bullish FVG found")
            else:
                bearish_signals += 1
                opportunity["reasons"].append("Bearish FVG found")
        
        # Check BOS
        if bos["bos"]:
            if bos["direction"] == "bullish":
                bullish_signals += 2
                opportunity["reasons"].append("Bullish break of structure")
            else:
                bearish_signals += 2
                opportunity["reasons"].append("Bearish break of structure")
        
        # Determine if opportunity exists
        if bullish_signals >= 3:
            opportunity["exists"] = True
            opportunity["direction"] = "long"
            opportunity["confidence"] = min(bullish_signals * 15, 95)
            
            # Set entry and stop loss from order block or FVG
            if order_blocks and order_blocks[-1].type == "bullish":
                opportunity["entry_price"] = order_blocks[-1].entry_price
                opportunity["stop_loss"] = order_blocks[-1].stop_loss
            else:
                opportunity["entry_price"] = None  # Use current price
                opportunity["stop_loss"] = structure.swing_lows[-1].price if structure.swing_lows else None
        
        elif bearish_signals >= 3:
            opportunity["exists"] = True
            opportunity["direction"] = "short"
            opportunity["confidence"] = min(bearish_signals * 15, 95)
            
            if order_blocks and order_blocks[-1].type == "bearish":
                opportunity["entry_price"] = order_blocks[-1].entry_price
                opportunity["stop_loss"] = order_blocks[-1].stop_loss
            else:
                opportunity["entry_price"] = None
                opportunity["stop_loss"] = structure.swing_highs[-1].price if structure.swing_highs else None
        
        # Calculate risk-reward if we have entry and stop loss
        if opportunity["entry_price"] and opportunity["stop_loss"]:
            risk = abs(opportunity["entry_price"] - opportunity["stop_loss"])
            # Assume 2:1 reward
            opportunity["take_profit"] = opportunity["entry_price"] + (
                risk * 2 if opportunity["direction"] == "long" else -risk * 2
            )
            opportunity["risk_reward"] = 2.0
        
        return opportunity
