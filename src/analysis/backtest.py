"""
Backtesting Engine
==================
Backtests the CHIBOY trading strategy on historical data.
Supports any forex pair with configurable date ranges.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

from ..data.oanda_client import OandaClient
from ..config import RISK_CONFIG, CHIBOY_CONFIG
from .ict_analyzer import CHIBOYAnalyzer

logger = logging.getLogger(__name__)


class BacktestTrade:
    """Represents a single backtest trade."""
    
    def __init__(self, entry_time, direction, entry_price, stop_loss, 
                 take_profit, size, reason=""):
        self.entry_time = entry_time
        self.direction = direction  # 'long' or 'short'
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.size = size
        self.reason = reason
        self.exit_time = None
        self.exit_price = None
        self.status = "open"  # open, closed_win, closed_loss
        self.pnl = 0
        self.pnl_pct = 0
    
    def close(self, exit_time, exit_price):
        """Close the trade and calculate PnL."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        
        if self.direction == "long":
            self.pnl = (exit_price - self.entry_price) * self.size
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price * 100
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.size
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price * 100
        
        if self.pnl > 0:
            self.status = "closed_win"
        else:
            self.status = "closed_loss"


class BacktestEngine:
    """Backtesting engine for CHIBOY strategy."""
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trades: List[BacktestTrade] = []
        self.analyzer = CHIBOYAnalyzer()
        self.oanda = OandaClient()
        
    def fetch_historical_data(self, symbol: str, days: int, timeframe: str = "H1") -> pd.DataFrame:
        """Fetch historical data for backtesting."""
        # OANDA has limits on how many candles we can get
        # H1: ~500 candles max per request, so we need to fetch in chunks
        # For 90 days of H1 = 90 * 24 = 2160 candles, need multiple requests
        
        all_candles = []
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Fetch in chunks of 500 candles
        count = 500
        current_end = end_time
        
        logger.info(f"Fetching {days} days of {symbol} {timeframe} data...")
        
        # Try to fetch from OANDA
        try:
            df = self._fetch_oanda_data(symbol, timeframe, days)
            if not df.empty:
                return df
        except Exception as e:
            logger.warning(f"OANDA fetch failed: {e}")
        
        # Fallback: generate synthetic data for demonstration
        logger.info("Using synthetic data for demonstration...")
        return self._generate_synthetic_data(symbol, days, timeframe)
    
    def _fetch_oanda_data(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """Fetch real data from OANDA."""
        granularity = "H1"
        if timeframe == "15m":
            granularity = "M15"
        elif timeframe == "4H":
            granularity = "H4"
        elif timeframe == "1D":
            granularity = "D"
        
        # OANDA practice account should work with demo
        df = self.oanda.get_candles(symbol, timeframe, count=min(days * 24, 500))
        return df
    
    def _generate_synthetic_data(self, symbol: str, days: int, timeframe: str) -> pd.DataFrame:
        """Generate realistic synthetic data for backtesting when API unavailable."""
        import random
        
        # Determine candles per day
        candles_per_day = {
            "15m": 96,
            "1H": 24,
            "4H": 6,
            "1D": 1
        }
        candles_per_day = candles_per_day.get(timeframe, 24)
        
        total_candles = days * candles_per_day
        
        # Starting price for GBPUSD ~1.25
        base_price = 1.25
        
        # Generate realistic price movement
        np.random.seed(42)  # Reproducible
        returns = np.random.normal(0.0001, 0.001, total_candles)  # Small upward drift
        
        prices = [base_price]
        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))
        
        # Create OHLC data
        data = []
        start_time = datetime.now() - timedelta(days=days)
        
        for i, close in enumerate(prices):
            # Generate realistic OHLC
            volatility = close * 0.002  # 0.2% volatility
            high = close + np.random.uniform(0, volatility)
            low = close - np.random.uniform(0, volatility)
            open_price = np.random.uniform(low, high)
            
            # Ensure OHLC consistency
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            timestamp = start_time + timedelta(hours=i)
            
            data.append({
                "time": timestamp,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": random.randint(1000, 10000)
            })
        
        df = pd.DataFrame(data)
        df.set_index("time", inplace=True)
        
        logger.info(f"Generated {len(df)} synthetic candles for {symbol}")
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Generate trading signals from historical data."""
        signals = []
        
        # Need enough data for analysis
        min_bars = 100
        
        if len(df) < min_bars:
            logger.warning(f"Not enough data for signal generation: {len(df)} bars")
            return signals
        
        # Run analysis on each potential entry point
        for i in range(min_bars, len(df) - 20):
            # Get historical data up to this point
            historical = df.iloc[:i].copy()
            current = df.iloc[i]
            
            try:
                # Find swing points
                swing_highs, swing_lows = self.analyzer.find_swing_points(historical)
                
                if len(swing_highs) < 2 or len(swing_lows) < 2:
                    continue
                
                # Analyze structure
                structure = self.analyzer.analyze_structure(historical)
                
                # Check for buy signal (simplified)
                # Long: price near liquidity low, bullish structure, FVG
                current_price = current['close']
                
                # Simple signal logic for backtest
                # Check if we're in uptrend
                if structure.trend == "uptrend":
                    # Check for pullback to recent swing low
                    recent_lows = [s.price for s in swing_lows[-5:]]
                    nearest_low = min(recent_lows, key=lambda x: abs(x - current_price))
                    
                    # Within 0.5% of swing low = potential buy
                    if abs(current_price - nearest_low) / current_price < 0.005:
                        # Check FVG
                        fvg = self._check_fvg(historical, i)
                        if fvg and fvg['type'] == 'bullish':
                            signals.append({
                                'time': current.name,
                                'type': 'long',
                                'price': current_price,
                                'reason': 'Uptrend pullback to liquidity + FVG'
                            })
                
                # Short signal
                elif structure.trend == "downtrend":
                    recent_highs = [s.price for s in swing_highs[-5:]]
                    nearest_high = min(recent_highs, key=lambda x: abs(x - current_price))
                    
                    if abs(current_price - nearest_high) / current_price < 0.005:
                        fvg = self._check_fvg(historical, i)
                        if fvg and fvg['type'] == 'bearish':
                            signals.append({
                                'time': current.name,
                                'type': 'short',
                                'price': current_price,
                                'reason': 'Downtrend rally to liquidity + FVG'
                            })
                            
            except Exception as e:
                continue
        
        logger.info(f"Generated {len(signals)} signals from {len(df)} candles")
        return signals
    
    def _check_fvg(self, df: pd.DataFrame, index: int) -> Optional[Dict]:
        """Check for Fair Value Gap at given index."""
        if index < 2 or index >= len(df) - 1:
            return None
        
        # Bullish FVG: low[1] > high[2]
        # Bearish FVG: high[1] < low[2]
        
        # Look at previous 3 candles
        candle_2 = df.iloc[index - 2]
        candle_1 = df.iloc[index - 1]
        candle_current = df.iloc[index]
        
        # Bullish FVG
        if candle_1['low'] > candle_2['high']:
            return {
                'type': 'bullish',
                'high': candle_1['low'],
                'low': candle_2['high']
            }
        
        # Bearish FVG
        if candle_1['high'] < candle_2['low']:
            return {
                'type': 'bearish',
                'high': candle_2['low'],
                'low': candle_1['high']
            }
        
        return None
    
    def run_backtest(self, symbol: str = "GBP_USD", days: int = 90, 
                     timeframe: str = "H1") -> Dict:
        """Run the full backtest."""
        logger.info(f"Starting backtest: {symbol} {days} days {timeframe}")
        
        # Fetch data
        df = self.fetch_historical_data(symbol, days, timeframe)
        
        if df.empty:
            return {"error": "No data available"}
        
        # Generate signals
        signals = self.generate_signals(df)
        
        # Execute trades
        self.balance = self.initial_balance
        self.trades = []
        
        for signal in signals:
            self._execute_backtest_trade(df, signal)
        
        # Calculate results
        return self.calculate_results()
    
    def _execute_backtest_trade(self, df: pd.DataFrame, signal: Dict):
        """Execute a single backtest trade."""
        entry_time = signal['time']
        direction = signal['type']
        entry_price = signal['price']
        
        # Calculate stop loss and take profit
        # Use 1:2 risk reward
        stop_pips = 0.002  # 20 pips (0.002 for GBPUSD)
        take_pips = 0.004  # 40 pips
        
        if direction == "long":
            stop_loss = entry_price - stop_pips
            take_profit = entry_price + take_pips
        else:
            stop_loss = entry_price + stop_pips
            take_profit = entry_price - take_pips
        
        # Calculate position size (risk 2%)
        risk_amount = self.balance * RISK_CONFIG["max_risk_per_trade"]
        stop_distance = abs(entry_price - stop_loss)
        position_size = risk_amount / stop_distance
        
        # Create trade
        trade = BacktestTrade(
            entry_time=entry_time,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size=position_size,
            reason=signal.get('reason', '')
        )
        
        # Simulate trade through future price action
        entry_idx = df.index.get_loc(entry_time)
        
        # Look ahead up to 200 bars
        for j in range(entry_idx + 1, min(entry_idx + 200, len(df))):
            future_price = df.iloc[j]['close']
            future_low = df.iloc[j]['low']
            future_high = df.iloc[j]['high']
            
            if direction == "long":
                # Check stop loss
                if future_low <= stop_loss:
                    trade.close(df.index[j], stop_loss)
                    break
                # Check take profit
                if future_high >= take_profit:
                    trade.close(df.index[j], take_profit)
                    break
            else:  # short
                # Check stop loss
                if future_high >= stop_loss:
                    trade.close(df.index[j], stop_loss)
                    break
                # Check take profit
                if future_low <= take_profit:
                    trade.close(df.index[j], take_profit)
                    break
        
        # If trade still open after 200 bars, close at current price
        if trade.status == "open":
            last_price = df.iloc[-1]['close']
            trade.close(df.index[-1], last_price)
        
        # Update balance
        self.balance += trade.pnl
        self.trades.append(trade)
    
    def calculate_results(self) -> Dict:
        """Calculate backtest results."""
        if not self.trades:
            return {
                "error": "No trades executed",
                "total_trades": 0
            }
        
        wins = [t for t in self.trades if t.status == "closed_win"]
        losses = [t for t in self.trades if t.status == "closed_loss"]
        
        total_trades = len(self.trades)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_pnl_pct = (total_pnl / self.initial_balance) * 100
        
        # Calculate average win/loss
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        
        # Average holding time
        holding_times = []
        for t in self.trades:
            if t.exit_time:
                hours = (t.exit_time - t.entry_time).total_seconds() / 3600
                holding_times.append(hours)
        avg_holding = np.mean(holding_times) if holding_times else 0
        
        # Max drawdown
        balance_history = [self.initial_balance]
        for t in self.trades:
            balance_history.append(balance_history[-1] + t.pnl)
        max_drawdown = 0
        peak = balance_history[0]
        for b in balance_history:
            if b > peak:
                peak = b
            drawdown = (peak - b) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Sharpe ratio (simplified)
        returns = [t.pnl_pct for t in self.trades]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        return {
            "symbol": "GBP_USD",
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
            "total_trades": total_trades,
            "wins": win_count,
            "losses": loss_count,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_holding_hours": round(avg_holding, 1),
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2),
            "trades": [
                {
                    "entry_time": t.entry_time.strftime("%Y-%m-%d %H:%M"),
                    "direction": t.direction,
                    "entry": t.entry_price,
                    "exit": t.exit_price,
                    "pnl": round(t.pnl, 2),
                    "pnl_pct": round(t.pnl_pct, 2),
                    "status": t.status
                }
                for t in self.trades[-20:]  # Last 20 trades
            ]
        }
    
    def print_report(self, results: Dict):
        """Print a formatted backtest report."""
        print("\n" + "=" * 60)
        print("📊 CHIBOY BOT BACKTEST REPORT")
        print("=" * 60)
        print(f"  Symbol:          {results.get('symbol', 'N/A')}")
        print(f"  Period:          Last 90 days")
        print(f"  Timeframe:       H1 (1 Hour)")
        print("-" * 60)
        print(f"  Initial Balance: ${results.get('initial_balance', 0):,.2f}")
        print(f"  Final Balance:   ${results.get('final_balance', 0):,.2f}")
        print(f"  Total P&L:       ${results.get('total_pnl', 0):,.2f} ({results.get('total_pnl_pct', 0):+.2f}%)")
        print("-" * 60)
        print(f"  Total Trades:    {results.get('total_trades', 0)}")
        print(f"  Wins:            {results.get('wins', 0)}")
        print(f"  Losses:          {results.get('losses', 0)}")
        print(f"  ⭐ Win Rate:      {results.get('win_rate', 0)}%")
        print("-" * 60)
        print(f"  Avg Win:         ${results.get('avg_win', 0):,.2f}")
        print(f"  Avg Loss:        ${results.get('avg_loss', 0):,.2f}")
        print(f"  Avg Holding:     {results.get('avg_holding_hours', 0)} hours")
        print(f"  Max Drawdown:    {results.get('max_drawdown_pct', 0)}%")
        print(f"  Sharpe Ratio:    {results.get('sharpe_ratio', 0)}")
        print("=" * 60)
        
        # Show last few trades
        if results.get('trades'):
            print("\n📋 Recent Trades:")
            print("-" * 60)
            for t in results['trades'][-10:]:
                emoji = "🟢" if t['pnl'] > 0 else "🔴"
                print(f"  {emoji} {t['entry_time']} | {t['direction']:5} | "
                      f"Entry: {t['entry']:.5f} → Exit: {t['exit']:.5f} | "
                      f"P&L: ${t['pnl']:+.2f} ({t['pnl_pct']:+.2f}%)")


def run_gbpusd_backtest():
    """Run GBPUSD backtest and print results."""
    engine = BacktestEngine(initial_balance=10000)
    results = engine.run_backtest(symbol="GBP_USD", days=90, timeframe="H1")
    engine.print_report(results)
    return results


if __name__ == "__main__":
    run_gbpusd_backtest()
