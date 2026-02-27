"""Stub analyzer for when pandas is not available."""
import random
import time

class QuickAnalyzer:
    """Stub analyzer that returns demo data."""
    
    @staticmethod
    def get_top_opportunities(limit=10):
        return []
    
    @staticmethod
    def analyze_all():
        return []
    
    @staticmethod
    def analyze_ict(symbol, market_type, current_price):
        return {"signal": "hold", "entry": None, "stop_loss": None, "take_profit": None}
    
    @staticmethod
    def analyze_ict_strict(symbol, market_type, current_price):
        # Generate dummy data without pandas
        df = QuickAnalyzer._generate_data(symbol, current_price, '1H', 100)
        current_close = df[-1]['close']
        
        # Simple trend
        recent = [d['close'] for d in df[-10:]]
        ma = sum(recent) / len(recent)
        trend = "bullish" if current_close > ma else "bearish"
        
        return {
            "signal": "hold",
            "entry": round(current_close, 5),
            "stop_loss": round(current_close * 0.99, 5),
            "take_profit": round(current_close * 1.03, 5),
            "trend": trend,
            "confidence": 50
        }
    
    @staticmethod
    def _generate_data(symbol, current_price, tf, num_candles):
        # Map timeframe to interval
        intervals = {'1M': 60, '5M': 300, '15M': 900, '1H': 3600, '4H': 14400, '1D': 86400, '1W': 604800}
        interval = intervals.get(tf, 3600)
        
        # Generate random data
        random.seed(hash(symbol) % 2**32)
        base = current_price
        candles = []
        price = base
        
        for i in range(num_candles):
            change = random.uniform(-0.002, 0.002) * price
            close = price + change
            open_price = close + random.uniform(-0.001, 0.001) * price
            high = max(open_price, close) + random.uniform(0, 0.001) * price
            low = min(open_price, close) - random.uniform(0, 0.001) * price
            
            candles.append({
                'open': round(open_price, 5),
                'high': round(high, 5),
                'low': round(low, 5),
                'close': round(close, 5)
            })
            price = close
        
        return candles

class MultiTimeframeAnalyzer:
    """Stub multi-timeframe analyzer."""
    
    def __init__(self):
        pass
    
    def get_top_opportunities(self, limit=10):
        return []
