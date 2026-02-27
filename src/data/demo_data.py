"""
Demo Data Generator
==================
Generates simulated market data for testing when API keys aren't available.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DemoDataGenerator:
    """Generate simulated market data for demo/testing."""
    
    def __init__(self):
        # Base prices for major pairs
        self.base_prices = {
            # Forex
            'EUR_USD': 1.0850,
            'GBP_USD': 1.2650,
            'USD_JPY': 149.50,
            'USD_CHF': 0.8820,
            'AUD_USD': 0.6550,
            'USD_CAD': 1.3550,
            'NZD_USD': 0.6080,
            'EUR_GBP': 0.8580,
            'EUR_JPY': 162.20,
            'GBP_JPY': 189.10,
            # Crypto
            'BTCUSDT': 43500.00,
            'ETHUSDT': 2650.00,
            'BNBUSDT': 315.00,
            'SOLUSDT': 98.50,
            'XRPUSDT': 0.52,
            'ADAUSDT': 0.48,
            'DOGEUSDT': 0.082,
            'MATICUSDT': 0.78,
        }
    
    def generate_candles(self, symbol: str, timeframe: str, count: int = 200) -> pd.DataFrame:
        """Generate simulated OHLCV data."""
        
        base_price = self.base_prices.get(symbol, 100.0)
        
        # Calculate volatility based on asset type
        is_crypto = symbol.endswith('USDT') and symbol != 'USDUSDT'
        volatility = 0.02 if is_crypto else 0.001
        
        # Generate price movements
        np.random.seed(hash(symbol) % 2**32)
        
        timestamps = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        # Start time
        if timeframe == '5m':
            start_time = datetime.now() - timedelta(minutes=5*count)
        elif timeframe == '15m':
            start_time = datetime.now() - timedelta(minutes=15*count)
        elif timeframe == '30m':
            start_time = datetime.now() - timedelta(minutes=30*count)
        elif timeframe == '1H':
            start_time = datetime.now() - timedelta(hours=count)
        elif timeframe == '4H':
            start_time = datetime.now() - timedelta(hours=4*count)
        elif timeframe == '1D':
            start_time = datetime.now() - timedelta(days=count)
        elif timeframe == '1W':
            start_time = datetime.now() - timedelta(weeks=count)
        else:
            start_time = datetime.now() - timedelta(hours=count)
        
        current_price = base_price
        
        for i in range(count):
            # Generate random price movement
            change = np.random.normal(0, volatility)
            open_price = current_price
            close_price = open_price * (1 + change)
            
            # Generate high/low with some randomness
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, volatility/2)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, volatility/2)))
            
            # Generate volume
            volume = np.random.randint(1000, 100000) if is_crypto else np.random.randint(100, 10000)
            
            # Calculate timestamp
            if timeframe == '15m':
                ts = start_time + timedelta(minutes=15*i)
            elif timeframe == '1H':
                ts = start_time + timedelta(hours=i)
            elif timeframe == '4H':
                ts = start_time + timedelta(hours=4*i)
            elif timeframe == '1D':
                ts = start_time + timedelta(days=i)
            elif timeframe == '1W':
                ts = start_time + timedelta(weeks=i)
            else:
                ts = start_time + timedelta(hours=i)
            
            timestamps.append(ts)
            opens.append(round(open_price, 5 if not is_crypto else 2))
            highs.append(round(high_price, 5 if not is_crypto else 2))
            lows.append(round(low_price, 5 if not is_crypto else 2))
            closes.append(round(close_price, 5 if not is_crypto else 2))
            volumes.append(volume)
            
            current_price = close_price
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        df.set_index('time', inplace=True)
        
        logger.debug(f"Generated {count} demo candles for {symbol} {timeframe}")
        
        return df
    
    def get_current_price(self, symbol: str) -> dict:
        """Get simulated current price."""
        base_price = self.base_prices.get(symbol, 100.0)
        is_crypto = symbol.endswith('USDT') and symbol != 'USDUSDT'
        
        # Add small random variation
        variation = np.random.normal(0, 0.0005 if not is_crypto else 0.001)
        price = base_price * (1 + variation)
        
        return {
            'price': round(price, 5 if not is_crypto else 2),
            'bid': round(price * 0.9995, 5 if not is_crypto else 2),
            'ask': round(price * 1.0005, 5 if not is_crypto else 2),
        }


# Singleton instance
demo_generator = DemoDataGenerator()
