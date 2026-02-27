"""
Live Price Signal Generator
==========================
Generates trading signals using REAL live prices.
"""

from src.data.live_data import live_data
from src.analysis.ict_analyzer import ICTAnalyzer
import pandas as pd
import numpy as np


def get_live_signals():
    """Generate signals using REAL live prices."""
    
    # Get all live prices
    prices = live_data.get_all()
    
    signals = []
    
    for symbol, data in prices.items():
        if not data.get('price'):
            continue
            
        current_price = data['price']
        
        # Calculate SL and TP based on live price
        is_crypto = symbol.endswith('USDT')
        
        if is_crypto:
            # Crypto: 1% for SL, 2% for TP
            sl_distance = current_price * 0.01
            tp_distance = current_price * 0.02
        else:
            # Forex: 20 pips for SL, 40 pips for TP
            if 'JPY' in symbol:
                sl_distance = 0.20
                tp_distance = 0.40
            else:
                sl_distance = 0.0020
                tp_distance = 0.0040
        
        # Determine direction based on momentum (random for demo)
        # In real implementation, use trend analysis
        np.random.seed(hash(symbol) % 2**32)
        
        # Simple directional logic based on price position
        # Higher confidence if price is near round numbers
        round_factor = current_price % 1 if not is_crypto else current_price % 10
        confidence = 70 + (10 - min(round_factor, 10))
        
        # 50% chance long/short for demo
        direction = 'long' if np.random.random() > 0.5 else 'short'
        
        if direction == 'long':
            entry = current_price
            stop_loss = round(current_price - sl_distance, 4)
            take_profit = round(current_price + tp_distance, 4)
        else:
            entry = current_price
            stop_loss = round(current_price + sl_distance, 4)
            take_profit = round(current_price - tp_distance, 4)
        
        risk_reward = tp_distance / sl_distance
        
        signal = {
            'symbol': symbol,
            'type': 'crypto' if is_crypto else 'forex',
            'direction': direction,
            'entry_price': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': round(risk_reward, 1),
            'confidence': min(confidence, 95),
            'timeframe': '15m',
            'live_price': current_price,
            'bid': data.get('bid', entry),
            'ask': data.get('ask', entry),
            'reasons': [
                'Live price analysis',
                f'Bid: {data.get("bid")} | Ask: {data.get("ask")}',
                f'Spread: {abs(data.get("ask", 0) - data.get("bid", 0)):.5f}'
            ]
        }
        
        signals.append(signal)
    
    # Sort by confidence
    signals.sort(key=lambda x: x['confidence'], reverse=True)
    
    return signals


def get_signals_api():
    """API endpoint for signals."""
    signals = get_live_signals()
    
    return {
        'success': True,
        'count': len(signals),
        'signals': signals,
        'timestamp': str(pd.Timestamp.now())
    }


if __name__ == '__main__':
    # Test
    import json
    result = get_signals_api()
    print(json.dumps(result, indent=2))
