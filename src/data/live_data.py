"""
Live Data Provider - Yahoo Finance with Caching
==============================================
"""

import urllib.request
import json
from datetime import datetime
import logging
import ssl

from ..config import CUSTOM_SPREADS

logger = logging.getLogger(__name__)


class LiveDataProvider:
    """Live market data from Yahoo Finance with caching."""
    
    def __init__(self):
        self._last_update = None
        self._cache = {}  # Cache for prices
        self._cache_time = {}  # Cache timestamps
        self._cache_ttl = 10  # Cache for only 10 seconds for real-time feel
        try:
            self._ssl_context = ssl.create_default_context()
        except:
            self._ssl_context = None
    
    def _fetch(self, url: str) -> dict:
        """Make HTTP request."""
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            req.add_header('Accept', '*/*')
            
            if self._ssl_context:
                with urllib.request.urlopen(req, context=self._ssl_context, timeout=5) as response:
                    return json.loads(response.read().decode())
            else:
                with urllib.request.urlopen(req, timeout=5) as response:
                    return json.loads(response.read().decode())
        except Exception as e:
            logger.error(f"Fetch error: {e}")
            return {}
    
    def get_forex(self) -> dict:
        """Get forex from Yahoo Finance - with caching."""
        
        # Check cache first
        now = datetime.now().timestamp()
        if 'forex' in self._cache and 'forex' in self._cache_time:
            if now - self._cache_time['forex'] < self._cache_ttl:
                logger.info("Using cached forex prices")
                return self._cache['forex']
        
        result = {}
        
        # Major forex pairs + XAUUSD (Gold)
        forex_pairs = [
            'EURUSD=X', 'GBPUSD=X', 'JPY=X', 'CHF=X', 
            'AUDUSD=X', 'CAD=X', 'NZDUSD=X', 'EURGBP=X', 
            'EURJPY=X', 'GBPJPY=X', 'GC=F'  # Gold futures
        ]
        
        try:
            # Try individual chart requests (more reliable)
            for y_sym in forex_pairs:
                try:
                    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{y_sym}?interval=1m&range=1d"
                    data = self._fetch(url)
                    
                    if data and 'chart' in data and data['chart'].get('result'):
                        price = data['chart']['result'][0]['meta']['regularMarketPrice']
                        
                        # Map to our symbol format (e.g., GBPUSD -> GBP_USD)
                        raw = y_sym.replace('=X', '')
                        
                        # Handle special cases
                        if y_sym == 'GC=F':
                            sym = 'XAUUSD'
                        elif y_sym == 'JPY=X':
                            sym = 'USD_JPY'
                        elif y_sym == 'CHF=X':
                            sym = 'USD_CHF'
                        elif y_sym == 'CAD=X':
                            sym = 'USD_CAD'
                        elif raw == 'JPY':
                            sym = 'USD_JPY'
                        else:
                            # For pairs like EURUSD, GBPUSD - add underscore
                            if len(raw) == 6:
                                sym = raw[:3] + '_' + raw[3:]
                            else:
                                sym = raw
                        
                        spread = 0.0001 if 'JPY' not in y_sym and y_sym != 'GC=F' else 0.01
                        if y_sym == 'GC=F':
                            spread = 0.50  # Gold spread
                            
                        decimals = 2 if y_sym == 'GC=F' or 'JPY' in y_sym else 5
                        
                        # Apply custom spread adjustment for broker matching
                        adjusted_price = price
                        if sym in CUSTOM_SPREADS:
                            adjustment = CUSTOM_SPREADS[sym]
                            adjusted_price = price - adjustment
                            logger.info(f"Applied spread adjustment for {sym}: {price} -> {adjusted_price} ( -{adjustment})")
                        
                        result[sym] = {
                            'price': adjusted_price,
                            'bid': round(adjusted_price - spread/2, decimals),
                            'ask': round(adjusted_price + spread/2, decimals)
                        }
                        logger.info(f"Got {sym}: {adjusted_price}")
                except Exception as e:
                    logger.error(f"Error fetching {y_sym}: {e}")
                    continue
            
            if result:
                return result
                    
            # Fallback to batch if individual failed
            symbols = ','.join(forex_pairs)
            url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbols}"
            
            data = self._fetch(url)
            
            if data and 'quoteResponse' in data and data['quoteResponse'].get('result'):
                for quote in data['quoteResponse']['result']:
                    # Find matching symbol
                    for sym, y_sym in forex_pairs.items():
                        if y_sym == quote.get('symbol'):
                            price = quote.get('regularMarketPrice')
                            if price:
                                spread = 0.0001 if 'JPY' not in sym else 0.01
                                result[sym] = {
                                    'price': price,
                                    'bid': round(price - spread/2, 5 if 'JPY' not in sym else 3),
                                    'ask': round(price + spread/2, 5 if 'JPY' not in sym else 3)
                                }
                                logger.info(f"Got {sym}: {price}")
                                break
                            
        except Exception as e:
            logger.error(f"Yahoo error: {e}")
        
        # Fallback
        if not result:
            result = {
                'EUR_USD': {'price': 1.0850, 'bid': 1.0849, 'ask': 1.0851},
                'GBP_USD': {'price': 1.3541, 'bid': 1.3540, 'ask': 1.3542},
                'USD_JPY': {'price': 149.50, 'bid': 149.49, 'ask': 149.51},
                'USD_CHF': {'price': 0.8820, 'bid': 0.8819, 'ask': 0.8821},
                'AUD_USD': {'price': 0.6520, 'bid': 0.6519, 'ask': 0.6521},
                'USD_CAD': {'price': 1.3580, 'bid': 1.3579, 'ask': 1.3581},
                'NZD_USD': {'price': 0.6120, 'bid': 0.6119, 'ask': 0.6121},
                'EUR_GBP': {'price': 0.8010, 'bid': 0.8009, 'ask': 0.8011},
                'EUR_JPY': {'price': 162.30, 'bid': 162.29, 'ask': 162.31},
                'GBP_JPY': {'price': 110.40, 'bid': 110.39, 'ask': 110.41},
                'XAUUSD': {'price': 2850.00, 'bid': 2849.50, 'ask': 2850.50}
            }
        
        # Cache the result
        if result:
            self._cache['forex'] = result
            self._cache_time['forex'] = datetime.now().timestamp()
        
        return result
    
    def get_indices(self) -> dict:
        """Get indices from Yahoo Finance."""
        result = {}
        
        # Index symbols for Yahoo Finance
        indices = [
            ('NAS100', '^NDX'),    # Nasdaq 100
            ('SPX500', '^GSPC'),   # S&P 500
            ('US30', '^DJI'),      # Dow Jones
            ('USOIL', 'CL=F'),     # Crude Oil
        ]
        
        for sym, y_sym in indices:
            try:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{y_sym}?interval=1m&range=1d"
                data = self._fetch(url)
                
                if data and 'chart' in data and data['chart'].get('result'):
                    price = data['chart']['result'][0]['meta']['regularMarketPrice']
                    spread = price * 0.0005  # 0.05% spread
                    
                    result[sym] = {
                        'price': round(price, 2),
                        'bid': round(price - spread, 2),
                        'ask': round(price + spread, 2)
                    }
                    logger.info(f"Got {sym}: {price}")
            except Exception as e:
                logger.error(f"Error fetching {sym}: {e}")
                # Fallback prices
                fallback = {'NAS100': 17850, 'SPX500': 5020, 'US30': 38500, 'USOIL': 78.50}
                result[sym] = {
                    'price': fallback.get(sym, 100),
                    'bid': round(fallback.get(sym, 100) * 0.9995, 2),
                    'ask': round(fallback.get(sym, 100) * 1.0005, 2)
                }
        
        return result
    
    def get_crypto(self) -> dict:
        """Get crypto from CoinGecko."""
        result = {}
        
        try:
            url = 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,binancecoin,solana,ripple,cardano,dogecoin&vs_currencies=usd'
            data = self._fetch(url)
            
            if data:
                mapping = {
                    'bitcoin': 'BTCUSDT', 'ethereum': 'ETHUSDT', 
                    'binancecoin': 'BNBUSDT', 'solana': 'SOLUSDT',
                    'ripple': 'XRPUSDT', 'cardano': 'ADAUSDT', 
                    'dogecoin': 'DOGEUSDT'
                }
                
                for coin, sym in mapping.items():
                    if coin in data:
                        price = data[coin]['usd']
                        result[sym] = {
                            'price': price,
                            'bid': round(price * 0.9995, 2),
                            'ask': round(price * 1.0005, 2)
                        }
        except Exception as e:
            logger.error(f"Crypto error: {e}")
        
        if not result:
            result = {
                'BTCUSDT': {'price': 68261, 'bid': 68227, 'ask': 68295},
                'ETHUSDT': {'price': 2058, 'bid': 2057, 'ask': 2059},
            }
        
        return result
    
    def get_all(self) -> dict:
        """Get all prices."""
        prices = {}
        prices.update(self.get_forex())
        prices.update(self.get_indices())
        prices.update(self.get_crypto())
        self._last_update = datetime.now()
        return prices
    
    def get(self, symbol: str) -> dict:
        """Get single symbol."""
        return self.get_all().get(symbol, {})


# Singleton
live_data = LiveDataProvider()
