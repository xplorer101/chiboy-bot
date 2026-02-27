"""
Binance API Connector
=====================
Handles all Binance API operations for crypto data and trade execution.
Supports both testnet and live trading.
"""

from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import datetime
import pandas as pd
import logging
from typing import Optional, List, Dict, Any

from ..config import BINANCE_CONFIG

logger = logging.getLogger(__name__)


class BinanceClient:
    """Binance API client for cryptocurrency trading."""
    
    def __init__(self, api_key: str = None, secret_key: str = None, testnet: bool = None):
        self.api_key = api_key or BINANCE_CONFIG["api_key"]
        self.secret_key = secret_key or BINANCE_CONFIG["secret_key"]
        self.testnet = testnet if testnet is not None else BINANCE_CONFIG["testnet"]
        
        # Initialize the Binance client
        try:
            self.client = Client(self.api_key, self.secret_key, testnet=self.testnet)
            logger.info(f"Binance Client initialized (testnet: {self.testnet})")
        except Exception as e:
            logger.error(f"Error initializing Binance client: {e}")
            self.client = None
    
    # ==================== DATA FETCHING ====================
    
    def get_klines(
        self, 
        symbol: str, 
        interval: str, 
        limit: int = 200,
        start_str: str = None,
        end_str: str = None
    ) -> pd.DataFrame:
        """
        Fetch historical kline (candle) data from Binance.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe (15m, 1h, 4h, 1d, 1w)
            limit: Number of klines to fetch (max 1000)
            start_str: Start time (optional)
            end_str: End time (optional)
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.client:
            logger.error("Binance client not initialized")
            return pd.DataFrame()
        
        # Map interval
        binance_interval = BINANCE_CONFIG["interval_map"].get(interval, interval)
        
        try:
            if start_str and end_str:
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=binance_interval,
                    startStr=start_str,
                    endStr=end_str,
                    limit=limit
                )
            else:
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=binance_interval,
                    limit=limit
                )
            
            # Parse klines
            data = []
            for k in klines:
                data.append({
                    "time": pd.to_datetime(k[0], unit="ms"),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_time": k[6],
                    "quote_volume": float(k[7]),
                    "trades": int(k[8])
                })
            
            df = pd.DataFrame(data)
            df.set_index("time", inplace=True)
            
            logger.debug(f"Fetched {len(df)} klines for {symbol} {interval}")
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error fetching klines: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current price for a symbol."""
        if not self.client:
            return None
        
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return {
                "symbol": ticker["symbol"],
                "price": float(ticker["price"])
            }
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None
    
    def get_24h_stats(self, symbol: str) -> Optional[Dict]:
        """Get 24-hour price change statistics."""
        if not self.client:
            return None
        
        try:
            stats = self.client.get_ticker(symbol=symbol)
            return {
                "symbol": stats["symbol"],
                "lastPrice": float(stats["lastPrice"]),
                "bidPrice": float(stats["bidPrice"]),
                "askPrice": float(stats["askPrice"]),
                "volume": float(stats["volume"]),
                "quoteVolume": float(stats["quoteVolume"]),
                "priceChange": float(stats["priceChange"]),
                "priceChangePercent": float(stats["priceChangePercent"]),
                "highPrice": float(stats["highPrice"]),
                "lowPrice": float(stats["lowPrice"])
            }
        except Exception as e:
            logger.error(f"Error fetching 24h stats for {symbol}: {e}")
            return None
    
    def get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """Get order book depth."""
        if not self.client:
            return None
        
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=limit)
            return {
                "bids": [[float(p[0]), float(p[1])] for p in depth["bids"]],
                "asks": [[float(p[0]), float(p[1])] for p in depth["asks"]],
                "lastUpdateId": depth["lastUpdateId"]
            }
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return None
    
    # ==================== ACCOUNT INFO ====================
    
    def get_account(self) -> Optional[Dict]:
        """Get account information."""
        if not self.client:
            return None
        
        try:
            return self.client.get_account()
        except Exception as e:
            logger.error(f"Error fetching account: {e}")
            return None
    
    def get_balance(self, asset: str = "USDT") -> Optional[float]:
        """Get balance for a specific asset."""
        account = self.get_account()
        if account:
            for balance in account["balances"]:
                if balance["asset"] == asset:
                    return float(balance.get("free", 0))
        return None
    
    def get_all_balances(self) -> Dict[str, float]:
        """Get all asset balances."""
        account = self.get_account()
        if account:
            return {
                b["asset"]: float(b.get("free", 0))
                for b in account["balances"]
                if float(b.get("free", 0)) > 0
            }
        return {}
    
    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get open orders."""
        if not self.client:
            return []
        
        try:
            if symbol:
                return self.client.get_open_orders(symbol=symbol)
            return self.client.get_open_orders()
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            return []
    
    # ==================== ORDER EXECUTION ====================
    
    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        test: bool = True
    ) -> Optional[Dict]:
        """
        Place a market order.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            side: 'BUY' or 'SELL'
            quantity: Quantity to trade
            test: If True, test order (doesn't execute)
            
        Returns:
            Order response or None
        """
        if not self.client:
            logger.error("Binance client not initialized")
            return None
        
        try:
            if test:
                # Test order - validate but don't execute
                result = self.client.create_test_order(
                    symbol=symbol,
                    side=side,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                logger.info(f"Test order placed: {side} {quantity} {symbol}")
                return result
            else:
                result = self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                logger.info(f"Order executed: {side} {quantity} {symbol}")
                return result
        except BinanceAPIException as e:
            logger.error(f"Binance API error placing order: {e}")
            return None
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        test: bool = True
    ) -> Optional[Dict]:
        """Place a limit order."""
        if not self.client:
            return None
        
        try:
            if test:
                result = self.client.create_test_order(
                    symbol=symbol,
                    side=side,
                    type=Client.ORDER_TYPE_LIMIT,
                    quantity=quantity,
                    price=price,
                    timeInForce=Client.TIME_IN_FORCE_GTC
                )
                logger.info(f"Test limit order placed: {side} {quantity} {symbol} @ {price}")
                return result
            else:
                result = self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type=Client.ORDER_TYPE_LIMIT,
                    quantity=quantity,
                    price=price,
                    timeInForce=Client.TIME_IN_FORCE_GTC
                )
                logger.info(f"Limit order placed: {side} {quantity} {symbol} @ {price}")
                return result
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return None
    
    def place_buy_order(
        self,
        symbol: str,
        quantity: float,
        stop_loss: float = None,
        take_profit: float = None,
        test: bool = True
    ) -> Optional[Dict]:
        """Place a buy market order."""
        return self.place_market_order(symbol, "BUY", quantity, test)
    
    def place_sell_order(
        self,
        symbol: str,
        quantity: float,
        stop_loss: float = None,
        take_profit: float = None,
        test: bool = True
    ) -> Optional[Dict]:
        """Place a sell market order."""
        return self.place_market_order(symbol, "SELL", quantity, test)
    
    def cancel_order(self, symbol: str, order_id: int) -> Optional[Dict]:
        """Cancel an existing order."""
        if not self.client:
            return None
        
        try:
            return self.client.cancel_order(symbol=symbol, orderId=order_id)
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return None
    
    def get_order(self, symbol: str, order_id: int) -> Optional[Dict]:
        """Get details of a specific order."""
        if not self.client:
            return None
        
        try:
            return self.client.get_order(symbol=symbol, orderId=order_id)
        except Exception as e:
            logger.error(f"Error fetching order: {e}")
            return None
    
    # ==================== UTILITY METHODS ====================
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol trading rules."""
        if not self.client:
            return None
        
        try:
            return self.client.get_symbol_info(symbol)
        except Exception as e:
            logger.error(f"Error fetching symbol info: {e}")
            return None
    
    def get_exchange_info(self) -> Dict:
        """Get exchange trading rules."""
        if not self.client:
            return {}
        
        try:
            return self.client.get_exchange_info()
        except Exception as e:
            logger.error(f"Error fetching exchange info: {e}")
            return {}


# Factory function
def create_binance_client() -> BinanceClient:
    """Create and return a Binance client instance."""
    return BinanceClient()
