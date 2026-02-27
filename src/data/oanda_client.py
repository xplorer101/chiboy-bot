"""
OANDA API Connector
===================
Handles all OANDA API operations for forex data and trade execution.
Supports both practice (demo) and live accounts.
"""

import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.accounts as accounts
from datetime import datetime, timedelta
import pandas as pd
import logging
from typing import Optional, List, Dict, Any

from ..config import OANDA_CONFIG

logger = logging.getLogger(__name__)


class OandaClient:
    """OANDA API v20 client for forex trading."""
    
    def __init__(self, api_key: str = None, account_id: str = None, environment: str = None):
        self.api_key = api_key or OANDA_CONFIG["api_key"]
        self.account_id = account_id or OANDA_CONFIG["account_id"]
        self.environment = environment or OANDA_CONFIG["environment"]
        
        # Initialize the OANDA client
        self.client = oandapyV20.API(access_token=self.api_key, environment=self.environment)
        
        logger.info(f"OANDA Client initialized for {self.environment} account")
    
    # ==================== DATA FETCHING ====================
    
    def get_candles(self, instrument: str, timeframe: str, count: int = 200) -> pd.DataFrame:
        """
        Fetch historical candle data from OANDA.
        
        Args:
            instrument: Forex pair (e.g., 'EUR_USD')
            timeframe: Timeframe (M15, H1, H4, D, W)
            count: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        granularity = OANDA_CONFIG["granularity_map"].get(timeframe, "H1")
        
        params = {
            "granularity": granularity,
            "count": count,
            "price": "M"  # Midpoint price
        }
        
        try:
            request = instruments.InstrumentsCandles(instrument=instrument, params=params)
            response = self.client.request(request)
            
            candles = []
            for candle in response.get("candles", []):
                if candle["complete"]:  # Only complete candles
                    candles.append({
                        "time": pd.to_datetime(candle["time"]),
                        "open": float(candle["mid"]["o"]),
                        "high": float(candle["mid"]["h"]),
                        "low": float(candle["mid"]["l"]),
                        "close": float(candle["mid"]["c"]),
                        "volume": int(candle["volume"])
                    })
            
            df = pd.DataFrame(candles)
            df.set_index("time", inplace=True)
            
            logger.debug(f"Fetched {len(df)} candles for {instrument} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching candles for {instrument}: {e}")
            return pd.DataFrame()
    
    def get_live_candle(self, instrument: str) -> Optional[Dict]:
        """Get the most recent live candle."""
        df = self.get_candles(instrument, "15m", count=1)
        if not df.empty:
            return df.iloc[-1].to_dict()
        return None
    
    def get_current_price(self, instrument: str) -> Optional[Dict]:
        """Get current bid/ask prices."""
        try:
            params = {"instruments": instrument}
            request = instruments.InstrumentsPricing(accountID=self.account_id, params=params)
            response = self.client.request(request)
            
            if response.get("prices"):
                price = response["prices"][0]
                return {
                    "instrument": price["instrument"],
                    "bid": float(price["bids"][0]["price"]),
                    "ask": float(price["asks"][0]["price"]),
                    "time": price["time"]
                }
        except Exception as e:
            logger.error(f"Error fetching price for {instrument}: {e}")
        return None
    
    # ==================== ACCOUNT INFO ====================
    
    def get_account_info(self) -> Optional[Dict]:
        """Get account information."""
        try:
            request = accounts.AccountDetails(accountID=self.account_id)
            response = self.client.request(request)
            return response["account"]
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            return None
    
    def get_balance(self) -> Optional[float]:
        """Get account balance."""
        account = self.get_account_info()
        if account:
            return float(account.get("balance", 0))
        return None
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        try:
            request = positions.OpenPositions(accountID=self.account_id)
            response = self.client.request(request)
            return response.get("positions", [])
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
    
    # ==================== ORDER EXECUTION ====================
    
    def place_market_order(
        self, 
        instrument: str, 
        units: float, 
        stop_loss: float = None, 
        take_profit: float = None
    ) -> Optional[Dict]:
        """
        Place a market order.
        
        Args:
            instrument: Forex pair
            units: Number of units (positive = buy, negative = sell)
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Order response or None
        """
        order_type = "MARKET"
        
        # Determine order side
        side = "BUY" if units > 0 else "SELL"
        units_abs = abs(int(units))
        
        # Build order data
        order_data = {
            "order": {
                "type": order_type,
                "instrument": instrument,
                "units": units if units > 0 else units,
                "timeInForce": "FOK",
                "positionFillout": "DEFAULT"
            }
        }
        
        # Add stop loss
        if stop_loss:
            order_data["order"]["stopLossOnFill"] = {
                "price": str(stop_loss),
                "timeInForce": "GTC"
            }
        
        # Add take profit
        if take_profit:
            order_data["order"]["takeProfitOnFill"] = {
                "price": str(take_profit),
                "timeInForce": "GTC"
            }
        
        try:
            request = orders.OrderCreate(accountID=self.account_id, data=order_data)
            response = self.client.request(request)
            logger.info(f"Order placed: {side} {units_abs} {instrument} @ market")
            return response
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def place_buy_order(
        self, 
        instrument: str, 
        units: int, 
        stop_loss: float = None, 
        take_profit: float = None
    ) -> Optional[Dict]:
        """Place a buy order."""
        return self.place_market_order(instrument, abs(units), stop_loss, take_profit)
    
    def place_sell_order(
        self, 
        instrument: str, 
        units: int, 
        stop_loss: float = None, 
        take_profit: float = None
    ) -> Optional[Dict]:
        """Place a sell order."""
        return self.place_market_order(instrument, -abs(units), stop_loss, take_profit)
    
    def close_position(self, instrument: str) -> Optional[Dict]:
        """Close an open position for an instrument."""
        try:
            # First get the position details
            positions = self.get_open_positions()
            for pos in positions:
                if pos["instrument"] == instrument and int(pos["long"]["units"]) != 0:
                    # Close long position
                    data = {"longUnits": "ALL"}
                    request = positions.PositionClose(
                        accountID=self.account_id, 
                        instrument=instrument, 
                        data=data
                    )
                    return self.client.request(request)
                elif pos["instrument"] == instrument and int(pos["short"]["units"]) != 0:
                    # Close short position
                    data = {"shortUnits": "ALL"}
                    request = positions.PositionClose(
                        accountID=self.account_id, 
                        instrument=instrument, 
                        data=data
                    )
                    return self.client.request(request)
        except Exception as e:
            logger.error(f"Error closing position: {e}")
        return None
    
    def get_order_book(self, instrument: str, depth: int = 25) -> Optional[Dict]:
        """Get order book for an instrument."""
        try:
            params = {"depth": depth}
            request = instruments.InstrumentsOrderBook(instrument=instrument, params=params)
            response = self.client.request(request)
            return response
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return None
    
    def get_position_book(self, instrument: str) -> Optional[Dict]:
        """Get position book for an instrument (shows liquidity zones)."""
        try:
            request = instruments.InstrumentsPositionBook(instrument=instrument)
            response = self.client.request(request)
            return response
        except Exception as e:
            logger.error(f"Error fetching position book: {e}")
            return None


# Factory function
def create_oanda_client() -> OandaClient:
    """Create and return an OANDA client instance."""
    return OandaClient()
