"""
Trading Signal Generator & Trade Executor
=========================================
Generates actionable trading signals and executes trades
on OANDA (Forex) and Binance (Crypto).
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass

from ..config import RISK_CONFIG, TRADING_MODE, OANDA_CONFIG, BINANCE_CONFIG
from ..data.oanda_client import OandaClient
from ..data.binance_client import BinanceClient

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """Represents a trading signal."""
    symbol: str
    asset_type: str  # 'forex' or 'crypto'
    direction: str    # 'long' or 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    confidence: float
    timeframe: str
    reasons: List[str]
    timestamp: datetime


class TradingSignals:
    """Generates and manages trading signals."""
    
    def __init__(self):
        self.oanda = OandaClient() if OANDA_CONFIG["api_key"] else None
        self.binance = BinanceClient() if BINANCE_CONFIG["api_key"] else None
        
    def create_signal(
        self,
        symbol: str,
        asset_type: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        timeframe: str,
        reasons: List[str]
    ) -> TradeSignal:
        """Create a trade signal."""
        # Calculate risk-reward
        if direction == "long":
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        
        risk_reward = reward / risk if risk > 0 else 0
        
        return TradeSignal(
            symbol=symbol,
            asset_type=asset_type,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
            confidence=confidence,
            timeframe=timeframe,
            reasons=reasons,
            timestamp=datetime.now()
        )
    
    def validate_signal(self, signal: TradeSignal) -> Dict:
        """Validate a trading signal."""
        validation = {
            "valid": True,
            "errors": []
        }
        
        # Check risk-reward
        if signal.risk_reward < RISK_CONFIG["min_risk_reward"]:
            validation["valid"] = False
            validation["errors"].append(
                f"Risk:Reward {signal.risk_reward:.2f} below minimum {RISK_CONFIG['min_risk_reward']}"
            )
        
        # Check confidence
        if signal.confidence < 50:
            validation["valid"] = False
            validation["errors"].append(
                f"Confidence {signal.confidence:.0f}% below minimum 50%"
            )
        
        # Check stop loss distance
        if signal.direction == "long":
            sl_distance = signal.entry_price - signal.stop_loss
        else:
            sl_distance = signal.stop_loss - signal.entry_price
        
        # For forex, ensure SL is at least 10 pips
        if signal.asset_type == "forex":
            if signal.direction == "long":
                pips = (signal.entry_price - signal.stop_loss) * 10000
            else:
                pips = (signal.stop_loss - signal.entry_price) * 10000
            
            if pips < 10:
                validation["valid"] = False
                validation["errors"].append(f"Stop loss {pips:.1f} pips too tight (min 10)")
        
        return validation
    
    def calculate_position_size(
        self,
        signal: TradeSignal,
        account_balance: float
    ) -> float:
        """Calculate position size based on risk management."""
        
        # Calculate risk amount
        risk_amount = account_balance * RISK_CONFIG["max_risk_per_trade"]
        
        # Calculate stop loss distance
        if signal.direction == "long":
            sl_distance = signal.entry_price - signal.stop_loss
        else:
            sl_distance = signal.stop_loss - signal.entry_price
        
        # Calculate position size
        if sl_distance > 0:
            position_size = risk_amount / sl_distance
        else:
            position_size = 0
        
        return position_size
    
    def execute_forex_trade(
        self,
        signal: TradeSignal,
        position_size: float
    ) -> Dict:
        """Execute a forex trade on OANDA."""
        
        if not self.oanda:
            logger.error("OANDA client not initialized")
            return {"success": False, "error": "OANDA not configured"}
        
        if TRADING_MODE["dry_run"]:
            logger.info(f"[DRY RUN] Would execute: {signal.direction.upper()} {position_size:.0f} {signal.symbol}")
            return {"success": True, "dry_run": True, "signal": signal}
        
        try:
            # Map pair format (EUR_USD -> EUR_USD)
            instrument = signal.symbol
            
            # Determine units (OANDA uses units for forex)
            units = int(position_size)
            
            if signal.direction == "long":
                result = self.oanda.place_buy_order(
                    instrument=instrument,
                    units=units,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit
                )
            else:
                result = self.oanda.place_sell_order(
                    instrument=instrument,
                    units=units,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit
                )
            
            if result:
                logger.info(f"Trade executed: {signal.direction.upper()} {units} {instrument}")
                return {"success": True, "result": result}
            else:
                return {"success": False, "error": "Order placement failed"}
                
        except Exception as e:
            logger.error(f"Error executing forex trade: {e}")
            return {"success": False, "error": str(e)}
    
    def execute_crypto_trade(
        self,
        signal: TradeSignal,
        quantity: float
    ) -> Dict:
        """Execute a crypto trade on Binance."""
        
        if not self.binance:
            logger.error("Binance client not initialized")
            return {"success": False, "error": "Binance not configured"}
        
        if TRADING_MODE["dry_run"]:
            logger.info(f"[DRY RUN] Would execute: {signal.direction.upper()} {quantity:.4f} {signal.symbol}")
            return {"success": True, "dry_run": True, "signal": signal}
        
        try:
            # Ensure symbol format (e.g., BTCUSDT)
            symbol = signal.symbol
            if not symbol.endswith("USDT"):
                symbol = f"{symbol}USDT"
            
            result = self.binance.place_market_order(
                symbol=symbol,
                side=signal.direction.upper(),
                quantity=quantity,
                test=True  # Always test first unless explicitly enabled
            )
            
            if result:
                logger.info(f"Trade executed: {signal.direction.upper()} {quantity} {symbol}")
                return {"success": True, "result": result}
            else:
                return {"success": False, "error": "Order placement failed"}
                
        except Exception as e:
            logger.error(f"Error executing crypto trade: {e}")
            return {"success": False, "error": str(e)}
    
    def execute_signal(
        self,
        signal: TradeSignal,
        account_balance: float = None
    ) -> Dict:
        """Execute a trading signal."""
        
        # Validate signal first
        validation = self.validate_signal(signal)
        
        if not validation["valid"]:
            logger.warning(f"Signal validation failed: {validation['errors']}")
            return {"success": False, "errors": validation["errors"]}
        
        # Get account balance if not provided
        if not account_balance:
            if signal.asset_type == "forex" and self.oanda:
                account_balance = self.oanda.get_balance() or 10000
            elif signal.asset_type == "crypto" and self.binance:
                account_balance = self.binance.get_balance() or 10000
            else:
                account_balance = 10000  # Default for testing
        
        # Calculate position size
        if signal.asset_type == "forex":
            position_size = self.calculate_position_size(signal, account_balance)
            return self.execute_forex_trade(signal, position_size)
        else:
            # For crypto, we need to calculate quantity
            quantity = (account_balance * RISK_CONFIG["max_risk_per_trade"]) / signal.entry_price
            return self.execute_crypto_trade(signal, quantity)
    
    def format_signal_report(self, signal: TradeSignal) -> str:
        """Format a trading signal into a readable report."""
        
        lines = []
        lines.append("=" * 50)
        lines.append(f"📊 TRADING SIGNAL: {signal.symbol}")
        lines.append("=" * 50)
        lines.append(f"  Type:        {signal.asset_type.upper()}")
        lines.append(f"  Direction:   {signal.direction.upper()}")
        lines.append(f"  Entry:       {signal.entry_price:.5f}")
        lines.append(f"  Stop Loss:   {signal.stop_loss:.5f}")
        lines.append(f"  Take Profit: {signal.take_profit:.5f}")
        lines.append(f"  Risk:Reward: 1:{signal.risk_reward:.2f}")
        lines.append(f"  Confidence:  {signal.confidence:.0f}%")
        lines.append(f"  Timeframe:   {signal.timeframe}")
        lines.append(f"  Time:        {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if signal.reasons:
            lines.append("")
            lines.append("  Reasons:")
            for reason in signal.reasons:
                lines.append(f"    • {reason}")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)


class SignalManager:
    """Manages multiple trading signals and tracks open positions."""
    
    def __init__(self):
        self.signals: List[TradeSignal] = []
        self.executed_trades: List[Dict] = []
        self.trading = TradingSignals()
        
    def add_signal(self, signal: TradeSignal):
        """Add a new signal to the queue."""
        self.signals.append(signal)
        logger.info(f"Signal added: {signal.symbol} {signal.direction} ({signal.confidence:.0f}%)")
    
    def execute_pending_signals(self, max_trades: int = None) -> List[Dict]:
        """Execute all pending signals that meet criteria."""
        
        max_trades = max_trades or RISK_CONFIG["max_open_trades"]
        
        results = []
        
        # Sort by confidence
        sorted_signals = sorted(
            self.signals,
            key=lambda s: s.confidence,
            reverse=True
        )
        
        for signal in sorted_signals[:max_trades]:
            # Validate signal
            validation = self.trading.validate_signal(signal)
            
            if validation["valid"]:
                result = self.trading.execute_signal(signal)
                result["signal"] = signal
                results.append(result)
                
                # Remove from pending
                if signal in self.signals:
                    self.signals.remove(signal)
            else:
                logger.warning(f"Signal validation failed for {signal.symbol}: {validation['errors']}")
        
        return results
    
    def get_pending_signals(self) -> List[TradeSignal]:
        """Get all pending signals."""
        return self.signals
    
    def clear_signals(self):
        """Clear all pending signals."""
        self.signals.clear()
        logger.info("All pending signals cleared")
