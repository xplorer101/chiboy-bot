#!/usr/bin/env python3
"""
CHIBOY BOT - Main Entry Point
===================================
A comprehensive trading bot implementing CHIBOY (Inner Circle Trader) concepts
for forex (OANDA) and crypto (Binance) markets.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import time
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    BOT_CONFIG,
    LOGGING_CONFIG,
    TRADING_MODE,
    OANDA_CONFIG,
    BINANCE_CONFIG
)
from src.data.oanda_client import OandaClient
from src.data.binance_client import BinanceClient
from src.analysis.multi_timeframe import MultiTimeframeAnalyzer
from src.analysis.ict_analyzer import CHIBOYAnalyzer
from src.signals.trading_signals import TradingSignals, SignalManager


def setup_logging():
    """Configure logging for the bot."""
    # Create logs directory
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_format = LOGGING_CONFIG["format"]
    date_format = LOGGING_CONFIG["date_format"]
    
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG["level"]),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(LOGGING_CONFIG["log_file"]),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


logger = setup_logging()


class CHIBOYTradingBot:
    """Main trading bot class."""
    
    def __init__(self):
        self.analyzer = MultiTimeframeAnalyzer()
        self.signals = TradingSignals()
        self.signal_manager = SignalManager()
        self.running = False
        
        logger.info(f"CHIBOY BOT v{BOT_CONFIG['name']} v{BOT_CONFIG['version']} initialized")
        logger.info(f"Trading mode: {'DRY RUN' if TRADING_MODE['dry_run'] else 'LIVE'}")
    
    def analyze_single_pair(self, symbol: str, asset_type: str = "forex") -> dict:
        """Analyze a single trading pair."""
        
        logger.info(f"Analyzing {symbol} ({asset_type})...")
        
        if asset_type == "forex":
            result = self.analyzer.analyze_forex_pair(symbol)
        else:
            result = self.analyzer.analyze_crypto_pair(symbol)
        
        return result
    
    def analyze_all_markets(self) -> list:
        """Analyze all configured markets."""
        
        logger.info("Analyzing all markets...")
        results = []
        
        # Analyze forex
        if OANDA_CONFIG["api_key"]:
            logger.info("Analyzing forex pairs...")
            for pair in OANDA_CONFIG["forex_pairs"]:
                try:
                    result = self.analyzer.analyze_forex_pair(pair)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing {pair}: {e}")
        
        # Analyze crypto
        if BINANCE_CONFIG["api_key"]:
            logger.info("Analyzing crypto pairs...")
            for symbol in BINANCE_CONFIG["crypto_pairs"]:
                try:
                    result = self.analyzer.analyze_crypto_pair(symbol)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
        
        return results
    
    def get_top_opportunities(self, limit: int = 5) -> list:
        """Get top trading opportunities."""
        return self.analyzer.get_top_opportunities(limit=limit)
    
    def run_analysis_cycle(self):
        """Run a single analysis cycle."""
        
        logger.info("=" * 60)
        logger.info("Starting analysis cycle...")
        
        # Get top opportunities
        opportunities = self.get_top_opportunities(limit=5)
        
        if opportunities:
            logger.info(f"Found {len(opportunities)} opportunities:")
            
            for i, opp in enumerate(opportunities, 1):
                print(f"\n{'='*50}")
                print(f"#{i} {opp['symbol']} ({opp['type'].upper()})")
                print(f"Direction: {opp['opportunity']['direction'].upper()}")
                print(f"Entry: {opp['opportunity'].get('entry_price', 'Market')}")
                print(f"SL: {opp['opportunity'].get('stop_loss', 'N/A')}")
                print(f"TP: {opp['opportunity'].get('take_profit', 'N/A')}")
                print(f"R:R: {opp['opportunity'].get('risk_reward', 'N/A')}")
                print(f"Confidence: {opp['opportunity']['confidence']:.0f}%")
                print(f"Timeframe: {opp['timeframe']}")
                
                if opp['opportunity'].get('reasons'):
                    print("Reasons:")
                    for reason in opp['opportunity']['reasons']:
                        print(f"  • {reason}")
        else:
            logger.info("No trading opportunities found in this cycle")
        
        return opportunities
    
    def start_live_trading(self, interval_seconds: int = None):
        """Start live trading loop."""
        
        interval = interval_seconds or BOT_CONFIG["run_interval_seconds"]
        
        logger.info(f"Starting live trading with {interval}s interval")
        self.running = True
        
        try:
            while self.running:
                try:
                    opportunities = self.run_analysis_cycle()
                    
                    # Execute trades if enabled
                    if TRADING_MODE["execute_trades"] and opportunities:
                        for opp in opportunities[:RISK_CONFIG["max_open_trades"]]:
                            signal_data = opp["opportunity"]
                            
                            # Create trade signal
                            signal = self.signals.create_signal(
                                symbol=opp["symbol"],
                                asset_type=opp["type"],
                                direction=signal_data["direction"],
                                entry_price=signal_data.get("entry_price", 0) or 0,
                                stop_loss=signal_data.get("stop_loss", 0) or 0,
                                take_profit=signal_data.get("take_profit", 0) or 0,
                                confidence=signal_data["confidence"],
                                timeframe=opp["timeframe"],
                                reasons=signal_data.get("reasons", [])
                            )
                            
                            self.signal_manager.add_signal(signal)
                        
                        # Execute pending signals
                        results = self.signal_manager.execute_pending_signals()
                        
                        for result in results:
                            if result.get("success"):
                                logger.info(f"Trade executed: {result['signal'].symbol}")
                
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}")
                
                # Wait for next cycle
                logger.debug(f"Waiting {interval}s until next cycle...")
                time.sleep(interval)
        
        except KeyboardInterrupt:
            logger.info("Trading stopped by user")
            self.running = False
    
    def stop(self):
        """Stop the trading bot."""
        logger.info("Stopping trading bot...")
        self.running = False


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="CHIBOY BOT - Forex & Crypto"
    )
    
    parser.add_argument(
        "--analyze",
        type=str,
        help="Analyze a single symbol (e.g., EUR_USD or BTCUSDT)"
    )
    
    parser.add_argument(
        "--type",
        type=str,
        choices=["forex", "crypto"],
        default="forex",
        help="Asset type for analysis"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all configured markets"
    )
    
    parser.add_argument(
        "--opportunities",
        type=int,
        default=5,
        help="Number of top opportunities to show"
    )
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="Start live trading mode"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Interval between live trading cycles (seconds)"
    )
    
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Enable trade execution (requires API keys)"
    )
    
    args = parser.parse_args()
    
    # Create bot instance
    bot = CHIBOYTradingBot()
    
    # Handle different modes
    if args.analyze:
        # Single symbol analysis
        result = bot.analyze_single_pair(args.analyze, args.type)
        report = bot.analyzer.format_analysis_report(result)
        print(report)
    
    elif args.all:
        # Analyze all markets
        results = bot.analyze_all_markets()
        
        print("\n" + "="*60)
        print("ALL MARKETS ANALYSIS COMPLETE")
        print("="*60)
        
        for result in results:
            if result.get("best_opportunity"):
                opp = result["best_opportunity"]["opportunity"]
                print(f"\n{result['symbol']}: {opp['direction'].upper()} "
                      f"(Confidence: {opp['confidence']:.0f}%)")
    
    elif args.live:
        # Live trading mode
        if args.execute:
            os.environ["EXECUTE_TRADES"] = "true"
            logger.info("⚠️ TRADE EXECUTION ENABLED ⚠️")
        
        bot.start_live_trading(args.interval)
    
    else:
        # Default: show top opportunities
        opportunities = bot.get_top_opportunities(args.opportunities)
        
        if opportunities:
            print("\n" + "="*60)
            print("TOP TRADING OPPORTUNITIES")
            print("="*60)
            
            for i, opp in enumerate(opportunities, 1):
                print(f"\n#{i} {opp['symbol']} ({opp['type'].upper()})")
                print(f"    Direction:    {opp['opportunity']['direction'].upper()}")
                print(f"    Entry:        {opp['opportunity'].get('entry_price', 'Market')}")
                print(f"    Stop Loss:    {opp['opportunity'].get('stop_loss', 'N/A')}")
                print(f"    Take Profit:  {opp['opportunity'].get('take_profit', 'N/A')}")
                print(f"    Risk:Reward:  {opp['opportunity'].get('risk_reward', 'N/A')}")
                print(f"    Confidence:   {opp['opportunity']['confidence']:.0f}%")
                print(f"    Timeframe:    {opp['timeframe']}")
                
                if opp['opportunity'].get('reasons'):
                    print("    Reasons:")
                    for reason in opp['opportunity']['reasons']:
                        print(f"      • {reason}")
        else:
            print("No trading opportunities found.")
            print("Run with --all to analyze all markets first.")
    
    print("\n")


if __name__ == "__main__":
    main()
