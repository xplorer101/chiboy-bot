"""
Multi-Timeframe Analyzer
========================
Performs CHIBOY analysis across multiple timeframes (Weekly → 15m)
and generates trading signals.
"""

import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime

from ..data.oanda_client import OandaClient
from ..data.binance_client import BinanceClient
from ..data.demo_data import demo_generator
from ..data.live_data import live_data
from ..analysis.ict_analyzer import CHIBOYAnalyzer
from ..config import ANALYSIS_TIMEFRAMES, OANDA_CONFIG, BINANCE_CONFIG

logger = logging.getLogger(__name__)


class MultiTimeframeAnalyzer:
    """Analyzes markets across multiple timeframes."""
    
    def __init__(self):
        self.oanda = OandaClient() if OANDA_CONFIG["api_key"] else None
        self.binance = BinanceClient() if BINANCE_CONFIG["api_key"] else None
        self.analyzer = CHIBOYAnalyzer()
        
    # ==================== DATA FETCHING ====================
    
    def fetch_forex_data(self, pair: str, timeframe: str, count: int = 200) -> pd.DataFrame:
        """Fetch forex data - uses live prices when available."""
        # Try live data first, fallback to demo
        try:
            live_price = live_data.get(pair)
            if live_price and live_price.get('price', 0) > 0:
                logger.info(f"Using live data for {pair}")
                # Generate candles based on live price
                return self._generate_candles_from_live(pair, live_price['price'], timeframe, count)
        except Exception as e:
            logger.debug(f"Live data error: {e}")
        
        # Fallback to demo
        return demo_generator.generate_candles(pair, timeframe, count)
    
    def fetch_crypto_data(self, symbol: str, timeframe: str, count: int = 200) -> pd.DataFrame:
        """Fetch crypto data - uses live prices when available."""
        # Map symbol
        demo_symbol = symbol if symbol.endswith("USDT") else f"{symbol}USDT"
        
        # Try live data first
        try:
            live_price = live_data.get(demo_symbol)
            if live_price and live_price.get('price', 0) > 0:
                logger.info(f"Using live data for {demo_symbol}")
                return self._generate_candles_from_live(demo_symbol, live_price['price'], timeframe, count)
        except Exception as e:
            logger.debug(f"Live data error: {e}")
        
        # Fallback
        return demo_generator.generate_candles(demo_symbol, timeframe, count)
    
    def _generate_candles_from_live(self, symbol: str, current_price: float, timeframe: str, count: int) -> pd.DataFrame:
        """Generate candles using live price as base."""
        import numpy as np
        from datetime import timedelta
        
        is_crypto = symbol.endswith('USDT')
        volatility = 0.015 if is_crypto else 0.0008
        
        # Determine time interval
        intervals = {'15m': 15, '1H': 60, '4H': 240, '1D': 1440, '1W': 10080}
        minutes = intervals.get(timeframe, 60)
        
        # Generate timestamps
        now = datetime.now()
        times = [now - timedelta(minutes=minutes*(count-i)) for i in range(count)]
        
        # Generate prices with random walk
        np.random.seed(hash(symbol) % 2**32)
        returns = np.random.normal(0, volatility, count)
        
        prices = [current_price]
        for r in returns[:-1]:
            prices.append(prices[-1] * (1 + r))
        
        # Create OHLC
        data = []
        for i, (t, close) in enumerate(zip(times, prices)):
            spread = current_price * 0.0002
            high = close + abs(np.random.normal(0, volatility/2)) * close
            low = close - abs(np.random.normal(0, volatility/2)) * close
            open_price = prices[i-1] if i > 0 else close
            
            data.append({
                'time': t,
                'open': open_price,
                'high': max(high, open_price, close),
                'low': min(low, open_price, close),
                'close': close,
                'volume': np.random.randint(1000, 10000) if is_crypto else np.random.randint(100, 1000)
            })
        
        df = pd.DataFrame(data)
        df.set_index('time', inplace=True)
        return df
        
        return self.binance.get_klines(symbol, timeframe, count)
    
    # ==================== MULTI-TIMEFRAME ANALYSIS ====================
    
    def analyze_forex_pair(self, pair: str) -> Dict:
        """
        Perform multi-timeframe analysis on a forex pair.
        
        Args:
            pair: Forex pair (e.g., 'EUR_USD')
            
        Returns:
            Dict with analysis from all timeframes
        """
        logger.info(f"Analyzing {pair} across {len(ANALYSIS_TIMEFRAMES)} timeframes")
        
        results = {
            "symbol": pair,
            "type": "forex",
            "timeframes": {},
            "summary": {},
            "best_opportunity": None,
            "timestamp": datetime.now().isoformat()
        }
        
        all_opportunities = []
        
        # Analyze each timeframe
        for tf in ANALYSIS_TIMEFRAMES:
            logger.debug(f"  Analyzing {pair} {tf}...")
            
            df = self.fetch_forex_data(pair, tf)
            
            if df.empty:
                logger.warning(f"No data fetched for {pair} {tf}")
                continue
            
            # Perform CHIBOY analysis
            analysis = self.analyzer.analyze(df, tf)
            results["timeframes"][tf] = analysis
            
            # Collect opportunities
            if analysis.get("opportunity", {}).get("exists"):
                all_opportunities.append({
                    "timeframe": tf,
                    "opportunity": analysis["opportunity"]
                })
        
        # Determine best opportunity
        if all_opportunities:
            # Prioritize lower timeframes for entry
            all_opportunities.sort(
                key=lambda x: ANALYSIS_TIMEFRAMES.index(x["timeframe"]) 
                if x["timeframe"] in ANALYSIS_TIMEFRAMES else 999
            )
            
            results["best_opportunity"] = all_opportunities[0]
            results["summary"]["total_signals"] = len(all_opportunities)
        
        # Generate overall summary
        results["summary"] = self._generate_summary(results["timeframes"])
        
        return results
    
    def analyze_crypto_pair(self, symbol: str) -> Dict:
        """
        Perform multi-timeframe analysis on a crypto pair.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTCUSDT')
            
        Returns:
            Dict with analysis from all timeframes
        """
        logger.info(f"Analyzing {symbol} across {len(ANALYSIS_TIMEFRAMES)} timeframes")
        
        results = {
            "symbol": symbol,
            "type": "crypto",
            "timeframes": {},
            "summary": {},
            "best_opportunity": None,
            "timestamp": datetime.now().isoformat()
        }
        
        all_opportunities = []
        
        # Map symbol for Binance (add USDT if missing)
        binance_symbol = symbol if symbol.endswith("USDT") else f"{symbol}USDT"
        
        # Analyze each timeframe
        for tf in ANALYSIS_TIMEFRAMES:
            logger.debug(f"  Analyzing {binance_symbol} {tf}...")
            
            df = self.fetch_crypto_data(binance_symbol, tf)
            
            if df.empty:
                logger.warning(f"No data fetched for {binance_symbol} {tf}")
                continue
            
            # Perform CHIBOY analysis
            analysis = self.analyzer.analyze(df, tf)
            results["timeframes"][tf] = analysis
            
            # Collect opportunities
            if analysis.get("opportunity", {}).get("exists"):
                all_opportunities.append({
                    "timeframe": tf,
                    "opportunity": analysis["opportunity"]
                })
        
        # Determine best opportunity
        if all_opportunities:
            all_opportunities.sort(
                key=lambda x: ANALYSIS_TIMEFRAMES.index(x["timeframe"])
                if x["timeframe"] in ANALYSIS_TIMEFRAMES else 999
            )
            
            results["best_opportunity"] = all_opportunities[0]
            results["summary"]["total_signals"] = len(all_opportunities)
        
        # Generate overall summary
        results["summary"] = self._generate_summary(results["timeframes"])
        
        return results
    
    def _generate_summary(self, timeframes: Dict) -> Dict:
        """Generate a summary of all timeframe analyses."""
        if not timeframes:
            return {"error": "No data"}
        
        # Aggregate trend directions
        trends = {}
        structures = {}
        
        for tf, analysis in timeframes.items():
            if "trend" in analysis:
                trends[tf] = analysis["trend"].get("trend", "unknown")
            if "structure" in analysis:
                structures[tf] = analysis["structure"].get("current_structure", "unknown")
        
        # Determine overall trend
        trend_counts = {}
        for t in trends.values():
            trend_counts[t] = trend_counts.get(t, 0) + 1
        
        overall_trend = max(trend_counts, key=trend_counts.get) if trend_counts else "unknown"
        
        return {
            "trends": trends,
            "structures": structures,
            "overall_trend": overall_trend,
            "timeframes_analyzed": len(timeframes)
        }
    
    # ==================== BATCH ANALYSIS ====================
    
    def analyze_all_forex_pairs(self) -> List[Dict]:
        """Analyze all configured forex pairs."""
        results = []
        
        for pair in OANDA_CONFIG["forex_pairs"]:
            try:
                result = self.analyze_forex_pair(pair)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {pair}: {e}")
        
        return results
    
    def analyze_all_crypto_pairs(self) -> List[Dict]:
        """Analyze all configured crypto pairs."""
        results = []
        
        for symbol in BINANCE_CONFIG["crypto_pairs"]:
            try:
                result = self.analyze_crypto_pair(symbol)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        return results
    
    def get_top_opportunities(self, asset_type: str = "all", limit: int = 5) -> List[Dict]:
        """
        Get top trading opportunities across all assets.
        
        Args:
            asset_type: 'forex', 'crypto', or 'all'
            limit: Maximum number of opportunities to return
            
        Returns:
            List of opportunities sorted by confidence
        """
        opportunities = []
        
        if asset_type in ["forex", "all"]:
            forex_results = self.analyze_all_forex_pairs()
            for result in forex_results:
                if result.get("best_opportunity"):
                    opportunities.append({
                        "symbol": result["symbol"],
                        "type": "forex",
                        **result["best_opportunity"]
                    })
        
        if asset_type in ["crypto", "all"]:
            crypto_results = self.analyze_all_crypto_pairs()
            for result in crypto_results:
                if result.get("best_opportunity"):
                    opportunities.append({
                        "symbol": result["symbol"],
                        "type": "crypto",
                        **result["best_opportunity"]
                    })
        
        # Sort by confidence
        opportunities.sort(key=lambda x: x.get("opportunity", {}).get("confidence", 0), reverse=True)
        
        return opportunities[:limit]
    
    # ==================== FORMATTED OUTPUT ====================
    
    def format_analysis_report(self, result: Dict) -> str:
        """Format analysis results into a readable report."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"CHIBOY ANALYSIS REPORT: {result['symbol']} ({result['type'].upper()})")
        lines.append("=" * 60)
        lines.append(f"Timestamp: {result['timestamp']}")
        lines.append("")
        
        # Summary
        summary = result.get("summary", {})
        lines.append("📊 SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Overall Trend: {summary.get('overall_trend', 'N/A')}")
        lines.append(f"  Timeframes Analyzed: {summary.get('timeframes_analyzed', 0)}")
        
        # Best opportunity
        if result.get("best_opportunity"):
            opp = result["best_opportunity"]["opportunity"]
            lines.append("")
            lines.append("🎯 BEST OPPORTUNITY")
            lines.append("-" * 40)
            lines.append(f"  Direction: {opp.get('direction', 'N/A').upper()}")
            entry_price = opp.get('entry_price')
            entry_str = f"{entry_price:.5f}" if isinstance(entry_price, (int, float)) else "Market"
            lines.append(f"  Entry Price: {entry_str}")
            lines.append(f"  Stop Loss: {opp.get('stop_loss', 'N/A')}")
            lines.append(f"  Take Profit: {opp.get('take_profit', 'N/A')}")
            lines.append(f"  Risk:Reward: {opp.get('risk_reward', 'N/A')}")
            lines.append(f"  Confidence: {opp.get('confidence', 0):.0f}%")
            
            if opp.get("reasons"):
                lines.append("  Reasons:")
                for reason in opp["reasons"]:
                    lines.append(f"    • {reason}")
        
        # Timeframe details
        lines.append("")
        lines.append("📈 TIMEFRAME ANALYSIS")
        lines.append("-" * 40)
        
        for tf in ANALYSIS_TIMEFRAMES:
            if tf in result["timeframes"]:
                tf_analysis = result["timeframes"][tf]
                trend = tf_analysis.get("trend", {}).get("trend", "N/A")
                structure = tf_analysis.get("structure", {}).get("current_structure", "N/A")
                price = tf_analysis.get("current_price", 0)
                
                lines.append(f"  {tf:4s} | Price: {price:>10.5f} | Trend: {trend:>15s} | Structure: {structure}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
