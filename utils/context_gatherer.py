"""
Context Gathering Utility for LLM Processing.

This module collects relevant market context and historical data for better LLM analysis of crypto alerts.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

import aiohttp

from data.db import Database
from data.models import Alert
from logger_config import setup_logging

logger = setup_logging("CONTEXT_GATHERER", "yellow")


class ContextGatherer:
    """Gathers relevant context for LLM to evaluate crypto alerts."""

    def __init__(self):
        self.db = Database()

    async def get_alert_context(self, symbol: str, alert_type: str) -> Dict[str, Any]:
        """
        Gather all relevant context for a given alert.

        Args:
            symbol: The trading pair symbol (e.g., 'BTC/USDT')
            alert_type: The type of alert being processed

        Returns:
            Dict containing all contextual information for LLM analysis
        """
        try:
            # Get historical alerts (previous implementation)
            historical_alerts = await self.get_historical_alerts(symbol, alert_type)

            # Get broader market context (previous implementation)
            market_context = await self.get_market_context()

            # Get symbol-specific details (previous implementation)
            symbol_details = await self.get_symbol_details(symbol.split("/")[0])

            # NEW: Get large order context if the analyzer is available
            large_order_context = {}
            if hasattr(self, "analyzer") and self.analyzer:
                large_order_context = await self.analyzer.get_recent_large_orders(symbol)

            context = {
                "historical_alerts": historical_alerts,
                "market_context": market_context,
                "symbol_details": symbol_details,
                "large_order_context": large_order_context,
            }

            return context

        except Exception as e:
            logger.error(f"Error getting alert context: {e}")
            return {
                "historical_alerts": [],
                "market_context": {},
                "symbol_details": {},
                "large_order_context": {},
            }

    async def get_historical_alerts(self, symbol: str, alert_type: str) -> List[Dict[str, Any]]:
        """Get historical alerts for the given symbol and alert type."""
        try:
            # Query past 7 days of alerts for this symbol
            seven_days_ago = datetime.now() - timedelta(days=7)

            # Query only columns that are guaranteed to exist
            alerts = (
                self.db.session.query(
                    Alert.id,
                    Alert.symbol,
                    Alert.alert_type,
                    Alert.timestamp,
                    Alert.last_alerted_at,
                    Alert.llm_sent,
                    Alert.llm_reasoning,
                )
                .filter(Alert.symbol == symbol)
                .filter(Alert.timestamp >= seven_days_ago)
                .order_by(Alert.timestamp.desc())
                .all()
            )

            # Format alerts into a list of dictionaries
            return [
                {
                    "alert_type": alert.alert_type,
                    "timestamp": alert.timestamp.isoformat(),
                    "sent": getattr(
                        alert, "llm_sent", False
                    ),  # Use getattr to handle missing attributes
                    "result": "Sent" if getattr(alert, "llm_sent", False) else "Not sent",
                    "reasoning": getattr(alert, "llm_reasoning", ""),
                }
                for alert in alerts
            ]
        except Exception as e:
            logger.error(f"Error fetching historical alerts: {e}")
            # Ensure we rollback on error
            self.db.session.rollback()
            return []

    async def get_market_context(self) -> Dict[str, Any]:
        """Get broader market context data."""
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch Fear & Greed Index
                async with session.get("https://api.alternative.me/fng/") as response:
                    if response.status == 200:
                        fear_greed_data = await response.json()
                        fear_greed_value = fear_greed_data.get("data", [{}])[0].get("value", "N/A")
                    else:
                        fear_greed_value = "N/A"

                # Get BTC dominance and market direction data
                # For demo purposes using a placeholder endpoint
                async with session.get("https://api.coingecko.com/api/v3/global") as response:
                    if response.status == 200:
                        global_data = await response.json()
                        market_data = global_data.get("data", {})
                        btc_dominance = market_data.get("bitcoin_dominance", "N/A")
                        total_market_cap = market_data.get("total_market_cap", {}).get("usd", 0)
                        market_cap_change = market_data.get(
                            "market_cap_change_percentage_24h_usd", "N/A"
                        )
                        market_direction = (
                            "UP" if market_cap_change and float(market_cap_change) > 0 else "DOWN"
                        )
                    else:
                        btc_dominance = "N/A"
                        market_direction = "N/A"
                        market_cap_change = "N/A"

            return {
                "btc_dominance": btc_dominance,
                "fear_greed_index": fear_greed_value,
                "market_direction": market_direction,
                "market_cap_change": market_cap_change,
            }

        except Exception as e:
            logger.error(f"Error fetching market context: {e}")
            return {
                "btc_dominance": "N/A",
                "fear_greed_index": "N/A",
                "market_direction": "N/A",
                "market_cap_change": "N/A",
            }

    async def get_symbol_details(self, symbol_base: str) -> Dict[str, Any]:
        """Get detailed information about the specific symbol."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://api.coingecko.com/api/v3/coins/{symbol_base.lower()}"
                ) as response:
                    if response.status == 200:
                        coin_data = await response.json()

                        return {
                            "name": coin_data.get("name", "N/A"),
                            "market_cap_rank": coin_data.get("market_cap_rank", "N/A"),
                            "price_change_24h": coin_data.get("market_data", {}).get(
                                "price_change_percentage_24h", "N/A"
                            ),
                            "price_change_7d": coin_data.get("market_data", {}).get(
                                "price_change_percentage_7d", "N/A"
                            ),
                            "volume_24h": coin_data.get("market_data", {})
                            .get("total_volume", {})
                            .get("usd", "N/A"),
                        }
                    else:
                        return {
                            "name": symbol_base,
                            "market_cap_rank": "N/A",
                            "price_change_24h": "N/A",
                            "price_change_7d": "N/A",
                            "volume_24h": "N/A",
                        }

        except Exception as e:
            logger.error(f"Error fetching symbol details for {symbol_base}: {e}")
            return {
                "name": symbol_base,
                "market_cap_rank": "N/A",
                "price_change_24h": "N/A",
                "price_change_7d": "N/A",
                "volume_24h": "N/A",
            }
