import json
import os
from datetime import datetime
from typing import Any, Dict, List

from openai import OpenAI

from logger_config import setup_logging

logger = setup_logging("LLM_SERVICE", "cyan")


class LLMService:
    """Service for interacting with LLM APIs for crypto alert evaluation."""

    def __init__(self):
        self.api_key = os.getenv("GROK_API_KEY")

        # Add the base URL for X.AI (Grok)
        self.base_url = "https://api.x.ai/v1"

        # Initialize the client with both API key and base URL
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.model = os.getenv("LLM_MODEL", "grok-2-latest")

    async def evaluate_alert(
        self,
        current_alert: Dict[str, Any],
        historical_alerts: List[Dict[str, Any]],
        market_context: Dict[str, Any],
        symbol_details: Dict[str, Any],
        large_order_context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Send alert data to LLM for evaluation.

        Args:
            current_alert: Dictionary with current alert information
            historical_alerts: List of recent alerts for the same symbol
            market_context: Broader market information
            symbol_details: Symbol-specific details
            large_order_context: Information about recent large orders

        Returns:
            Dictionary with LLM evaluation results
        """
        try:
            # Construct the prompt
            prompt = self._construct_evaluation_prompt(
                current_alert,
                historical_alerts,
                market_context,
                symbol_details,
                large_order_context,
            )

            # Make the API call
            response = await self._call_llm_api(prompt)

            # Parse and return the response
            return self._parse_llm_response(response, current_alert)

        except Exception as e:
            logger.error(f"Error in LLM evaluation: {str(e)}")
            # Default to sending the alert if there's an error
            return {
                "should_send": True,
                "analysis": f"Unable to complete LLM analysis due to error: {str(e)}",
                "reasoning": "Error occurred during evaluation, defaulting to sending alert",
                "market_context": "",
                "trading_recommendation": "",
            }

    async def _call_llm_api(self, prompt: str) -> str:
        """Call the LLM API with the constructed prompt."""

        messages = [
            {
                "role": "system",
                "content": "You are an expert cryptocurrency market analyst and technical trader. Your job is to analyze trading signals and determine which ones are worth alerting traders about.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.2, max_tokens=2000
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM API call failed: {str(e)}")
            raise

    def _construct_evaluation_prompt(
        self,
        current_alert: Dict[str, Any],
        historical_alerts: List[Dict[str, Any]],
        market_context: Dict[str, Any],
        symbol_details: Dict[str, Any],
        large_order_context: Dict[str, Any] = None,
    ) -> str:
        """Construct a prompt for the LLM to evaluate the alert."""
        # Format the current alert details
        symbol = current_alert.get("symbol", "Unknown")
        alert_type = current_alert.get("alert_type", "Unknown")
        price = current_alert.get("price", 0)
        timestamp = current_alert.get("timestamp", "Unknown")

        # Get indicator data
        indicators = current_alert.get("indicators", {})

        # Calculate price trend direction (last 3 days)
        price_direction = "neutral"
        recent_price_change = indicators.get("price_change_3d", 0)
        if recent_price_change > 2:
            price_direction = "strongly bullish"
        elif recent_price_change > 0.5:
            price_direction = "bullish"
        elif recent_price_change < -2:
            price_direction = "strongly bearish"
        elif recent_price_change < -0.5:
            price_direction = "bearish"

        # Calculate if the asset is moving with or against the market
        market_trend = market_context.get("market_direction", "neutral")
        market_alignment = "aligned with market"
        if ("bullish" in price_direction and "bearish" in market_trend) or (
            "bearish" in price_direction and "bullish" in market_trend
        ):
            market_alignment = "moving against market trend"

        # Create a concise alert details section with trend context
        alert_details = f"""
        ## Current Alert: {symbol}

        - **Signal**: {alert_type} at ${price:.4f}
        - **Time**: {timestamp}
        - **Price Trend**: {price_direction} ({recent_price_change:.2f}% over 3 days)
        - **Market Alignment**: {market_alignment}
        """

        # Add volume context for volume-based alerts
        if "RVOL" in alert_type:
            volume_ratio = indicators.get("volume_ratio", 0)
            if "bearish" in price_direction and volume_ratio > 2:
                alert_details += f"- **Volume Context**: {volume_ratio:.2f}x average volume on downtrend (likely distribution)\n"
            elif "bullish" in price_direction and volume_ratio > 2:
                alert_details += f"- **Volume Context**: {volume_ratio:.2f}x average volume on uptrend (likely accumulation)\n"
            else:
                alert_details += f"- **Volume Context**: {volume_ratio:.2f}x average volume\n"

        # Add relevant technical indicators
        if indicators:
            alert_details += f"""
        - **Technical Data**:
        """
            if "volume_ratio" in indicators:
                alert_details += (
                    f"      - Volume: {indicators.get('volume_ratio', 0):.2f}x average\n"
                )
            if "rsi" in indicators:
                alert_details += f"      - RSI: {indicators.get('rsi', 0):.2f}\n"
            if "macd" in indicators and "signal" in indicators:
                alert_details += f"      - MACD: {indicators.get('macd', 0):.4f}, Signal: {indicators.get('signal', 0):.4f}\n"
            if "histogram" in indicators:
                alert_details += f"      - Histogram: {indicators.get('histogram', 0):.4f}\n"
            if "price_change_24h" in indicators:
                alert_details += (
                    f"      - 24h Change: {indicators.get('price_change_24h', 0):.2f}%\n"
                )

        # Format historical alerts
        history = f"\n## Recent Alerts for {symbol}\n\n"
        if historical_alerts:
            for idx, alert in enumerate(historical_alerts[:3]):  # Limit to 3 most recent
                history += f"- {alert.get('alert_type')}: ${alert.get('price', 0):.4f} ({alert.get('time_ago', 'unknown')} ago)\n"
        else:
            history += "No recent alerts for this symbol.\n"

        # Format market context
        market_info = f"\n## Market Context\n\n"
        btc_dominance = market_context.get("btc_dominance", "N/A")
        market_direction = market_context.get("market_direction", "N/A")
        market_cap_change = market_context.get("market_cap_change", "N/A")

        market_info += f"- BTC Dominance: {btc_dominance}\n"
        market_info += f"- Market Direction (24h): {market_direction}\n"
        market_info += f"- Market Cap Change (24h): {market_cap_change}\n"

        # Format symbol details
        symbol_info = f"\n## {symbol} Details\n\n"
        if symbol_details:
            for key, value in symbol_details.items():
                if key not in ["id", "image", "description"]:  # Skip verbose fields
                    symbol_info += f"- {key.replace('_', ' ').capitalize()}: {value}\n"
        else:
            symbol_info += "No additional details available for this symbol.\n"

        # Format large order information
        large_order_info = "\n## Large Orders Context\n\n"
        if large_order_context and large_order_context.get("count", 0) > 0:
            large_order_info += f"""
            - Total Large Orders: {large_order_context.get('count', 0)}
            - Buy Orders: {large_order_context.get('buy_count', 0)}
            - Sell Orders: {large_order_context.get('sell_count', 0)}
            - Net Value: ${large_order_context.get('net_value_usd', 0):,.2f}
            - Total Value: ${large_order_context.get('total_value_usd', 0):,.2f}

            ### Largest Orders:
            """

            for idx, order in enumerate(large_order_context.get("largest_orders", [])[:3]):
                order_time = datetime.fromtimestamp(order["timestamp"] / 1000).strftime("%H:%M:%S")
                large_order_info += f"{idx + 1}. {order['side'].upper()} {order['amount']} at ${order['price']} (${order['value_usd']:,.2f}) at {order_time}\n"
        else:
            large_order_info = ""  # Skip large order section if no data

        # Instructions for the model with more permissive approval criteria
        instructions = """
        ## Your Task

        Evaluate this alert from a professional trader's perspective and determine if it represents a meaningful trading opportunity.

        Consider:
        1. Price direction and trend strength
        2. Volume patterns and their interpretation (distribution vs. accumulation)
        3. Whether the asset is moving with or against the broader market
        4. Key support/resistance levels
        5. Technical indicator confirmations or divergences

        Trading Interpretation Guidelines:
        - High volume on downtrends often indicates distribution (bearish)
        - High volume on uptrends often indicates accumulation (bullish)
        - Assets moving against market trend need stronger confirmation signals
        - Understand the difference between breakout and breakdown signals
        - Consider the asset's market cap and liquidity in your assessment

        Alert Approval Guidelines:
        - APPROVE ALL GROUPED_ALERT signals (these represent multiple technical signals)
        - APPROVE ALL alerts for BTC and ETH (these are market-leading indicators)
        - APPROVE ALL MACD crossovers for top 20 cryptocurrencies by market cap
        - Be more permissive for top 100 market cap coins
        - Even with bearish indicators, if volume is notable, the alert is worth sending
        - When an asset is showing divergent behavior from the market, it's worth alerting
        - If in doubt between sending or not, choose to send the alert

        Formatting Guidelines:
        - DO NOT mention technical alert types (e.g., "RVOL_UP_EXTREME") in your analysis
        - Use professional trading terminology appropriate for experienced traders
        - Be explicit about whether the signal is bullish, bearish, or needs more confirmation
        - When volume is increasing during a price drop, make it clear this is likely distribution, not accumulation

        Provide your evaluation in the following JSON format:
        ```json
        {
            "should_send": true/false,
            "title": "Brief, professionally-worded title (max 35 chars)",
            "analysis": "1-2 concise sentences with professional market analysis and likely direction",
            "market_context": "Only include if market conditions directly impact this asset",
            "key_levels": "1-2 key support/resistance levels to watch"
        }
        ```

        Your analysis should provide actionable and accurate information for professional traders.
        """

        # Combine all parts of the prompt
        full_prompt = (
            alert_details + history + market_info + symbol_info + large_order_info + instructions
        )
        return full_prompt

    def _parse_llm_response(self, response_text: str, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the LLM response to extract structured data."""
        try:
            # Try to find and parse JSON from the response
            import re

            json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)

            if json_match:
                json_str = json_match.group(1)
                result = json.loads(json_str)
                return result

            # If no JSON block found, look for any JSON-like structure
            json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                result = json.loads(json_str)
                return result

            # If all parsing fails, provide a default response
            logger.warning(
                f"Failed to parse LLM response as JSON. Response: {response_text[:100]}..."
            )
            return {
                "should_send": True,  # Default to sending
                "title": f"Volume Spike on {alert.get('symbol', 'Unknown')}",
                "analysis": "Failed to parse LLM analysis.",
                "market_context": "Current market conditions could not be analyzed.",
                "key_levels": "No key levels identified.",
            }

        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return {
                "should_send": True,  # Default to sending
                "title": f"Volume Spike on {alert.get('symbol', 'Unknown')}",
                "analysis": "Failed to parse LLM analysis.",
                "market_context": "Error occurred during analysis.",
                "key_levels": "No key levels identified.",
            }
