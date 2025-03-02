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
        # Format current alert details
        alert_details = f"""
        ## Current Alert
        Symbol: {current_alert['symbol']}
        Type: {current_alert['alert_type']}
        Time: {current_alert['timestamp']}
        Price: {current_alert.get('price', 'N/A')}
        Volume: {current_alert.get('volume', 'N/A')}
        """

        # Add triggered alerts if this is a grouped alert
        if current_alert["alert_type"] == "GROUPED_ALERT" and "triggered_alerts" in current_alert:
            alert_details += "\n### Triggered Alerts\n"
            for idx, alert_type in enumerate(current_alert["triggered_alerts"]):
                alert_details += f"{idx + 1}. {alert_type}\n"

        # Add technical indicators if available
        indicators = current_alert.get("indicators", {})
        if indicators:
            alert_details += "\n### Technical Indicators\n"
            for key, value in indicators.items():
                alert_details += f"- {key}: {value}\n"

        # Format historical alerts
        history = "\n## Recent Alert History\n"
        if historical_alerts:
            for idx, alert in enumerate(historical_alerts[:5]):  # Limit to 5 most recent
                history += f"{idx + 1}. {alert['alert_type']} on {alert['timestamp']}: {alert.get('reasoning', 'No reasoning available')}\n"
        else:
            history += "No recent alerts for this symbol.\n"

        # Format market context
        market_info = "\n## Market Context\n"
        market_info += f"- BTC Dominance: {market_context.get('btc_dominance', 'N/A')}\n"
        market_info += f"- Fear & Greed Index: {market_context.get('fear_greed_index', 'N/A')}\n"
        market_info += (
            f"- Market Direction (24h): {market_context.get('market_direction', 'N/A')}\n"
        )
        market_info += (
            f"- Market Cap Change (24h): {market_context.get('market_cap_change', 'N/A')}%\n"
        )

        # Format symbol-specific details
        symbol_info = "\n## Symbol Details\n"
        symbol_info += (
            f"- Name: {symbol_details.get('name', current_alert['symbol'].split('/')[0])}\n"
        )
        if symbol_details.get("market_cap_rank", "N/A") != "N/A":
            symbol_info += f"- Market Cap Rank: {symbol_details.get('market_cap_rank', 'N/A')}\n"
        symbol_info += f"- Price Change (24h): {symbol_details.get('price_change_24h', 'N/A')}%\n"
        symbol_info += f"- Price Change (7d): {symbol_details.get('price_change_7d', 'N/A')}%\n"
        symbol_info += f"- Trading Volume (24h): {symbol_details.get('volume_24h', 'N/A')}\n"

        # Add large order context if available
        large_order_info = ""
        if large_order_context and large_order_context.get("count", 0) > 0:
            large_order_info = f"""
            ## Recent Large Orders
            - Total Large Orders: {large_order_context.get("count", 0)} in the last 4 hours
            - Buy Orders: {large_order_context.get("buy_count", 0)}
            - Sell Orders: {large_order_context.get("sell_count", 0)}
            - Net Order Flow: {"BUY" if large_order_context.get("net_value_usd", 0) > 0 else "SELL"} (${abs(large_order_context.get("net_value_usd", 0)):,.2f})

            ### Largest Orders:
            """

            for idx, order in enumerate(large_order_context.get("largest_orders", [])[:3]):
                order_time = datetime.fromtimestamp(order["timestamp"] / 1000).strftime("%H:%M:%S")
                large_order_info += f"{idx + 1}. {order['side'].upper()} {order['amount']} at ${order['price']} (${order['value_usd']:,.2f}) at {order_time}\n"

        # Update instructions for grouped alerts
        if current_alert["alert_type"] == "GROUPED_ALERT":
            instructions = """
            ## Task
            You're evaluating multiple technical signals that were triggered simultaneously for this asset.

            Analyze the confluence of these signals and determine if together they represent a significant trading opportunity.

            1. Consider how these signals interact and reinforce or contradict each other
            2. Evaluate their combined significance in the current market context
            3. Provide your evaluation in the following JSON format:
               {
                   "should_send": true/false,
                   "title": "Concise title highlighting the most significant signal combination (max 60 chars)",
                   "analysis": "Brief, comprehensive analysis of what these combined signals mean (3-4 sentences)",
                   "market_context": "Only include if the broader market conditions significantly impact or explain this signal. Otherwise, omit this field completely.",
                   "key_levels": "Support and resistance levels or other important price points to watch"
               }

            Be concise but comprehensive in your analysis, focusing on the interaction between the different signals.
            """
        else:
            # Original instructions for single alerts
            instructions = """
            ## Task
            Evaluate this alert and determine if it represents a meaningful trading signal worth notifying traders about.

            1. Analyze the technical indicators in the context of:
               - The overall market conditions
               - The token's recent performance
               - Historical alert patterns for this token
               - Volume and price action

            2. Provide your evaluation in the following JSON format:
               {
                   "should_send": true/false,
                   "title": "Concise, attention-grabbing title focusing on the key signal (max 60 chars)",
                   "analysis": "Brief, technical analysis of what this alert means (2-3 sentences)",
                   "market_context": "Only include if the broader market conditions significantly impact or explain this signal. Otherwise, omit this field completely.",
                   "key_levels": "Support and resistance levels or other important price points to watch"
               }

            Only recommend sending alerts for genuine signals with high potential impact. Filter out noise and false signals.
            Be concise and direct in your analysis - focus on facts and technical data, not opinions or predictions.
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
