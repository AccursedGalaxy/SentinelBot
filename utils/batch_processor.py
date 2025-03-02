"""
Batch Processing Utility for LLM Alert Evaluation.

This module handles batch processing of alerts to reduce API costs and improve efficiency.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict

from data.db import Database
from data.models import Alert
from logger_config import setup_logging
from utils.llm_service import LLMService

logger = setup_logging("BATCH_PROCESSOR", "blue")


class BatchProcessor:
    """Process multiple alerts in batches for efficient LLM usage."""

    def __init__(self, max_batch_size=5, batch_interval_seconds=60):
        """
        Initialize the batch processor.

        Args:
            max_batch_size: Maximum number of alerts to process in a single batch
            batch_interval_seconds: Time in seconds to wait before processing a batch
        """
        self.db = Database()
        self.llm_service = LLMService()
        self.max_batch_size = max_batch_size
        self.batch_interval = batch_interval_seconds
        self.current_batch = []
        self.processing = False

    async def add_alert(
        self, symbol: str, alert_type: str, alert_data: Dict[str, Any], context: Dict[str, Any]
    ):
        """
        Add an alert to the current batch.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            alert_type: Type of alert (e.g., 'MACD_CROSSOVER_UP')
            alert_data: Data specific to this alert
            context: Context for LLM evaluation

        Returns:
            str: Batch ID for this alert
        """
        batch_id = str(uuid.uuid4())

        # Create an alert record
        current_time = datetime.now()
        alert = Alert(
            symbol=symbol,
            alert_type=alert_type,
            timestamp=current_time,
            last_alerted_at=current_time,
            data=alert_data,
            llm_processed=False,
            llm_sent=False,
            batch_id=batch_id,
        )

        self.db.session.add(alert)
        self.db.session.commit()

        # Add to current batch
        self.current_batch.append(
            {
                "alert_id": alert.id,
                "symbol": symbol,
                "alert_type": alert_type,
                "alert_data": alert_data,
                "context": context,
                "batch_id": batch_id,
            }
        )

        # Start batch processing if needed
        if len(self.current_batch) >= self.max_batch_size and not self.processing:
            asyncio.create_task(self.process_batch())
        elif not self.processing:
            # Start a timer to process the batch after the interval
            asyncio.create_task(self._schedule_processing())

        return batch_id

    async def _schedule_processing(self):
        """Schedule batch processing after the interval."""
        if not self.processing and self.current_batch:
            self.processing = True
            await asyncio.sleep(self.batch_interval)
            await self.process_batch()

    async def process_batch(self):
        """Process all alerts in the current batch."""
        if not self.current_batch:
            self.processing = False
            return

        logger.info(f"Processing batch of {len(self.current_batch)} alerts")
        self.processing = True

        batch_to_process = self.current_batch.copy()
        self.current_batch = []

        results = []

        for alert_item in batch_to_process:
            try:
                # Format data for LLM
                current_alert = {
                    "symbol": alert_item["symbol"],
                    "alert_type": alert_item["alert_type"],
                    "timestamp": datetime.now().isoformat(),
                    "price": alert_item["alert_data"].get("price", "N/A"),
                    "volume": alert_item["alert_data"].get("volume", "N/A"),
                    "indicators": alert_item["alert_data"].get("indicators", {}),
                }

                # Get LLM evaluation
                llm_result = await self.llm_service.evaluate_alert(
                    current_alert=current_alert,
                    historical_alerts=alert_item["context"].get("historical_alerts", []),
                    market_context=alert_item["context"].get("market_context", {}),
                    symbol_details=alert_item["context"].get("symbol_details", {}),
                )

                # Update the alert in the database
                alert = self.db.session.query(Alert).get(alert_item["alert_id"])
                if alert:
                    alert.llm_processed = True
                    alert.llm_sent = llm_result.get("should_send", False)
                    alert.llm_analysis = llm_result.get("analysis", "")
                    alert.llm_reasoning = llm_result.get("reasoning", "")
                    self.db.session.commit()

                # Store result
                results.append(
                    {
                        "alert_id": alert_item["alert_id"],
                        "batch_id": alert_item["batch_id"],
                        "symbol": alert_item["symbol"],
                        "alert_type": alert_item["alert_type"],
                        "llm_result": llm_result,
                    }
                )

            except Exception as e:
                logger.error(f"Error processing alert in batch: {e}")
                # Still commit the DB record
                alert = self.db.session.query(Alert).get(alert_item["alert_id"])
                if alert:
                    alert.llm_processed = True
                    alert.llm_sent = True  # Default to sending on error
                    alert.llm_analysis = f"Error during LLM processing: {str(e)}"
                    alert.llm_reasoning = (
                        "Error occurred during evaluation, defaulting to sending alert"
                    )
                    self.db.session.commit()

        self.processing = False

        # Check if new alerts arrived during processing
        if self.current_batch:
            asyncio.create_task(self._schedule_processing())

        return results

    async def get_result(self, batch_id: str) -> Dict[str, Any]:
        """
        Get the result for a specific alert by batch ID.

        Args:
            batch_id: The batch ID of the alert

        Returns:
            Dict containing the LLM evaluation results
        """
        # Wait for a reasonable time for processing to complete
        for _ in range(10):  # Try for 10 seconds
            alert = self.db.session.query(Alert).filter_by(batch_id=batch_id).first()
            if alert and alert.llm_processed:
                return {
                    "should_send": alert.llm_sent,
                    "analysis": alert.llm_analysis,
                    "reasoning": alert.llm_reasoning,
                }
            await asyncio.sleep(1)

        # If we reach here, the alert might still be processing
        alert = self.db.session.query(Alert).filter_by(batch_id=batch_id).first()
        if alert:
            # Force it to be considered processed and return
            alert.llm_processed = True
            alert.llm_sent = True  # Default to sending
            alert.llm_analysis = "Timed out waiting for LLM processing"
            alert.llm_reasoning = "Processing took too long, defaulting to sending alert"
            self.db.session.commit()

            return {
                "should_send": True,
                "analysis": "Timed out waiting for LLM processing",
                "reasoning": "Processing took too long, defaulting to sending alert",
            }

        return {
            "should_send": True,
            "analysis": "Alert not found in database",
            "reasoning": "Unable to locate alert record, defaulting to sending",
        }
