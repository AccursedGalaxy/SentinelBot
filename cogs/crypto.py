import logging
import os

import colorlog
import disnake
from disnake import Embed
from disnake.ext import commands

from utils.chart import PlotChart
from utils.crypto_data import fetch_current_price, validate_ticker
from utils.paginators import ButtonPaginator as Paginator

logger = logging.getLogger("CryptoSentinel")


class CryptoCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.slash_command(
        name="price",
        description="Gets the current price of a cryptocurrency. - usage: /price <ticker> - example: /price btcusdt",
    )
    async def price(self, inter: disnake.ApplicationCommandInteraction, ticker: str):
        """Gets the current price of a cryptocurrency."""
        await inter.response.defer()

        if not await validate_ticker(ticker):
            await inter.edit_original_message(
                content=f"Invalid ticker: {ticker.upper()}"
            )
            return

        # Get current price
        price = await fetch_current_price(ticker)

        # Send message
        await inter.edit_original_response(
            content=f"Current price of {ticker.upper()}: ${price}"
        )

    @commands.slash_command(
        name="chart",
        description="Get a chart for a ticker",
    )
    async def chart(
        self,
        inter: disnake.ApplicationCommandInteraction,
        symbol: str = commands.Param(description="Cryptocurrency symbol (e.g., BTC)"),
        timeframe: str = commands.Param(
            description="Timeframe for the chart (e.g., 1d, 1h)"
        ),
    ):
        """Plot a chart for a given cryptocurrency symbol and timeframe."""
        await inter.response.defer()

        try:
            # Call the plot_ohlcv_chart method from PlotChart class
            chart_file = await PlotChart.plot_ohlcv_chart(symbol, timeframe)

            if chart_file:
                # Send the chart image
                await inter.followup.send(file=disnake.File(chart_file))
                os.remove(chart_file)

            else:
                await inter.followup.send("Failed to generate chart.")

        except Exception as e:
            logger.error(f"Error in plot_chart command: {e}")
            await inter.followup.send(
                "An error occurred while processing your request."
            )


def setup(bot):
    bot.add_cog(CryptoCommands(bot))
