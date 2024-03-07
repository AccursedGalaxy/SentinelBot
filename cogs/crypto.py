import logging
import os

import colorlog
import disnake
from disnake import Embed
from disnake.ext import commands

from config.settings import CG_API_KEY
from utils.chart import PlotChart
from utils.crypto_data import (fetch_current_price, fetch_top_gainers_losers,
                               validate_ticker)
from utils.paginators import ButtonPaginator as Paginator

logger = logging.getLogger("CryptoSentinel")

import io
import time

import aiohttp
import matplotlib.pyplot as plt

# Cache to store historical data
historical_data_cache = {}


async def fetch_historical_data(coin_id, days=1):
    cache_key = f"{coin_id}_{days}"
    if cache_key in historical_data_cache:
        # Return cached data if available
        return historical_data_cache[cache_key]

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    if not 2 <= days <= 90:
        params["interval"] = "daily"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                historical_data_cache[cache_key] = data["prices"]  # Cache the data
                return data["prices"]
            elif response.status == 429:
                logger.error(
                    f"Rate limit exceeded. Retrying after 60 seconds. Response: {await response.text()}"
                )
                time.sleep(60)  # Wait for 60 seconds before retrying
                return await fetch_historical_data(coin_id, days)
            else:
                logger.error(
                    f"Failed to fetch historical data for {coin_id}. Status: {response.status}. Response: {await response.text()}"
                )
                return None


async def fetch_trading_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/tickers"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("tickers", [])
            else:
                logger.error(
                    f"Failed to fetch trading data for {coin_id}. Status: {response.status}. Response: {await response.text()}"
                )
                return None


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

    @commands.slash_command(name="gainers", description="Shows the top 5 gainers")
    async def gainers(self, inter: disnake.ApplicationCommandInteraction):
        await inter.response.defer()
        top_gainers = await fetch_top_gainers_losers(
            api_key=CG_API_KEY, category="gainers"
        )
        for coin in top_gainers[:5]:
            embed = disnake.Embed(
                title=f"{coin['name']} ({coin['symbol'].upper()})",
                color=disnake.Color.green(),
            )
            embed.set_thumbnail(url=coin["image"])
            embed.add_field(name="Price", value=f"${coin['usd']:.6f}", inline=True)
            embed.add_field(
                name="24h Change", value=f"{coin['usd_24h_change']:.2f}%", inline=True
            )
            embed.add_field(
                name="Market Cap Rank", value=f"{coin['market_cap_rank']}", inline=True
            )
            await inter.followup.send(embed=embed)

    @commands.slash_command(name="losers", description="Shows the top 5 losers")
    async def losers(self, inter: disnake.ApplicationCommandInteraction):
        await inter.response.defer()
        top_losers = await fetch_top_gainers_losers(
            api_key=CG_API_KEY, category="losers"
        )
        for coin in top_losers[:5]:
            embed = disnake.Embed(
                title=f"{coin['name']} ({coin['symbol'].upper()})",
                color=disnake.Color.red(),
            )
            embed.set_thumbnail(url=coin["image"])
            embed.add_field(name="Price", value=f"${coin['usd']:.6f}", inline=True)
            embed.add_field(
                name="24h Change", value=f"{coin['usd_24h_change']:.2f}%", inline=True
            )
            embed.add_field(
                name="Market Cap Rank", value=f"{coin['market_cap_rank']}", inline=True
            )
            await inter.followup.send(embed=embed)

    @commands.slash_command(
        name="liquidity",
        description="Shows liquidity data for Bitcoin across various exchanges",
    )
    async def liquidity(self, inter: disnake.ApplicationCommandInteraction):
        await inter.response.defer()
        coin_id = "bitcoin"  # CoinGecko ID for Bitcoin
        trading_data = await fetch_trading_data(coin_id)

        if not trading_data:
            await inter.followup.send("Failed to fetch liquidity data.")
            return

        # Process and display the data
        embeds = []
        for exchange in trading_data[:10]:  # Limit to 10 exchanges for brevity
            embed = disnake.Embed(
                title=f"{exchange['market']['name']} - {exchange['base']}/{exchange['target']}",
                url=exchange.get("trade_url", "Not available"),
                color=disnake.Color.blue(),
            )
            embed.add_field(
                name="Last Trade Price", value=f"{exchange['last']}", inline=True
            )
            embed.add_field(name="Volume", value=f"{exchange['volume']}", inline=True)
            embed.add_field(
                name="Trust Score", value=f"{exchange['trust_score']}", inline=True
            )
            embed.add_field(
                name="Bid-Ask Spread",
                value=f"{exchange['bid_ask_spread_percentage']}",
                inline=True,
            )
            embed.add_field(
                name="Last Traded At",
                value=f"{exchange['last_traded_at']}",
                inline=True,
            )
            embeds.append(embed)

        # Use a paginator to handle multiple embeds
        paginator = Paginator(inter, embeds)
        await paginator.run()


def setup(bot):
    bot.add_cog(CryptoCommands(bot))
