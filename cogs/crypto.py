import functools
import io
import logging
import os
import time

import aiohttp
import colorlog
import disnake
import matplotlib.pyplot as plt
from disnake import Embed
from disnake.ext import commands

from config.settings import CG_API_KEY
from utils.chart import PlotChart
from utils.crypto_data import (fetch_coin_data, fetch_current_price,
                               fetch_historical_data, fetch_new_coins,
                               fetch_top_gainers_losers, fetch_trending_coins,
                               validate_ticker)
from utils.paginators import ButtonPaginator as Paginator

logger = logging.getLogger("CryptoSentinel")


def cache_response(timeout):
    def decorator(func):
        cache = {}

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            result, expiry = cache.get(key, (None, 0))

            if time.time() < expiry:
                return result

            result = await func(*args, **kwargs)
            cache[key] = (result, time.time() + timeout)
            return result

        return wrapper

    return decorator


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
    @cache_response(3600)
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
    @cache_response(3600)
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
        name="new_listings",
        description="Shows newly listed cryptocurrencies from CoinGecko",
    )
    @cache_response(3600)
    async def new_listings(self, inter: disnake.ApplicationCommandInteraction):
        await inter.response.defer()
        new_coins = await fetch_new_coins()

        if new_coins:
            for coin in new_coins[:5]:  # Limiting to 5 coins for brevity
                # Fetch additional coin data
                coin_data = await fetch_coin_data(coin["id"])

                embed = disnake.Embed(
                    title=f"{coin['name']} ({coin['symbol'].upper()})",
                    description=f"ID: {coin['id']}\n{coin_data['description'][:200]}...",  # Show a snippet of the description
                    color=disnake.Color.blue(),
                )
                embed.add_field(
                    name="Activated At", value=coin["activated_at"], inline=False
                )
                embed.add_field(
                    name="Sentiment Up Votes",
                    value=f"{coin_data['sentiment_votes_up_percentage']}%",
                    inline=True,
                )
                embed.add_field(
                    name="Sentiment Down Votes",
                    value=f"{coin_data['sentiment_votes_down_percentage']}%",
                    inline=True,
                )
                # Check if market_cap_rank is available
                market_cap_rank = coin.get(
                    "market_cap_rank", "N/A"
                )  # Use 'N/A' if not available
                embed.add_field(
                    name="Market Cap Rank",
                    value=market_cap_rank,
                    inline=True,
                )

                # Check if the thumbnail URL is valid
                thumbnail_url = coin_data["image"]["thumb"]
                if thumbnail_url.startswith("http://") or thumbnail_url.startswith(
                    "https://"
                ):
                    embed.set_thumbnail(url=thumbnail_url)
                else:
                    embed.set_thumbnail(url="https://i.imgur.com/3s9fXtj.png")

                await inter.followup.send(embed=embed)

        else:
            await inter.followup.send("Failed to retrieve new coin listings.")

    @commands.slash_command(
        name="trending", description="Shows trending cryptocurrencies on CoinGecko"
    )
    @cache_response(3600)
    async def trending(self, inter: disnake.ApplicationCommandInteraction):
        await inter.response.defer()
        trending_coins = await fetch_trending_coins()

        if trending_coins:
            for coin in trending_coins["coins"][:5]:  # Limiting to 5 coins for brevity
                coin_data = coin["item"]
                embed = disnake.Embed(
                    title=f"{coin_data['name']} ({coin_data['symbol'].upper()})",
                    description=f"Market Cap Rank: {coin_data['market_cap_rank']}",
                    color=disnake.Color.gold(),
                )
                embed.set_thumbnail(url=coin_data["large"])

                # Format the price to handle small numbers
                price = float(coin_data["data"]["price"].replace("$", ""))
                if price < 0.01:
                    formatted_price = f"${price:.8f}"  # Show more decimal places for very small numbers
                else:
                    formatted_price = (
                        f"${price:.2f}"  # Standard formatting for larger numbers
                    )

                embed.add_field(name="Price", value=formatted_price, inline=True)
                embed.add_field(
                    name="Market Cap",
                    value=coin_data["data"]["market_cap"],
                    inline=True,
                )
                embed.add_field(
                    name="24h Change",
                    value=f"{coin_data['data']['price_change_percentage_24h']['usd']:.2f}%",
                    inline=False,
                )
                await inter.followup.send(embed=embed)
        else:
            await inter.followup.send("Failed to retrieve trending cryptocurrencies.")


def setup(bot):
    bot.add_cog(CryptoCommands(bot))
