import functools
import io
import logging
import os
import time

import aiohttp
import colorlog
import disnake
import matplotlib.pyplot as plt
import numpy as np
from disnake import Embed
from disnake.ext import commands

from config.settings import CG_API_KEY
from utils.cache import cache_response
from utils.chart import PlotChart
from utils.crypto_data import (fetch_coin_data, fetch_coins_by_category,
                               fetch_current_price, fetch_new_coins,
                               fetch_top_gainers_losers, fetch_trending_coins,
                               validate_ticker)
from utils.paginators import ButtonPaginator as Paginator

logger = logging.getLogger("CryptoSentinel")


def create_category_embed(category, coins_data, sparkline_image_path=None):
    # Sort coins by market cap to find the top 3
    top_coins_by_market_cap = sorted(
        coins_data, key=lambda x: x.get("market_cap", 0), reverse=True
    )[:3]

    # Filter out coins where 'price_change_percentage_24h' is None and find the min/max
    valid_24h_changes = [
        coin
        for coin in coins_data
        if coin.get("price_change_percentage_24h") is not None
    ]
    if valid_24h_changes:
        highest_24h_increase = max(
            valid_24h_changes, key=lambda x: x["price_change_percentage_24h"]
        )
        min_24h_change = min(
            valid_24h_changes, key=lambda x: x["price_change_percentage_24h"]
        )
    else:
        highest_24h_increase = None
        min_24h_change = None

    embed = disnake.Embed(
        title=f"{category.capitalize()} - Category Stats:",
        description=f"Top coins in the {category} category based on market cap and price changes.",
        color=disnake.Color.blue(),
    )

    # Add fields for top coins by market cap
    for i, coin in enumerate(top_coins_by_market_cap, 1):
        embed.add_field(
            name=f"Top {i}: {coin['name']} ({coin['symbol'].upper()})",
            value=f"Market Cap: ${coin['market_cap']}\nCurrent Price: ${coin['current_price']:.6f}\n24h Change: {coin.get('price_change_percentage_24h', 'N/A')}",
            inline=False,
        )

    # Add field for the highest 24h increase, if available
    if highest_24h_increase:
        embed.add_field(
            name=f"Highest 24h Increase: {highest_24h_increase['name']} ({highest_24h_increase['symbol'].upper()})",
            value=f"Current Price: ${highest_24h_increase['current_price']:.6f}\n24h Change: {highest_24h_increase['price_change_percentage_24h']:.2f}%",
            inline=False,
        )

    # Add field for the lowest 24h change, if available
    if min_24h_change:
        embed.add_field(
            name=f"Lowest 24h Change: {min_24h_change['name']} ({min_24h_change['symbol'].upper()})",
            value=f"Current Price: ${min_24h_change['current_price']:.6f}\n24h Change: {min_24h_change['price_change_percentage_24h']:.2f}%",
            inline=False,
        )

    if sparkline_image_path:
        embed.set_image(url="attachment://sparkline.png")

    return embed


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

    @commands.slash_command(
        name="list_categories",
        description="Lists all cryptocurrency categories from CoinGecko.",
    )
    async def list_categories(self, inter: disnake.ApplicationCommandInteraction):
        """Lists all cryptocurrency categories from CoinGecko."""

        # Fetch categories from CoinGecko
        url = "https://pro-api.coingecko.com/api/v3/coins/categories/list"
        headers = {"x-cg-pro-api-key": CG_API_KEY}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    categories = await response.json()
                else:
                    logger.error(
                        f"Failed to fetch categories. Status: {response.status}. Response: {await response.text()}"
                    )
                    await inter.followup.send("Failed to retrieve categories.")
                    return

        # Prepare the categories for pagination, 25 per page
        category_pages = []
        for i in range(0, len(categories), 25):
            embed = disnake.Embed(
                title=f"Cryptocurrency Categories (Page {i//25 + 1})",
                description="",
                color=disnake.Color.blue(),
            )
            category_ids = "\n".join(
                [category["category_id"] for category in categories[i : i + 25]]
            )
            embed.add_field(name="Category IDs", value=category_ids, inline=False)
            category_pages.append(embed)

        # Use Paginator to display the categories
        paginator = Paginator(self.bot, inter, category_pages)
        await paginator.run()

    @commands.slash_command(
        name="category",
        description="Shows aggregated stats about coins in a specified category.",
    )
    async def category(
        self, inter: disnake.ApplicationCommandInteraction, category: str
    ):
        """Shows aggregated stats about coins in a specified category."""
        await inter.response.defer()
        coins_data = await fetch_coins_by_category(category)

        if not coins_data:
            await inter.followup.send(
                f"Failed to retrieve data for category: {category}"
            )
            return

        # Extract sparkline data, ensuring each is a valid list of prices
        sparklines = [
            coin["sparkline_in_7d"]["price"]
            for coin in coins_data
            if "sparkline_in_7d" in coin
            and isinstance(coin["sparkline_in_7d"]["price"], list)
        ]

        # Handle cases where sparkline data may be missing or of unequal lengths
        if not sparklines:
            embed = create_category_embed(category, coins_data)
            await inter.followup.send(embed=embed)
            return

        # Normalize the length of sparklines to handle missing data points
        min_length = min(len(sparkline) for sparkline in sparklines)
        normalized_sparklines = [
            sparkline[:min_length]
            for sparkline in sparklines
            if len(sparkline) >= min_length
        ]

        # Calculate the average sparkline
        avg_sparkline = np.mean(np.array(normalized_sparklines), axis=0)

        # Generate and save the plot
        plt.figure(figsize=(6, 2))
        plt.plot(avg_sparkline, color="blue")
        plt.axis("off")
        sparkline_image_path = "sparkline.png"
        plt.savefig(sparkline_image_path)
        plt.close()

        # Create and send the embed with the sparkline image
        embed = create_category_embed(category, coins_data, sparkline_image_path)
        await inter.followup.send(
            file=disnake.File(sparkline_image_path, filename="sparkline.png"),
            embed=embed,
        )
        os.remove(sparkline_image_path)


def setup(bot):
    bot.add_cog(CryptoCommands(bot))
