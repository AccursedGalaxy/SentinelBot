import asyncio
import io
import os
from typing import Dict, List, Optional

import aiohttp
import disnake
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from disnake.ext import commands
from PIL import Image

from config.settings import CATEGORIES, CG_API_KEY
from logger_config import setup_logging
from utils.cache import cache_response
from utils.crypto_data import (
    fetch_coin_info,
    fetch_coins_by_category,
    fetch_current_price,
    fetch_top_gainers_losers,
    fetch_trending_coins,
    validate_ticker,
)
from utils.paginators import ButtonPaginator as Paginator

logger = setup_logging()


def format_number(value):
    if value == 0:
        return "0"
    elif abs(value) < 1e-9:
        return f"{value:.10f}"  # For very small numbers, show 10 decimal places
    elif abs(value) < 1e-6:
        return f"{value:.8f}"  # For very small numbers, show 8 decimal places
    elif abs(value) < 1e-3:
        return f"{value:.6f}"  # For small numbers, show 6 decimal places
    else:
        return f"{value:.2f}"  # For larger numbers, show 2 decimal places


async def download_image(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                image_data = await response.read()
                return Image.open(io.BytesIO(image_data))


async def generate_bar_chart(coins, category):
    coin_names = [coin["name"] for coin in coins]
    price_changes = [coin["usd_24h_change"] for coin in coins]

    # Using softer color shades
    color_gainers = "#006400"
    color_losers = "#8B0000"
    colors = [color_gainers if category == "gainers" else color_losers for _ in coins]

    # Create the figure
    fig = go.Figure(
        data=[go.Bar(x=coin_names, y=price_changes, marker_color=colors, text=price_changes)]
    )

    # Customize layout
    fig.update_layout(
        title=f"Top 5 {category.capitalize()}",
        xaxis_title="Coin",
        yaxis_title="24h Change (%)",
        template="plotly_dark",
        showlegend=False,
    )

    # Improve legibility
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")

    # Save the figure to a file
    image_path = f"{category}_chart.png"
    fig.write_image(image_path)
    return image_path


async def generate_pie_chart(coins_data):
    # Sort coins by market cap
    sorted_coins = sorted(
        coins_data,
        key=lambda x: x.get("market_cap", 0) if x.get("market_cap") is not None else 0,
        reverse=True,
    )
    labels = [coin["name"] for coin in sorted_coins[:5]]
    sizes = [
        coin["market_cap"] if coin["market_cap"] is not None else 0 for coin in sorted_coins[:5]
    ]
    # Ensure that the pie chart includes other coins as well
    labels.append("Others")
    sizes.append(
        sum(
            coin["market_cap"] if coin["market_cap"] is not None else 0 for coin in sorted_coins[5:]
        )
    )

    # Check if all sizes are zero, which would cause an error in pie chart generation
    if all(size == 0 for size in sizes):
        return None  # Return None or handle this case as needed

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    pie_chart_path = "market_cap_distribution.png"
    plt.savefig(pie_chart_path)
    plt.close()
    return pie_chart_path


def format_currency(value: float) -> str:
    """Format a float as a currency string, converting to a more readable format."""
    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.1f}T"
    elif value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.1f}K"
    else:
        return f"${value:.2f}"


def create_category_embed(
    category: str,
    coins_data: List[Dict],
    sparkline_image_path: Optional[str] = None,
    pie_chart_path: Optional[str] = None,
) -> disnake.Embed:
    embed = disnake.Embed(
        title=f"{category.capitalize()} - Category Stats",
        description=f"Top gainers of the {category} category, index sparkline and market cap distribution.",
        color=disnake.Color.blue(),
    )

    # Top gainers
    top_gainers = sorted(
        coins_data, key=lambda x: x.get("price_change_percentage_24h", 0), reverse=True
    )[:3]
    for gainer in top_gainers:
        embed.add_field(
            name=f"{gainer['name']} ({gainer['symbol'].upper()})",
            value=f"Price: ${format_number(gainer['current_price'])}\n24h Change: {gainer['price_change_percentage_24h']:.2f}%",
            inline=False,
        )

    if sparkline_image_path:
        embed.set_image(url="attachment://sparkline.png")

    if pie_chart_path:
        embed.set_thumbnail(url=f"attachment://{pie_chart_path}")

    return embed


class CryptoCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.slash_command(name="gainers", description="Shows the top 5 gainers")
    @cache_response(3600)
    async def gainers(self, inter: disnake.ApplicationCommandInteraction):
        await inter.response.defer()
        top_gainers = await fetch_top_gainers_losers(api_key=CG_API_KEY, category="gainers")
        if top_gainers:
            chart_path = await generate_bar_chart(top_gainers, "gainers")
            embed = disnake.Embed(
                title="Top 5 Gainers",
                description="",  # Empty on purpose
                color=disnake.Color.from_rgb(0, 100, 0),
            )
            for i, coin in enumerate(top_gainers[:5], 1):
                embed.add_field(
                    name=f"{i}. {coin['name']}",
                    value=f"Price: ${format_number(coin['usd'])}\n24h Change: {format_number(coin['usd_24h_change'])}%",
                    inline=False,
                )
            await inter.followup.send(embed=embed, file=disnake.File(chart_path))
            os.remove(chart_path)
        else:
            await inter.followup.send("No data available for top gainers.")

    @commands.slash_command(name="losers", description="Shows the top 5 losers")
    @cache_response(3600)
    async def losers(self, inter: disnake.ApplicationCommandInteraction):
        await inter.response.defer()
        top_losers = await fetch_top_gainers_losers(api_key=CG_API_KEY, category="losers")
        if top_losers:
            chart_path = await generate_bar_chart(top_losers, "losers")
            embed = disnake.Embed(
                title="Top 5 Losers",
                description="",  # Empty on purpose
                color=disnake.Color.from_rgb(139, 0, 0),
            )
            for i, coin in enumerate(top_losers[:5], 1):
                embed.add_field(
                    name=f"{i}. {coin['name']}",
                    value=f"Price: ${format_number(coin['usd'])}\n24h Change: {format_number(coin['usd_24h_change'])}%",
                    inline=False,
                )
            await inter.followup.send(embed=embed, file=disnake.File(chart_path))
            os.remove(chart_path)

    @commands.slash_command(
        name="trending", description="Shows trending cryptocurrencies on CoinGecko"
    )
    @cache_response(3600)
    async def trending(self, inter: disnake.ApplicationCommandInteraction):
        await inter.response.defer()
        trending_coins = await fetch_trending_coins()

        if trending_coins:
            embed = disnake.Embed(
                title="Top 5 Trending Cryptocurrencies",
                description="",
                color=disnake.Color.blue(),
            )
            # Download the thumbnail images for the top 5 trending coins
            images = await asyncio.gather(
                *(download_image(coin["item"]["thumb"]) for coin in trending_coins[:5])
            )

            total_width = sum(image.width for image in images)
            total_height = max(image.height for image in images)
            collage = Image.new("RGB", (total_width, total_height))

            x_offset = 0
            for image in images:
                collage.paste(image, (x_offset, 0))
                x_offset += image.width

            collage_path = "collage.png"
            collage.save(collage_path)

            # Initialize an empty string to collect coin details
            coin_details_str = ""

            # Iterate over the top 5 trending coins to build the details string
            for coin in trending_coins[:5]:
                coin_details = coin.get("item", {})
                coin_name = coin_details.get("name", "N/A")
                coin_symbol = coin_details.get("symbol", "N/A").upper()
                price = coin_details.get("data", {}).get("price", "N/A")
                change_24h = (
                    coin_details.get("data", {})
                    .get("price_change_percentage_24h", {})
                    .get("usd", "N/A")
                )
                market_cap = coin_details.get("data", {}).get("market_cap", "N/A")
                total_volume = coin_details.get("data", {}).get("total_volume", "N/A")

                # Format numbers and percentages using the format_number and
                # format_currency functions
                if isinstance(price, (int, float)):
                    price = f"${format_number(price)}"
                if isinstance(change_24h, (int, float)):
                    change_24h = f"{format_number(change_24h)}%"
                if isinstance(market_cap, (int, float)):
                    market_cap = format_currency(market_cap)
                if isinstance(total_volume, (int, float)):
                    total_volume = format_currency(total_volume)

                coin_details_str += f"**{coin_name} ({coin_symbol})**\n"
                coin_details_str += f"Price: {price}\n"
                coin_details_str += f"24h Change: {change_24h}\n"
                coin_details_str += f"Market Cap: {market_cap}\n"
                coin_details_str += f"Total Volume: {total_volume}\n\n"

            # Add the consolidated coin details to the embed
            embed.add_field(name="", value=coin_details_str, inline=False)

            # Send the embed with the collage
            await inter.followup.send(
                file=disnake.File(collage_path, filename="collage.png"), embed=embed
            )

            # Clean up the collage image
            os.remove(collage_path)

        else:
            await inter.followup.send("Failed to retrieve trending coins.")

    @commands.slash_command(
        name="list_categories",
        description="Lists selected cryptocurrency categories from CoinGecko.",
    )
    async def list_categories(self, inter: disnake.ApplicationCommandInteraction):
        """Lists selected cryptocurrency categories from CoinGecko."""

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

        # Filter the fetched categories to include only those in the CATEGORIES env
        filtered_category_ids = CATEGORIES.split(",")
        filtered_categories = [
            category for category in categories if category["category_id"] in filtered_category_ids
        ]
        # Prepare the filtered categories for pagination, 25 per page
        category_pages = []
        items_per_page = 25
        for i in range(0, len(filtered_categories), items_per_page):
            embed = disnake.Embed(
                title=f"Cryptocurrency Categories (Page {i // items_per_page + 1})",
                description="",  # This is empty on purpose
                color=disnake.Color.blue(),
            )
            # Adjust the index for each item based on the current page
            category_list = "\n".join(
                [
                    f"{i + index + 1}. {category['category_id']}"
                    for index, category in enumerate(filtered_categories[i : i + items_per_page])
                ]
            )
            embed.add_field(
                name="Categories",
                value=category_list or "No categories available",
                inline=False,
            )
            category_pages.append(embed)

        # Use Paginator to display the categories
        paginator = Paginator(self.bot, inter, category_pages)
        await paginator.run()

    @commands.slash_command(
        name="category",
        description="Shows aggregated stats about coins in a specified category.",
    )
    @cache_response(3600)
    async def category(self, inter: disnake.ApplicationCommandInteraction, category: str):
        """Shows aggregated stats about coins in a specified category."""
        await inter.response.defer()
        coins_data = await fetch_coins_by_category(category)

        if not coins_data:
            await inter.followup.send(f"Failed to retrieve data for category: {category}")
            return

        # Generate a pie chart for the market cap distribution
        pie_chart_path = await generate_pie_chart(coins_data)

        # Extract sparkline data, ensuring each is a valid list of prices
        sparklines = [
            coin["sparkline_in_7d"]["price"]
            for coin in coins_data
            if "sparkline_in_7d" in coin and isinstance(coin["sparkline_in_7d"]["price"], list)
        ]

        # Handle cases where sparkline data may be missing or of unequal lengths
        if not sparklines:
            embed = create_category_embed(category, coins_data)
            await inter.followup.send(embed=embed)
            return

        # Normalize the length of sparklines to handle missing data points
        min_length = min(len(sparkline) for sparkline in sparklines)
        normalized_sparklines = [
            sparkline[:min_length] for sparkline in sparklines if len(sparkline) >= min_length
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
        files_to_send = [disnake.File(sparkline_image_path, filename="sparkline.png")]
        if pie_chart_path:
            files_to_send.append(
                disnake.File(pie_chart_path, filename="market_cap_distribution.png")
            )

        embed = create_category_embed(category, coins_data, sparkline_image_path, pie_chart_path)
        await inter.followup.send(
            files=files_to_send,
            embed=embed,
        )
        os.remove(sparkline_image_path)
        if pie_chart_path:
            os.remove(pie_chart_path)


def setup(bot):
    bot.add_cog(CryptoCommands(bot))
