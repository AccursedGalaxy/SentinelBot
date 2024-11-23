"""
Money Flow Analysis Module for Cryptocurrency Categories.

This module provides functionality to analyze and visualize money flow between
different cryptocurrency categories using CoinGecko API data. It includes tools
for fetching category data, analyzing market movements, and generating visual
reports.
"""

import asyncio
import os
import sys
import time
from typing import Dict, List, Tuple

# Third-party imports
import aiohttp
import disnake
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from config.settings import CG_API_KEY
from logger_config import setup_logging

# Configure paths before local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = setup_logging("MoneyFlow", "purple")


async def fetch_overall_market_data(api_key: str) -> Dict:  # type: ignore
    """Fetch overall cryptocurrency market data.

    Args:
        api_key: CoinGecko API key for authentication

    Returns:
        Dict containing market overview data
    """
    pass


def cleanup_report_files(file_paths: List[str]) -> None:
    """Clean up generated report files from the filesystem.

    Args:
        file_paths: List of file paths to be removed
    """
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info("Deleted file: %s", file_path)


async def fetch_coins_by_category(category: str, api_key: str | None = CG_API_KEY) -> List[Dict]:
    """Fetch all coins belonging to a specific category.

    Args:
        category: Category name to fetch coins for
        api_key: CoinGecko API key for authentication

    Returns:
        List of dictionaries containing coin data
    """
    coins_data = []
    page = 1
    while True:
        url = "https://pro-api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "category": category,
            "per_page": 350,
            "page": page,
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d,30d,1y",
        }
        headers = {"x-cg-pro-api-key": api_key}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    page_data = await response.json()

                    if not page_data:
                        break
                    coins_data.extend(page_data)
                    page += 1
                else:
                    logger.error(
                        "Failed to fetch coins by category '%s'. " "Status: %s. Response: %s",
                        category,
                        response.status,
                        await response.text(),
                    )
                    break
    return coins_data


async def fetch_category_info(api_key: str | None = CG_API_KEY) -> List[Dict]:
    """Fetch information about all cryptocurrency categories.

    Args:
        api_key: CoinGecko API key for authentication

    Returns:
        List of dictionaries containing category information
    """
    url = "https://pro-api.coingecko.com/api/v3/coins/categories"
    headers = {"x-cg-pro-api-key": api_key}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                logger.error(
                    "Failed to fetch category info. " "Status: %s. Response: %s",
                    response.status,
                    await response.text(),
                )
                return []


async def analyze_categories(
    categories: List[Dict],
) -> Tuple[Dict[str, float], List[Dict]]:
    """Analyze category data to determine money flow and top performers.

    Args:
        categories: List of category data dictionaries

    Returns:
        Tuple containing money flow analysis and top categories
    """
    total_volume_24h = sum(
        category["volume_24h"] for category in categories if category["volume_24h"]
    )
    logger.info("Total volume: %s", total_volume_24h)

    total_market_cap_change_24h = sum(
        category["market_cap_change_24h"]
        for category in categories
        if category["market_cap_change_24h"]
    )
    logger.info("Total market cap change: %s", total_market_cap_change_24h)

    money_flow_analysis = {}
    for category in categories:
        if category["volume_24h"] and category["market_cap_change_24h"]:
            volume_percentage = category["volume_24h"] / total_volume_24h
            market_cap_change_percentage = (
                category["market_cap_change_24h"] / total_market_cap_change_24h
            )
            normalized_money_flow = (volume_percentage - market_cap_change_percentage) / (
                volume_percentage + market_cap_change_percentage
            )
            money_flow_analysis[category["name"]] = normalized_money_flow

    top_categories = sorted(
        categories,
        key=lambda x: abs(x["market_cap_change_24h"] if x["market_cap_change_24h"] else 0),
        reverse=True,
    )[:5]
    logger.info("Top categories: %s", [category["name"] for category in top_categories])
    return money_flow_analysis, top_categories


def plot_performance(categories: List[Dict], title: str) -> None:
    """Generate a performance plot for the given categories.

    Args:
        categories: List of category data dictionaries
        title: Title for the plot
    """
    if not all(isinstance(cat, dict) for cat in categories):
        logger.error("Invalid data structure for categories in plot_performance")
        return

    names = [category["name"] for category in categories]
    values = [
        (category["market_cap_change_24h"] if category["market_cap_change_24h"] is not None else 0)
        for category in categories
    ]

    plt.figure(figsize=(10, 6))
    plt.barh(names, values, color="green")
    plt.xlabel("Normalized Money Flow")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()


def plot_money_flow(money_flow_analysis: Dict[str, float]) -> None:
    """Generate a Sankey diagram showing money flow between categories.

    Args:
        money_flow_analysis: Dictionary mapping category names to flow values
    """
    top_categories = sorted(money_flow_analysis.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    labels, values = zip(*top_categories)

    inflows = [value if value > 0 else 0 for value in values]
    outflows = [-value if value < 0 else 0 for value in values]

    source = [i for i in range(len(labels))] * 2
    target = [len(labels), len(labels) + 1] * len(labels)
    values = inflows + outflows
    labels = list(labels) + ["Inflow", "Outflow"]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                ),
                link=dict(source=source, target=target, value=values),
            )
        ]
    )
    fig.update_layout(title_text="Money Flow Between Top Categories", font_size=10)
    fig.write_image("money_flow.png")


async def generate_report() -> Tuple[disnake.Embed, List[disnake.File]]:
    """Generate a comprehensive money flow analysis report.

    Returns:
        Tuple containing the report embed and list of file objects
    """
    embed = disnake.Embed(
        title="Crypto Category Money Flow Report",
        description=(
            "This report provides an in-depth analysis of the money flow "
            "within various cryptocurrency categories, highlighting "
            "significant trends and movements."
        ),
        color=0x1E90FF,
    )

    report_files = []
    file_objects = []
    categories = await fetch_category_info()
    if categories:
        money_flow_analysis, top_categories = await analyze_categories(categories)

        performance_plot_filename = "top_categories_by_normalized_money_flow.png"
        plot_performance(top_categories, "Top Categories by Normalized Money Flow")
        report_files.append(performance_plot_filename)

        money_flow_plot_filename = "money_flow.png"
        plot_money_flow(money_flow_analysis)
        report_files.append(money_flow_plot_filename)

        for file_path in report_files:
            file = disnake.File(file_path, filename=file_path)
            file_objects.append(file)

    return embed, file_objects


if __name__ == "__main__":
    start_time = time.time()
    logger.info("Generating report...")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(generate_report())
    logger.info("Report generated in %.2f seconds.", time.time() - start_time)
    loop.close()
