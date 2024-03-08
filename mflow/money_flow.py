import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import aiohttp
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from config.settings import CG_API_KEY
from logger_config import setup_logging

logger = setup_logging("MoneyFlow", "purple")


# TODO: Implement a function to fetch overall market data (total market cap, volume, etc.)
async def fetch_overall_market_data(api_key=CG_API_KEY):
    pass


async def fetch_coins_by_category(category, api_key=CG_API_KEY):
    coins_data = []
    page = 1
    while True:
        url = f"https://pro-api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "category": category,
            "per_page": 250,
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
                        f"Failed to fetch coins by category '{category}'. Status: {response.status}. Response: {await response.text()}"
                    )
                    break
    return coins_data


async def fetch_category_info(api_key=CG_API_KEY):
    url = "https://pro-api.coingecko.com/api/v3/coins/categories"
    headers = {"x-cg-pro-api-key": api_key}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                logger.error(
                    f"Failed to fetch category info. Status: {response.status}. Response: {await response.text()}"
                )
                return None


async def analyze_categories(categories):
    total_volume = sum(
        category["volume_24h"] for category in categories if category["volume_24h"]
    )
    total_market_cap_change = sum(
        category["market_cap_change_24h"]
        for category in categories
        if category["market_cap_change_24h"]
    )

    money_flow_analysis = {}
    for category in categories:
        if category["volume_24h"] and category["market_cap_change_24h"]:
            volume_percentage = category["volume_24h"] / total_volume
            market_cap_change_percentage = (
                category["market_cap_change_24h"] / total_market_cap_change
            )
            normalized_money_flow = (
                volume_percentage - market_cap_change_percentage
            ) / (volume_percentage + market_cap_change_percentage)
            money_flow_analysis[category["name"]] = normalized_money_flow

    top_categories = sorted(
        categories,
        key=lambda x: abs(
            x["market_cap_change_24h"] if x["market_cap_change_24h"] else 0
        ),
        reverse=True,
    )[:5]
    return money_flow_analysis, top_categories


def plot_performance(categories, title):
    if not all(isinstance(cat, dict) for cat in categories):
        logger.error("Invalid data structure for categories in plot_performance")
        return

    names = [category["name"] for category in categories]
    values = [
        category["market_cap_change_24h"]
        if category["market_cap_change_24h"] is not None
        else 0
        for category in categories
    ]

    plt.figure(figsize=(10, 6))
    plt.barh(names, values, color="green")
    plt.xlabel("Normalized Money Flow")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()


def plot_money_flow(money_flow_analysis):
    top_categories = sorted(
        money_flow_analysis.items(), key=lambda x: abs(x[1]), reverse=True
    )[:10]
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


async def generate_report():
    # TODO: Include overall market data in the report
    # TODO: Enhance report with historical context comparison
    categories = await fetch_category_info()
    if categories:
        money_flow_analysis, top_categories = await analyze_categories(categories)
        plot_performance(top_categories, "Top Categories by Normalized Money Flow")
        plot_money_flow(money_flow_analysis)
        # TODO: Add more sections to the report as per requirements (e.g., volatility analysis, top coins spotlight)
        # Now you can use the generated images


if __name__ == "__main__":
    start_time = time.time()
    logger.info("Generating report...")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(generate_report())
    logger.info(f"Report generated in {time.time() - start_time:.2f} seconds.")
    loop.close()
