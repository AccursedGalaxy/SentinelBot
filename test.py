import asyncio
import os
import signal
import sys
from io import BytesIO

import disnake
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from disnake.ext import commands, tasks
from trending_analysis import analyze_trending_coins

from alerts import CryptoAnalyzer
from config.settings import TEST_GUILDS, TOKEN
from data.db import Database
from data.models import AlertsChannel, Guild, User
from logger_config import setup_logging
from mflow.money_flow import cleanup_report_files, generate_report

logger = setup_logging("Sentinel", "green")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Bot initialization
intents = disnake.Intents.all()
bot = commands.InteractionBot(test_guilds=TEST_GUILDS, intents=intents)
MONEY_FLOW_CHANNEL = 1186336783261249638


class SimpleImageSender:
    def __init__(self, bot):
        self.bot = bot

    async def create_simple_plot(self):
        # Create a simple plot
        plt.plot([1, 2, 3, 4])
        plt.ylabel("some numbers")

        # Save the plot to a BytesIO object
        image_bytes = BytesIO()
        plt.savefig(image_bytes, format="png")
        image_bytes.seek(0)
        plt.close()

        return image_bytes

    async def send_image(self, channel_id):
        channel = self.bot.get_channel(channel_id)
        if channel:
            try:
                image_bytes = await self.create_simple_plot()
                file = disnake.File(image_bytes, filename="plot.png")
                await channel.send(file=file)
                print("Image sent successfully.")
            except Exception as e:
                print(f"Failed to send image: {e}")


@bot.event
async def on_ready():
    # on read send the simple image
    sender = SimpleImageSender(bot)
    logger.info("Sending simple image...")

    await sender.send_image(alerts_channel_id)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    Database.create_tables()
    try:
        bot.run(TOKEN)
    finally:
        loop.close()
        logger.info("Successfully shutdown the bot.")
