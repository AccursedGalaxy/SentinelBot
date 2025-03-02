import asyncio
import os
import signal
import sys

import disnake
from disnake.ext import commands

from alerts import CryptoAnalyzer
from config.settings import (
    LARGE_ORDERS_ALERTS_CHANNEL,
    MACD_ALERTS_CHANNEL,
    MAIN_ALERTS_CHANNEL,
    RVOL_ALERTS_CHANNEL,
    TEST_GUILDS,
    TOKEN,
    VWAP_ALERTS_CHANNEL,
)
from data.db import Database
from data.models import AlertsChannel, Guild, User
from logger_config import setup_logging

logger = setup_logging("Sentinel", "green")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Bot initialization
intents = disnake.Intents.all()
bot = commands.InteractionBot(test_guilds=TEST_GUILDS, intents=intents)
MONEY_FLOW_CHANNEL = 1186336783261249638
exchange = "binance"
analysis_timeframe = "4h"
analysis_lookback = 31

ping_main_alerts = "<@&1217104257216679987>"
ping_rvol_alerts = "<@&1217105162351673455>"
ping_macd_alerts = "<@&1217105204856488018>"
ping_vwap_alerts = "<@&1217483711893733478>"
ping_large_order_alerts = "<@&1217483711893733478>"

main_alerts_channel = MAIN_ALERTS_CHANNEL
rvol_alerts_channel = RVOL_ALERTS_CHANNEL
macd_alerts_channel = MACD_ALERTS_CHANNEL
vwap_alerts_channel = VWAP_ALERTS_CHANNEL
large_order_alerts_channel = LARGE_ORDERS_ALERTS_CHANNEL

alert_channels = {
    # "RVOL_UP_EXTREME": rvol_alerts_channel,
    "MACD_CROSSOVER_UP": macd_alerts_channel,
    "MACD_CROSSOVER_DOWN": macd_alerts_channel,
    "RVOL_MACD_CROSS_UP": main_alerts_channel,
    "RVOL_MACD_CROSS_DOWN": main_alerts_channel,
    # "VWAP_ALERT": vwap_alerts_channel,
    "LARGE_ORDER": large_order_alerts_channel,
}
ping_roles = {
    "RVOL_UP_EXTREME": ping_rvol_alerts,
    "MACD_CROSSOVER_UP": ping_macd_alerts,
    "MACD_CROSSOVER_DOWN": ping_macd_alerts,
    "RVOL_MACD_CROSS_UP": ping_main_alerts,
    "RVOL_MACD_CROSS_DOWN": ping_main_alerts,
    "VWAP_ALERT": ping_vwap_alerts,
    "LARGE_ORDER": ping_large_order_alerts,
}


# Bot events
@bot.event
async def on_ready():
    session = Database.get_session()
    try:
        # add guilds to the database
        for guild in bot.guilds:
            guild_record = session.query(Guild).filter_by(guild_id=guild.id).first()
            if not guild_record:
                new_guild = Guild(
                    guild_id=guild.id,
                    guild_name=guild.name,
                    joined_at=disnake.utils.now(),
                )
                session.add(new_guild)

            # add members to the database
            for member in guild.members:
                user_record = session.query(User).filter_by(user_id=member.id).first()
                if not user_record:
                    new_user = User(
                        user_id=member.id,
                        username=member.name,
                        joined_at=member.joined_at,
                        is_bot=member.bot,
                    )
                    session.add(new_user)

        session.commit()

    finally:
        Database.close_session()
        analyzer = CryptoAnalyzer(
            exchange,
            analysis_timeframe,
            analysis_lookback,
            bot,
            alert_channels,
            ping_roles,
        )
        bot.loop.create_task(analyzer.run())

    # Set the bot's status
    await bot.change_presence(
        activity=disnake.Activity(
            type=disnake.ActivityType.watching,
            name="Sentinel | /help for commands",
        )
    )
    logger.info(f"Logged in as {bot.user.name} ({bot.user.id})")
    logger.info(f"Connected to {len(bot.guilds)} guilds")
    logger.info(f"Connected with {len(bot.users)} users")
    logger.info("Bot is ready!")


@bot.event
async def on_member_join(member):
    session = Database.get_session()
    try:
        user_record = session.query(User).filter_by(user_id=member.id).first()
        if not user_record:
            new_user = User(
                user_id=member.id,
                username=member.name,
                joined_at=member.joined_at,
                is_bot=member.bot,
            )
            session.add(new_user)
            session.commit()
            logger.info("Added user %s to the database", member.name)
    finally:
        Database.close_session()


@bot.event
async def on_guild_join(guild):
    session = Database.get_session()
    try:
        guild_record = session.query(Guild).filter_by(guild_id=guild.id).first()
        if not guild_record:
            new_guild = Guild(
                guild_id=guild.id,
                guild_name=guild.name,
                joined_at=disnake.utils.now(),
            )
            session.add(new_guild)
            session.commit()
            logger.info("Added guild %s to the database", guild.name)
    finally:
        Database.close_session()


async def load_cogs():
    cogs_dir = os.path.join(os.path.dirname(__file__), "cogs")
    if not os.path.exists(cogs_dir):
        logger.error("Could not find cogs directory")
        return

    for filename in os.listdir(cogs_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            cog_path = f"cogs.{filename[:-3]}"
            try:
                bot.load_extension(cog_path)
                logger.info("Loaded Cog: %s", filename)
            except Exception as e:
                logger.exception("Failed to load cog: %s", e)


# get alerts channel ID for the guild
async def get_alerts_channel():
    session = Database.get_session()
    try:
        alerts_channel = session.query(AlertsChannel).filter_by(guild_id=TEST_GUILDS).first()
        if alerts_channel:
            return alerts_channel.channel_id
    finally:
        Database.close_session()


# Graceful shutdown
async def shutdown(signal, loop):
    logger.info(f"Received exit signal {signal.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]

    logger.info("Canceling outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(s, lambda s=s: asyncio.create_task(shutdown(s, loop)))

    # Create tables if they don't exist
    Database.create_tables()

    try:
        loop.run_until_complete(load_cogs())
        bot.run(TOKEN)
    finally:
        loop.close()
        logger.info("Successfully shutdown the bot.")
