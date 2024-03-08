import asyncio
import os
import signal
import sys

import disnake
from disnake.ext import commands

from config.settings import TEST_GUILDS, TOKEN
from data.db import Database
from data.models import Guild, User
from logger_config import setup_logging

logger = setup_logging()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Bot initialization
intents = disnake.Intents.all()
bot = commands.InteractionBot(test_guilds=TEST_GUILDS, intents=intents)


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
                    joined_at=disnake.utils.utcnow(),
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

    # Set the bot's status
    await bot.change_presence(
        activity=disnake.Activity(
            type=disnake.ActivityType.watching,
            name="Sentinel | /sentinel - to get a overview",
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
                joined_at=disnake.utils.utcnow(),
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
