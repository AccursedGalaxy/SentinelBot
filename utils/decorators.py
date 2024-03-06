import logging

import colorlog
import disnake
from disnake.ext import commands
from disnake.ext.commands import CheckFailure

from config.settings import CREATOR_ID
from data.db import Database
from data.models import BotChannel

# Logging setup
handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter("%(log_color)s%(levelname)s:%(name)s:%(message)s")
)

logger = colorlog.getLogger("CryptoSentinel")


def is_creator():
    def predicate(ctx):
        if str(ctx.author.id) == str(
            CREATOR_ID
        ):  # Ensuring both are strings for comparison
            return True
        return False

    return commands.check(predicate)


def is_admin():
    def predicate(ctx):
        if ctx.guild and ctx.author.guild_permissions.administrator:
            return True
        return False

    return commands.check(predicate)


def is_lowcaphunter():
    async def predicate(ctx):
        moderator_role_name = "LowCapHunter"  # Replace with your role name
        if ctx.guild:  # Check if the command is used in a server
            moderator_role = disnake.utils.get(
                ctx.guild.roles, name=moderator_role_name
            )
            if moderator_role and moderator_role in ctx.author.roles:
                return True
        await ctx.send(
            f"{ctx.author.mention} You do not have the required role to use this command."
        )
        return False

    return commands.check(predicate)


def is_bot_channel():
    async def predicate(ctx):
        if ctx.guild:
            db = Database()
            # Query the BotChannel table using SQLAlchemy syntax
            bot_channel = (
                db.session.query(BotChannel)
                .filter(BotChannel.guild_id == ctx.guild.id)
                .first()
            )
            if bot_channel:
                if ctx.channel.id == bot_channel.channel_id:
                    return True
                else:
                    await ctx.send(
                        f"{ctx.author.mention} Please use bot commands in <#{bot_channel.channel_id}>."
                    )
                    return False
            else:
                await ctx.send(
                    f"{ctx.author.mention} Please set a bot channel with the `set_bot_channel` command."
                )
                return False
        else:
            return True

    return commands.check(predicate)
