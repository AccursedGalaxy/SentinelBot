import disnake
from disnake.ext import commands

from data.db import Database
from data.models import AlertsChannel, Guild, User
from logger_config import setup_logging
from utils.decorators import is_admin
from utils.paginators import ButtonPaginator as Paginator

logger = setup_logging()


class AdminCommands(commands.Cog, name="Admin Commands"):
    def __init__(self, bot):
        self.bot = bot

    @commands.slash_command(
        name="set_alerts_channel",
        description="Set the alerts channel you want to use. This is where the bot will send alerts.",
    )
    @is_admin()
    async def set_alerts_channel(
        self, inter: disnake.ApplicationCommandInteraction, channel: disnake.TextChannel
    ):
        session = Database.get_session()
        try:
            alerts_channel = (
                session.query(AlertsChannel).filter_by(guild_id=inter.guild.id).first()
            )
            if alerts_channel:
                alerts_channel.channel_id = channel.id
                alerts_channel.channel_name = channel.name
            else:
                new_channel = AlertsChannel(
                    channel_id=channel.id,
                    channel_name=channel.name,
                    guild_id=inter.guild.id,
                )
                session.add(new_channel)
            session.commit()
            await inter.response.send_message(f"Set bot channel to {channel.mention}")
            logger.info(f"Set bot channel to {channel.mention}")
        finally:
            Database.close_session()

    @commands.slash_command(
        name="remove_alerts_channel",
        description="Remove the alerts channel you want to use. This is where the bot will send alerts.",
    )
    @is_admin()
    async def remove_alerts_channel(self, inter: disnake.ApplicationCommandInteraction):
        session = Database.get_session()
        try:
            alerts_channel = (
                session.query(AlertsChannel).filter_by(guild_id=inter.guild.id).first()
            )
            if alerts_channel:
                session.delete(alerts_channel)
                session.commit()
                await inter.response.send_message("Removed bot channel")
                logger.info("Removed bot channel")
            else:
                await inter.response.send_message("No bot channel set")
        finally:
            Database.close_session()

    @commands.slash_command(
        name="show_alerts_channel",
        description="See what alerts channel is set currently.",
    )
    @is_admin()
    async def show_alerts_channel(self, inter: disnake.ApplicationCommandInteraction):
        session = Database.get_session()
        try:
            alerts_channel = (
                session.query(AlertsChannel).filter_by(guild_id=inter.guild.id).first()
            )
            if alerts_channel:
                # respond with the channel as clickable link
                await inter.response.send_message(
                    f"Bot channel is set to {inter.guild.get_channel(alerts_channel.channel_id).mention}"
                )
            else:
                await inter.response.send_message("No bot channel set")
        finally:
            Database.close_session()


def setup(bot):
    bot.add_cog(AdminCommands(bot))
