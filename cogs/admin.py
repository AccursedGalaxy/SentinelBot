import logging

import disnake
from disnake.ext import commands

from data.db import Database
from data.models import Guild, User
from utils.decorators import is_admin
from utils.paginators import ButtonPaginator as Paginator

logger = logging.getLogger("Sentinel")


class AdminCommands(commands.Cog, name="Admin Commands"):
    def __init__(self, bot):
        self.bot = bot


def setup(bot):
    bot.add_cog(AdminCommands(bot))
