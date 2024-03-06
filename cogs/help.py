import logging

import disnake
from disnake.ext import commands

from utils.paginators import ButtonPaginator as Paginator

logger = logging.getLogger("Sentinel")


class HelpCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.slash_command(description="Get help with commands.")
    async def help(
        self, inter: disnake.ApplicationCommandInteraction, command_name: str = None
    ):
        if command_name:
            # Find the command by name
            command = self.bot.get_slash_command(command_name)
            if command:
                await self.send_command_help(inter, command)
            else:
                await inter.send("Command not found.", ephemeral=True)
        else:
            await self.send_general_help(inter)

    async def send_command_help(self, inter, command):
        embed = disnake.Embed(
            title=f"Help for /{command.name}",
            description=command.description or "No description",
            color=disnake.Color.blue(),
        )
        usage = f"/{command.name} " + " ".join(
            [f"<{option.name}>" for option in command.options]
        )
        embed.add_field(name="Usage", value=usage, inline=False)
        await inter.send(embed=embed, ephemeral=True)

    async def send_general_help(self, inter):
        items_per_page = 5
        commands_list = [
            cmd for cmd in self.bot.slash_commands if await cmd.can_run(inter)
        ]
        total_pages = (len(commands_list) + items_per_page - 1) // items_per_page
        embeds = []

        for i in range(0, len(commands_list), items_per_page):
            embed = disnake.Embed(title="Commands Help", color=disnake.Color.blue())
            for cmd in commands_list[i : i + items_per_page]:
                embed.add_field(
                    name=f"/{cmd.name}",
                    value=cmd.description or "No description",
                    inline=False,
                )
            page_info = f"Page {(i // items_per_page) + 1} of {total_pages}"
            embed.set_footer(text=page_info)
            embeds.append(embed)

        paginator = Paginator(self.bot, inter, embeds)
        await paginator.run()


def setup(bot):
    bot.add_cog(HelpCommands(bot))
