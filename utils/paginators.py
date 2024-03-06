import logging

import colorlog
import disnake
from disnake.ext import commands

# Set up colorful logging
handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter("%(log_color)s%(asctime)s - %(levelname)s - %(message)s")
)

logger = colorlog.getLogger("Sentinel")
logger.setLevel(logging.INFO)


# Paginator class


class ButtonPaginator(disnake.ui.View):
    def __init__(self, bot, ctx, embeds, timeout=None):  # Set timeout to None
        super().__init__(timeout=timeout)
        self.bot = bot
        self.ctx = ctx
        self.embeds = embeds
        self.current_page = 0
        self.total_pages = len(embeds)
        self.message = None
        self.previous_button.disabled = self.current_page == 0
        self.next_button.disabled = self.current_page == self.total_pages - 1
        self.update_embed_footer()

    def update_embed_footer(self):
        page_info = f"Page {self.current_page + 1} of {self.total_pages}"
        self.embeds[self.current_page].set_footer(text=page_info)

    @disnake.ui.button(label="Previous", style=disnake.ButtonStyle.grey, disabled=True)
    async def previous_button(
        self, button: disnake.ui.Button, interaction: disnake.MessageInteraction
    ):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_embed_footer()
            self.next_button.disabled = False
            if self.current_page == 0:
                button.disabled = True
            await interaction.response.edit_message(
                embed=self.embeds[self.current_page], view=self
            )

    @disnake.ui.button(label="Next", style=disnake.ButtonStyle.grey, disabled=False)
    async def next_button(
        self, button: disnake.ui.Button, interaction: disnake.MessageInteraction
    ):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.update_embed_footer()
            self.previous_button.disabled = False
            if self.current_page == self.total_pages - 1:
                button.disabled = True
            await interaction.response.edit_message(
                embed=self.embeds[self.current_page], view=self
            )

    async def run(self):
        # Defer the response
        await self.ctx.response.defer()
        self.message = await self.ctx.send(
            embed=self.embeds[self.current_page], view=self
        )
