import asyncio

import disnake
from disnake.ext import commands

from config.settings import ANNOUNCEMENT_CHANNEL


class IntroView(disnake.ui.View):
    def __init__(self, embeds):
        super().__init__()
        self.embeds = embeds
        self.current_index = 0
        self.update_footer()

    def update_footer(self):
        total_embeds = len(self.embeds)
        for i, embed in enumerate(self.embeds):
            embed.set_footer(
                text=f"Page {i + 1} of {total_embeds}. Use the buttons to navigate."
            )

    @disnake.ui.button(label="Previous", style=disnake.ButtonStyle.red, disabled=True)
    async def previous_button(
        self, button: disnake.ui.Button, interaction: disnake.MessageInteraction
    ):
        self.current_index -= 1
        button.disabled = self.current_index == 0
        self.children[1].disabled = (
            self.current_index == len(self.embeds) - 1
        )  # 'Next' button
        await interaction.response.edit_message(
            embed=self.embeds[self.current_index], view=self
        )

    @disnake.ui.button(label="Next", style=disnake.ButtonStyle.green, disabled=False)
    async def next_button(
        self, button: disnake.ui.Button, interaction: disnake.MessageInteraction
    ):
        if self.current_index < len(self.embeds) - 1:
            self.current_index += 1
            button.disabled = self.current_index == len(self.embeds) - 1
            self.children[0].disabled = self.current_index == 0  # 'Previous' button
            await interaction.response.edit_message(
                embed=self.embeds[self.current_index], view=self
            )


class StartCommand(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.slash_command(
        name="start",
        description="Begin your journey with SentinelBot and discover its features!",
    )
    async def start(self, inter: disnake.ApplicationCommandInteraction):
        if inter.channel.name != "▶︱get-started":
            await inter.response.send_message(
                "Please use this command in the #get-started channel.", ephemeral=True
            )
            return

        # Define the series of embeds for the intro experience
        embeds = [
            disnake.Embed(
                title="🎉 Welcome to SentinelBot! 🎉",
                description=(
                    "I'm here to guide you through the exciting world of cryptocurrencies! 🚀\n\n"
                    "With SentinelBot, you can track real-time market data, get detailed insights into various cryptocurrencies, "
                    "and stay updated with the latest trends in the crypto world.\n\n"
                    "Let's embark on this journey together to explore the dynamic landscape of cryptocurrencies! 💼"
                ),
                color=disnake.Color.from_rgb(0, 255, 255),
            ).set_image(url="attachment://start1.png"),
            disnake.Embed(
                title="📈 Real-Time Market Data",
                description=(
                    "Stay updated with the latest market data and track the performance of various cryptocurrencies in real-time.\n\n"
                    "You can do **/price <ticker>** to get the current price of a cryptocurrency, for example **/price BTCUSDT**."
                ),
                color=disnake.Color.from_rgb(0, 255, 255),
            ).set_image(url="attachment://start1.png"),
            disnake.Embed(
                title="📊 Detailed Insights",
                description=(
                    "Get detailed insights into various cryptocurrencies and their historical performance.\n\n"
                    "You can do **/coin <name>** to get detailed information about a cryptocurrency, for example **/coin gala**. \n\n"
                    "Please note that here you have to use the full coin name, not the ticker. (we are working on a way to use the ticker as well)"
                ),
                color=disnake.Color.from_rgb(0, 255, 255),
            ).set_image(url="attachment://start1.png"),
            disnake.Embed(
                title="📈📉 Trending Coins",
                description=(
                    "Discover the trending cryptocurrencies and stay updated with the latest trends in the crypto world.\n\n"
                    "**/gainers** will give you a list of the recent top gainers among the top cryptocurrencies. \n\n"
                    "**/losers** will give you a list of the recent top losers among the top cryptocurrencies. \n\n"
                    "**/trending** will give you a list of the trending cryptocurrencies."
                ),
                color=disnake.Color.from_rgb(0, 255, 255),
            ).set_image(url="attachment://start1.png"),
            disnake.Embed(
                title="📊📈 Advanced Features",
                description=(
                    "Explore advanced features such as tracking the money flow in the cryptocurrency market and more.\n\n"
                    "You can do **/category <category>** to get detailed statistics about any category from CoinGecko. \n\n"
                    "If you want to see the main categories, you can do **/list_categories**. \n\n"
                    "The **/money_flow** command will give you a report of the money flow between the different categories. - Unfortunatly this feature is still in development. \n"
                    f"Stay tuned for more updates in the <#{ANNOUNCEMENT_CHANNEL}> channel!"
                ),
                color=disnake.Color.from_rgb(0, 255, 255),
            ).set_image(url="attachment://start1.png"),
        ]

        # Initialize the view with the embeds
        view = IntroView(embeds)

        # Send the first embed with the image as an attachment
        file = disnake.File("assets/start1.png", filename="start1.png")
        await inter.response.send_message(
            embed=embeds[0], view=view, file=file, ephemeral=True
        )


def setup(bot):
    bot.add_cog(StartCommand(bot))
