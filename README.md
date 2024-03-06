# Sentinel

A Discord bot designed to help crypto traders save time browsing through charts to find the best setups.
User will get signals about important price and volume events right into discord.
-> Starting features are Volume Spike and RVOL (Relative Volume) alerts.

## Directory Structure

```
├── cogs/
│   ├── admin.py       # Admin commands for the bot.
│   └── help.py        # Help commands providing information about other commands.
├── config/
│   └── settings.py    # Configuration settings for the bot, pulling from environment variables.
├── data/
│   ├── db.py          # Database interactions, session management, and query execution.
│   └── models.py      # SQLAlchemy models for User and Guild.
├── utils/
│   ├── decorators.py  # Decorators for command checks, like is_creator or is_admin.
│   └── paginators.py  # Paginator utility for Discord messages.
├── main.py            # The main driver script for the bot. Initializes and runs the bot.
├── README.md          # This README file.
└── requirements.txt   # Python package dependencies for the project.

```

## Project Details
### Alerts Functionality
#### Trending Plays
Utilize RVOL with a threshold of 1.3 to identify elevated volume. Allow for manual adjustment of this threshold by the administrator to adapt to changing market conditions.
Determine a Higher High/Higher Low structure based on the last week's H1 bars, ensuring the analysis adapts dynamically to recent market behavior.
Include the flexibility to fetch OHLCV data for various timeframes (e.g., H1, D1, H4, etc.) using ccxt, allowing potential future expansion to monitor multiple timeframes.

#### Oversold Bounce Plays
Define "oversold" as an RSI below 30. Monitor this condition in the H1 timeframe.
Consider a bar as "closed positive" if its closing price is higher than its opening price, regardless of the specific dollar or percentage amount.

### User Interaction
#### Access Control
Integrate with the user's subscription status on a BuyMeACoffee page. Users with the appropriate role obtained through the subscription can access the bot's functionalities.
Design the bot to recognize and verify user roles, ensuring that only subscribed members can use the bot commands.

### Bot Commands
Implement commands (/trending and /bplays) to fetch the latest alerts of their respective types for the current day.
Consider adding an optional attribute to commands or a separate command to allow users to fetch historical alerts.
Ensure the bot is capable of handling multiple simultaneous commands, leveraging disnake's inherent capabilities for request handling and concurrency.

### Tech Stack and Scalability
Utilize ccxt for market data fetching, disnake for Discord bot integration, and PostgreSQL for database management.
Initially, the bot will operate on a single server. However, plan for a scalable architecture that can handle an increasing number of users and data volume. This includes efficient database management and possibly incorporating asynchronous processing to handle multiple requests concurrently.


## Main Components

### main.py

This is the entry point of the bot. It sets up logging, the bot object, signal handling for graceful shutdown, and database table creation. It also loads all cogs (extensions) and starts the bot.

### cogs

#### admin.py

Defines administrative commands for the bot. These can include moderation tasks, server management, etc.

#### help.py

Provides a help command that lists all available commands or detailed help for a specific command.

### config

#### settings.py

Contains settings for the bot, such as token, database URL, and other environment-specific settings.

### data

#### db.py

Handles database connections, session management, and provides utility functions for executing queries.

#### models.py

Defines SQLAlchemy models for database tables.

### utils

#### decorators.py

Provides decorators for command functions, adding checks like whether the user is the bot creator or an admin.

#### paginators.py

Utility for creating paginated messages in Discord, useful for commands that return large amounts of data.
