# Sentinel 

## Description
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
