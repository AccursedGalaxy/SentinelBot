# Sentinel Discord Bot Project Specification

## Project Overview

Sentinel is a sophisticated Discord bot engineered to deliver real-time trading alerts to cryptocurrency traders. Its primary aim is to assist users in pinpointing lucrative market opportunities without the need for continuous chart monitoring. This document meticulously outlines the vision, specific feature requirements, and technical specifications for the bot.

## Key Functionalities

### MVP Alerts

1. **Trending Coins Alert:**
   - Dispatches a daily morning alert enumerating "trending" coins.
   - A coin is deemed "trending" based on two pivotal conditions:
     - RVOL exceeding 1.3, a threshold that is admin-adjustable to cater to fluctuating market dynamics.
     - Identification of a Higher High/Higher Low structure from the preceding week's H1 bars.
   - Provision for dynamic adjustment of the timeframe for the Higher High/Higher Low structure analysis and RVOL, accommodating different bar intervals such as 4H or 1D.

2. **Oversold Bounce Plays Alert:**
   - Triggers alerts for coins showcasing specific oversold characteristics.
   - The criteria for triggering an alert are as follows:
     - RSI plummeting below 30 within the H1 timeframe.
     - A bar is considered "closed positive" if its closing price surpasses the opening price.

### User Interaction

- Interaction with Sentinel is streamlined through a specific alerts channel within Discord.
- In the MVP phase, users have access to two principal commands:
  - `/trending`: Retrieves the most recent "Trending Coins" alerts.
  - `/bplays`: Gathers the latest alerts for "Oversold Bounce Plays."
- Access to the bot's services is gated behind a subscription through BuyMeACoffee, granting appropriate Discord roles to subscribers.

## Technical Requirements

- **Tech Stack:**
  - Python as the primary programming language.
  - PostgreSQL for database management.
  - Essential libraries: ccxt (for fetching market data) and disnake (for Discord API interactions).

- **Scalability:**
  - Initially hosted on a singular server, the bot's architecture should be robust enough to accommodate a growing user base and escalating data volumes.

## Technical Specifications

- Sentinel's architecture includes a main access point (main.py) that handles command interactions and alert dissemination.
- An additional "worker" module is dedicated to analyzing OHLCV data for cryptocurrencies fetched via ccxt. This module is responsible for detecting alert conditions, crafting alert messages, and storing these in the database.
- The main.py script then retrieves these alerts from the database to broadcast them on the designated Discord channel. (open to change this and have a additional module for this but each module will cost extra in hosting.)
- Alert frequencies are as follows: "Trending Coins" are alerted once every morning, and "Oversold Bounce Plays" are alerted hourly, with flexibility in adjusting these intervals.

