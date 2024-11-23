# Sentinel Discord Bot

## Overview
Sentinel is a sophisticated Discord bot designed to provide cryptocurrency traders with real-time market analysis and alerts. It monitors various technical indicators and market conditions to help traders identify potential trading opportunities.

## Key Features
- **MACD Alerts**: Real-time notifications for MACD crossovers and trend changes
- **RVOL (Relative Volume) Analysis**: Alerts for significant volume spikes compared to average
- **VWAP Monitoring**: Price analysis relative to Volume Weighted Average Price
- **Large Order Detection**: Notifications for significant market orders
- **Category Analysis**: Track and analyze different cryptocurrency categories
- **Trending Coins**: Monitor and report on trending cryptocurrencies
- **Interactive Commands**: User-friendly slash commands for market data access

## Technical Stack
- Built with Python 3.10
- Uses Discord.py for bot functionality
- CCXT for cryptocurrency exchange interactions
- SQLAlchemy for database management
- Plotly and Matplotlib for chart generation

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with required environment variables
4. Run the bot: `python main.py`

## Commands
- `/help` - Display all available commands
- `/price <ticker>` - Get current price for a cryptocurrency
- `/coin <name>` - Get detailed coin information
- `/trending` - View currently trending coins
- `/gainers` - View top gaining cryptocurrencies
- `/losers` - View top losing cryptocurrencies
- `/category <category>` - View category-specific analysis

## Contributing
This is a private project and contributions are not currently being accepted.

## License
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

This project is licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International - see the [LICENSE](LICENSE) file for details.

**Important Notice:**
- ⛔ Commercial use is strictly prohibited
- ⛔ Distribution of modified versions is not allowed
- ⛔ Use in other projects is not permitted
- ✅ Code viewing and learning is permitted
- ✅ Contributions to this repository may be accepted
- ✅ Attribution is required

The full license terms are available in the [LICENSE](LICENSE) file and at [Creative Commons BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
