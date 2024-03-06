# Sentinel Discord Bot Project Specification - Expanded Edition

## Project Overview
Sentinel is a sophisticated Discord bot designed to deliver trading alerts to cryptocurrency traders, enabling them to identify lucrative market opportunities efficiently.
This document provides a comprehensive overview, detailing the bot's functionalities, user interaction, technical architecture, and future expansion plans.

## Key Functionalities

### MVP Alerts
1. **Trending Coins Alert:**
   - Daily alerts for coins meeting specific criteria like elevated RVOL (>1.3 - admin adjustable to fit market conditions) and a Higher Higher (HH), Higher Low (HL) pattern.

2. **Oversold Bounce Plays Alert:**
   - Hourly alerts for coins with RSI below 30 where last bar closed positively.

--> A unique alert will only be sent once per day, ensuring users receive only the most relevant and actionable information.

### Future Alert Types (Post-MVP)
- **Swing Trade Alerts:** For coins ranging within a specified Average True Range (ATR) band.
- **Breakout Plays Alerts:** For coins breaking significant levels (e.g., 52-week highs or all-time highs).

### User Interaction
- Users can use commands like `/trending` and `/bplays` to fetch the latest alerts.
- Post-MVP, users will be able to request specific data or analyses, like current RVOL or ATR for a coin.

### User Interface Enhancements
- Alerts will include OHLCV charts with relevant indicators, enhancing clarity and user engagement.
- Emojis, symbols, and color-coded embeds will be used for clear and attractive alert presentation.

## Technical Architecture and Specifications
### Data Processing
- A separate worker module will handle market data analysis, using asynchronous and batch processing to scan thousands of symbols.
--> using ccxt we can fetch data for a symbol, analyze, create a alert and store it in the database and do the next. -> this for all available symbols in a async loop with efficient batch sizes should do the trick.
----> we cant go overboard with resources, so we need to find a balance between performance and resource usage.
- The system will ensure no duplicate alerts are stored, maintaining database efficiency and relevance.

### Modular Architecture
- The bot will be designed with modularity, balancing between complexity, scalability, maintenance ease, and hosting costs.
- This approach facilitates feature expansions, testing, and system scalability.

### Scalability and Performance
- The architecture will support scaling to accommodate an increasing number of users and alert types.
- Efficient data handling and storage strategies will be crucial for maintaining performance as the system grows.

### Monitoring and Maintenance
- Implement monitoring tools like Prometheus or Grafana to track the bot's health and performance.
- Regular audits and updates will be scheduled to ensure the bot remains efficient, secure, and aligned with user needs.

## Future Considerations
- Establish a roadmap for introducing new alert types and user commands.
- Develop a feedback mechanism for users to contribute ideas and report issues, guiding the bot's continuous improvement.
