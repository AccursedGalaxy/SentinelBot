# Sentinel Discord Bot Project Specification

## Project Overview

Sentinel is a sophisticated Discord bot engineered to deliver real-time trading alerts to cryptocurrency traders, aiming to assist users in pinpointing lucrative market opportunities without continuous chart monitoring. This document details the vision, feature requirements, technical specifications, and future expansion plans for the bot.

## Key Functionalities

### MVP Alerts

1. **Trending Coins Alert:**
   - Dispatches a daily morning alert listing "trending" coins based on RVOL exceeding 1.3 (admin-adjustable) and a Higher High/Higher Low structure from the last week's H1 bars.
   - Allows dynamic adjustment of the timeframe for analysis and RVOL.

2. **Oversold Bounce Plays Alert:**
   - Triggers alerts for coins with an RSI below 30 on the H1 timeframe and a closing price higher than the opening price.

### User Interaction

- Users interact through a specific alerts channel within Discord, with commands like `/trending` and `/bplays` to fetch recent alerts.
- Subscribers can set custom RVOL thresholds and choose timeframes for analysis, enhancing personalization and relevance.

### Future Features and Alerts

- **Custom Subscription Alerts:** Users will be able to subscribe to alerts for specific coins or categories, enhancing user engagement and personalization.
- **Additional Alert Types:** Including Swing Trade alerts and Breakout Plays, with different frequencies based on the alert type, e.g., Swing Trade alerts weekly, Breakout Plays as they happen.

### User Interface and Presentation

- Utilize Discord's UI elements effectively, with clear, concise embeds, emojis, symbols, and well-formatted text for readability.
- Include OHLCV charts with relevant indicators (e.g., RSI for Oversold alerts, moving averages for Trending alerts) to enhance understanding and engagement.

## Technical Specifications

- **Data Handling and Processing:**
  - A separate worker module will asynchronously process a large array of coin OHLCV data, generating and storing alerts in the database.
  - The system will ensure no duplicate alerts are stored and will be designed to handle a significant amount of data efficiently.

- **Modular Architecture:**
  - While modularity increases complexity and hosting costs, it offers benefits like easier maintenance, scalability, feature addition, and testing.

- **Scalability and Performance:**
  - The bot should be designed to scale with an increasing number of users and alert types, maintaining performance and responsiveness.

## Monitoring and Maintenance

- **Monitoring Tools:**
  - Suggest employing tools like Prometheus for monitoring system performance and Grafana for visualizing metrics, ensuring the bot's health and performance are continuously assessed.
  - Implement logging and alerting mechanisms to detect and address issues proactively.

- **Maintenance Strategy:**
  - Establish a routine maintenance schedule to update dependencies, review and optimize code, and integrate user feedback into feature enhancements.

## Conclusion

Sentinel aims to be a dynamic, user-centric Discord bot that evolves with its user base, continuously enhancing its alert system and user experience while maintaining a robust, scalable architecture.

