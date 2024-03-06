import os

from dotenv import load_dotenv

load_dotenv()
near_percentage = 0.03
# import env variables
TOKEN = os.getenv("TOKEN")
CREATOR_ID = os.getenv("CREATOR_ID")
TEST_GUILDS = [
    int(guild_id)
    for guild_id in os.getenv("TEST_GUILDS", "").split(",")
    if guild_id.strip()
]
SQL_DATABASE_URL = os.getenv("SQL_DATABASE_URL")
STARTING_BALANCE = int(os.getenv("STARTING_BALANCE", 100))
X_RAPIDAPI_KEY = os.getenv("X_RAPIDAPI_KEY")
CMC_API_KEY = os.getenv("CMC_API_KEY")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 100))
