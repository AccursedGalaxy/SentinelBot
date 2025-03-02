"""
Database initialization script.
Drops and recreates all tables.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db import Database
from data.models import Base
from logger_config import setup_logging

logger = setup_logging("DB_INIT", "red")


def initialize_database():
    """Drop and recreate all tables"""
    db = Database()
    try:
        logger.warning("Dropping all tables... This will delete all data!")
        Base.metadata.drop_all(db.engine)
        logger.info("Creating all tables...")
        Base.metadata.create_all(db.engine)
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    finally:
        db.session.close()


if __name__ == "__main__":
    logger.info("Starting database initialization")
    answer = input("This will DELETE ALL DATA. Are you sure? (yes/no): ")
    if answer.lower() == "yes":
        initialize_database()
        logger.info("Database initialization complete")
    else:
        logger.info("Database initialization cancelled")
