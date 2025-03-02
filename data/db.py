from sqlalchemy import create_engine, text
from sqlalchemy.orm import scoped_session, sessionmaker

from config.settings import SQL_DATABASE_URL
from logger_config import setup_logging

logger = setup_logging()


class Database:
    """Database connection manager."""

    _engine = None
    _session_factory = None
    _scoped_session = None
    _initialized = False

    def __init__(self):
        if not Database._initialized:
            Database._initialize()

    @classmethod
    def _initialize(cls):
        """Initialize the database engine and session factory."""
        try:
            if not SQL_DATABASE_URL:
                logger.error("Database URL is not set in environment variables")
                raise ValueError("Database URL is not set")

            cls._engine = create_engine(
                SQL_DATABASE_URL,
                echo=False,
                pool_pre_ping=True,  # Add connection testing
                pool_recycle=3600,  # Recycle connections after an hour
            )
            cls._session_factory = sessionmaker(bind=cls._engine)
            cls._scoped_session = scoped_session(cls._session_factory)
            cls._initialized = True
            logger.info("Database engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise

    @property
    def session(self):
        """Get a database session."""
        if not Database._initialized:
            Database._initialize()
        return Database._scoped_session()

    @property
    def engine(self):
        """Get the database engine."""
        if not Database._initialized:
            Database._initialize()
        return Database._engine

    @classmethod
    def create_tables(cls):
        """Create all tables defined in the models."""
        try:
            if not cls._initialized:
                cls._initialize()

            if not cls._engine:
                logger.error("Cannot create tables: Database engine is not initialized")
                return

            # Import Base here to avoid circular imports
            from data.models import Base

            Base.metadata.create_all(cls._engine)
            logger.info("Tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")

    @classmethod
    def get_session(cls):
        """Get a session from the scoped session registry."""
        if not cls._initialized:
            cls._initialize()
        return cls._scoped_session()

    @classmethod
    def close_session(cls):
        """Close the current session."""
        if cls._scoped_session:
            cls._scoped_session.remove()

    def execute_query(self, query, params=None):
        result = self.session.execute(query, params or ())
        if query.strip().upper().startswith("SELECT"):
            return result.fetchall()
        return None

    def fetch_one(self, query, params=None):
        result = self.session.execute(text(query), params or ())
        return result.fetchone()

    def fetch_all(self, query, params=None):
        result = self.session.execute(query, params or ())
        return result.fetchall()

    # Robust execute method to delete, update, or insert
    def execute(self, query, params=None):
        self.session.execute(text(query), params or ())
        self.session.commit()

    def execute_many(self, query, params):
        self.session.execute_many(query, params)
        self.session.commit()

    def save(self, model):
        # save data to database
        self.session.add(model)
        self.session.commit()

    def bulk_fetch(self, query, params=None):
        result = self.session.execute(query, params or ())
        return result.fetchall()

    def bulk_save(self, models):
        # save data to database
        self.session.bulk_save_objects(models)
        self.session.commit()

    def delete_all_from_table(self, table):
        # delete all data from table
        self.session.query(table).delete()
        self.session.commit()
        logger.info(f"Deleted all data from {table.__tablename__}")

    def remove(self, model):
        # remove data from database
        self.session.delete(model)
        self.session.commit()

    def rollback(self):
        """Rollback the current session."""
        try:
            self.session.rollback()
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
