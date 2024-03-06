# db.py
import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import scoped_session, sessionmaker

from config.settings import SQL_DATABASE_URL

from .models import Base

logger = logging.getLogger("Sentinel")

engine = create_engine(SQL_DATABASE_URL)
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)


class Database:
    def __init__(self):
        self.session = Session()

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

    @staticmethod
    def create_tables():
        Base.metadata.create_all(engine)

    @staticmethod
    def get_session():
        return Session()

    @staticmethod
    def close_session():
        Session.remove()
