# database models
import logging

import colorlog
from sqlalchemy import (BigInteger, Boolean, Column, DateTime, Float, Integer,
                        String, Text)
from sqlalchemy.ext.declarative import declarative_base

logger = logging.getLogger("Sentinel")

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    user_id = Column(BigInteger, primary_key=True)
    username = Column(String)
    joined_at = Column(DateTime)
    is_bot = Column(Boolean)

    def __repr__(self):
        return "<User(user_id='%s', username='%s', joined_at='%s', is_bot='%s')>" % (
            self.user_id,
            self.username,
            self.joined_at,
            self.is_bot,
        )


class Guild(Base):
    __tablename__ = "guilds"

    guild_id = Column(BigInteger, primary_key=True)
    guild_name = Column(String)
    joined_at = Column(DateTime)

    def __repr__(self):
        return "<Guild(guild_id='%s', guild_name='%s', joined_at='%s')>" % (
            self.guild_id,
            self.guild_name,
            self.joined_at,
        )
