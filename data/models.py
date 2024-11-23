from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from logger_config import setup_logging

logger = setup_logging()

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


class AlertsChannel(Base):
    __tablename__ = "alerts_channels"

    id = Column(Integer, primary_key=True)
    channel_id = Column(BigInteger)
    channel_name = Column(String)
    guild_id = Column(BigInteger)

    def __repr__(self):
        return (
            "<AlertsChannel(id='%s', channel_id='%s', channel_name='%s', guild_id='%s')>"
            % (
                self.id,
                self.channel_id,
                self.channel_name,
                self.guild_id,
            )
        )


class CommandsChannel(Base):
    __tablename__ = "commands_channels"

    id = Column(Integer, primary_key=True)
    channel_id = Column(BigInteger)
    channel_name = Column(String)
    guild_id = Column(BigInteger)

    def __repr__(self):
        return (
            "<CommandsChannel(id='%s', channel_id='%s', channel_name='%s', guild_id='%s')>"
            % (
                self.id,
                self.channel_id,
                self.channel_name,
                self.guild_id,
            )
        )


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    alert_type = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    last_alerted_at = Column(DateTime, nullable=False)

    def __repr__(self):
        return f"<Alert(id='{self.id}', symbol='{self.symbol}', alert_type='{self.alert_type}', timestamp='{self.timestamp}', last_alerted_at='{self.last_alerted_at}')>"
