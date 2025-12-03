"""
Database models for the Giving Intelligence platform.
Uses SQLAlchemy for ORM with async support.
"""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
    Enum as SQLEnum,
)
from sqlalchemy.ext.asyncio import AsyncAttrs, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
import enum


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all models."""
    pass


# Enums
class CampaignCategory(str, enum.Enum):
    MEDICAL = "medical"
    EDUCATION = "education"
    EMERGENCY = "emergency"
    COMMUNITY = "community"
    ANIMALS = "animals"
    ENVIRONMENT = "environment"
    MEMORIAL = "memorial"
    SPORTS = "sports"
    OTHER = "other"


class DonorStatus(str, enum.Enum):
    ACTIVE = "active"
    ENGAGED = "engaged"
    AT_RISK = "at_risk"
    LAPSED = "lapsed"
    NEW = "new"


class CircleType(str, enum.Enum):
    FAMILY = "family"
    FRIENDS = "friends"
    WORKPLACE = "workplace"
    COMMUNITY = "community"
    ALUMNI = "alumni"


# Models
class Donor(Base):
    """Donor model."""
    
    __tablename__ = "donors"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(255))
    
    # Location
    city: Mapped[Optional[str]] = mapped_column(String(100))
    state: Mapped[Optional[str]] = mapped_column(String(100))
    country: Mapped[str] = mapped_column(String(100), default="US")
    latitude: Mapped[Optional[float]] = mapped_column(Float)
    longitude: Mapped[Optional[float]] = mapped_column(Float)
    
    # Affiliations
    employer: Mapped[Optional[str]] = mapped_column(String(255))
    employer_id: Mapped[Optional[str]] = mapped_column(String(36))
    
    # Status
    status: Mapped[DonorStatus] = mapped_column(
        SQLEnum(DonorStatus), default=DonorStatus.NEW
    )
    
    # Metrics
    total_donated: Mapped[float] = mapped_column(Float, default=0.0)
    donation_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Profile data (JSON for flexibility)
    profile_data: Mapped[Optional[dict]] = mapped_column(JSON)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    first_donation_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    last_donation_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Relationships
    donations: Mapped[List["Donation"]] = relationship(back_populates="donor")
    
    def __repr__(self) -> str:
        return f"<Donor {self.display_name} ({self.id})>"


class Campaign(Base):
    """Campaign model."""
    
    __tablename__ = "campaigns"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    title: Mapped[str] = mapped_column(String(500))
    description: Mapped[str] = mapped_column(Text)
    
    # Category
    category: Mapped[CampaignCategory] = mapped_column(
        SQLEnum(CampaignCategory), default=CampaignCategory.OTHER
    )
    subcategories: Mapped[Optional[list]] = mapped_column(JSON)
    
    # Location
    city: Mapped[Optional[str]] = mapped_column(String(100))
    state: Mapped[Optional[str]] = mapped_column(String(100))
    country: Mapped[str] = mapped_column(String(100), default="US")
    latitude: Mapped[Optional[float]] = mapped_column(Float)
    longitude: Mapped[Optional[float]] = mapped_column(Float)
    
    # Funding
    goal_amount: Mapped[float] = mapped_column(Float, default=0.0)
    raised_amount: Mapped[float] = mapped_column(Float, default=0.0)
    donor_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Social
    share_count: Mapped[int] = mapped_column(Integer, default=0)
    comment_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Organizer
    organizer_id: Mapped[str] = mapped_column(String(36), index=True)
    organizer_name: Mapped[str] = mapped_column(String(255))
    organizer_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_recurring_need: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Analysis data (JSON for flexibility)
    analysis_data: Mapped[Optional[dict]] = mapped_column(JSON)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    
    # Relationships
    donations: Mapped[List["Donation"]] = relationship(back_populates="campaign")
    updates: Mapped[List["CampaignUpdate"]] = relationship(back_populates="campaign")
    
    def __repr__(self) -> str:
        return f"<Campaign {self.title[:30]} ({self.id})>"


class Donation(Base):
    """Donation model."""
    
    __tablename__ = "donations"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    # Foreign keys
    donor_id: Mapped[str] = mapped_column(String(36), ForeignKey("donors.id"), index=True)
    campaign_id: Mapped[str] = mapped_column(String(36), ForeignKey("campaigns.id"), index=True)
    
    # Amount
    amount: Mapped[float] = mapped_column(Float)
    
    # Options
    is_anonymous: Mapped[bool] = mapped_column(Boolean, default=False)
    is_recurring: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Source
    source: Mapped[str] = mapped_column(String(50), default="direct")  # direct, shared, search, email
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    donor: Mapped["Donor"] = relationship(back_populates="donations")
    campaign: Mapped["Campaign"] = relationship(back_populates="donations")
    
    def __repr__(self) -> str:
        return f"<Donation ${self.amount} from {self.donor_id} to {self.campaign_id}>"


class CampaignUpdate(Base):
    """Campaign update model."""
    
    __tablename__ = "campaign_updates"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    campaign_id: Mapped[str] = mapped_column(String(36), ForeignKey("campaigns.id"), index=True)
    
    update_type: Mapped[str] = mapped_column(String(50))  # milestone, progress, completion, gratitude
    content: Mapped[str] = mapped_column(Text)
    media_url: Mapped[Optional[str]] = mapped_column(String(500))
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    campaign: Mapped["Campaign"] = relationship(back_populates="updates")


class GivingCircle(Base):
    """Giving circle model."""
    
    __tablename__ = "giving_circles"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    
    circle_type: Mapped[CircleType] = mapped_column(
        SQLEnum(CircleType), default=CircleType.FRIENDS
    )
    
    # Pool
    pool_balance: Mapped[float] = mapped_column(Float, default=0.0)
    total_contributed: Mapped[float] = mapped_column(Float, default=0.0)
    total_distributed: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Settings
    voting_threshold: Mapped[float] = mapped_column(Float, default=0.5)
    min_contribution: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Stats
    member_count: Mapped[int] = mapped_column(Integer, default=0)
    campaigns_funded: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    
    # Members and other data (JSON for flexibility)
    members_data: Mapped[Optional[dict]] = mapped_column(JSON)
    
    def __repr__(self) -> str:
        return f"<GivingCircle {self.name} ({self.id})>"


class CircleMembership(Base):
    """Circle membership model."""
    
    __tablename__ = "circle_memberships"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    circle_id: Mapped[str] = mapped_column(String(36), ForeignKey("giving_circles.id"), index=True)
    donor_id: Mapped[str] = mapped_column(String(36), ForeignKey("donors.id"), index=True)
    
    role: Mapped[str] = mapped_column(String(50), default="member")  # admin, member, contributor, viewer
    
    total_contributed: Mapped[float] = mapped_column(Float, default=0.0)
    contribution_count: Mapped[int] = mapped_column(Integer, default=0)
    
    joined_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_contribution_at: Mapped[Optional[datetime]] = mapped_column(DateTime)


class SocialConnection(Base):
    """Social connection model."""
    
    __tablename__ = "social_connections"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("donors.id"), index=True)
    connected_user_id: Mapped[str] = mapped_column(String(36), ForeignKey("donors.id"), index=True)
    
    connection_type: Mapped[str] = mapped_column(String(50))  # friend, family, colleague, etc.
    strength: Mapped[float] = mapped_column(Float, default=0.5)
    is_mutual: Mapped[bool] = mapped_column(Boolean, default=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class EngagementTouchpoint(Base):
    """Engagement touchpoint model."""
    
    __tablename__ = "engagement_touchpoints"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    donor_id: Mapped[str] = mapped_column(String(36), ForeignKey("donors.id"), index=True)
    
    engagement_type: Mapped[str] = mapped_column(String(50))
    channel: Mapped[str] = mapped_column(String(50), default="email")
    
    title: Mapped[str] = mapped_column(String(255))
    message: Mapped[str] = mapped_column(Text)
    
    # Delivery
    scheduled_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Engagement tracking
    opened: Mapped[bool] = mapped_column(Boolean, default=False)
    clicked: Mapped[bool] = mapped_column(Boolean, default=False)
    converted: Mapped[bool] = mapped_column(Boolean, default=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# Database setup functions
async def create_tables(database_url: str):
    """Create all tables."""
    engine = create_async_engine(database_url, echo=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine


async def get_session(database_url: str):
    """Get an async session."""
    engine = create_async_engine(database_url)
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    return async_session()





