from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime, timezone
from app.config import settings

Base = declarative_base()


class Product(Base):
    """Product model - stores information about scraped products"""
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    url = Column(String(1000), unique=True, nullable=False, index=True)
    images = Column(JSON)
    created_at = Column(DateTime, default=datetime.now(timezone.utc).isoformat())
    updated_at = Column(DateTime,
                        default=datetime.now(timezone.utc).isoformat(),
                        onupdate=datetime.now(timezone.utc).isoformat())

    reviews = relationship("Review", back_populates="product", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Product(id={self.id}, title='{self.title[:50]}...')>"


class Review(Base):
    """Review model - stores individual product reviews"""
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False, index=True)
    text = Column(Text, nullable=False)
    sentiment = Column(String(20))
    sentiment_score = Column(Float)
    confidence = Column(Float)
    vector_id = Column(String(50))
    created_at = Column(DateTime, default=datetime.now(timezone.utc).isoformat())

    product = relationship("Product", back_populates="reviews")
    media = relationship("ReviewMedia", back_populates="review", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Review(id={self.id}, sentiment='{self.sentiment}', text='{self.text[:30]}...')>"

class ReviewMedia(Base):
    """Review media model - stores images/videos from reviews"""
    __tablename__ = "review_media"

    id = Column(Integer, primary_key=True, index=True)
    review_id = Column(Integer, ForeignKey("reviews.id"), nullable=False, index=True)
    media_type = Column(String(20), nullable=False)
    original_url = Column(String(1000))
    cloud_url = Column(String(1000))
    file_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.now(timezone.utc).isoformat())

    review = relationship("Review", back_populates="media")

    def __repr__(self):
        return f"<ReviewMedia(id={self.id}, type='{self.media_type}', review_id={self.review_id})>"

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all database tables"""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_sync():
    """Synchronous version of get_db for testing"""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()
