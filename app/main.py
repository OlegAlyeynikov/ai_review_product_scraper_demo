from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.models.database import create_tables
from app.api.routes import router
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    logger.info("Starting Product Review Analysis System...")
    create_tables()
    logger.info("   - Database initialized")
    logger.info("   - Services loaded:")
    logger.info("   - Web Scraper: MockProductScraper (ready)")
    logger.info("   - Sentiment Analysis: Multi-method analyzer (ready)")
    logger.info("   - Storage: Mock storage (ready)")
    logger.info("   - Vector Database: TF-IDF based (ready)")
    logger.info("   - API Documentation available at /docs")

    yield

    logger.info("Shutting down Product Review Analysis System...")

app = FastAPI(
    title="Product Review Analysis System",
    description="""
    A comprehensive system for scraping, analyzing, and searching product reviews.

    ## Features
    - **Web Scraping**: Extract product information and reviews from web pages
    - **Sentiment Analysis**: Analyze review sentiment using AI and fallback methods
    - **Cloud Storage**: Store images and media files
    - **Vector Search**: Semantic search of reviews using TF-IDF
    - **Analytics**: Comprehensive sentiment statistics and insights

    ## Workflow
    1. Use `/scrape-product` to start scraping a product page
    2. Check `/products` to see scraped products
    3. Use `/search-reviews` for semantic search
    4. Get analytics with `/analytics/sentiment-stats`
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["reviews"])

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Product Review Analysis System",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs",
        "api_prefix": "/api/v1",
        "key_endpoints": {
            "scrape_product": "/api/v1/scrape-product",
            "list_products": "/api/v1/products",
            "search_reviews": "/api/v1/search-reviews",
            "sentiment_stats": "/api/v1/analytics/sentiment-stats",
            "health_check": "/api/v1/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    from app.config import settings

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )