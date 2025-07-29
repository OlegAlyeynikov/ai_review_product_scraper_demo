from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, timezone
import logging

from app.models.database import get_db, Product, Review, ReviewMedia
from app.services.scraper import MockProductScraper
from app.services.sentiment import sentiment_service
from app.services.storage import storage_service
from app.services.vector_db_pinecone import vector_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


router = APIRouter()

scraper = MockProductScraper()

@router.post("/scrape-product")
async def scrape_product(
        url: str = Query(..., description="URL of the product page to scrape"),
        background_tasks: BackgroundTasks = BackgroundTasks(),
        db: Session = Depends(get_db)
):
    """
    Start scraping a product page for reviews
    """
    try:

        existing_product = db.query(Product).filter(Product.url == url).first()
        if existing_product:
            return {
                "message": "Product already exists",
                "product_id": existing_product.id,
                "status": "exists"
            }

        background_tasks.add_task(process_product_scraping, url, db)

        return {
            "message": "Product scraping started",
            "url": url,
            "status": "processing",
            "note": "Check /products endpoint to see results"
        }

    except Exception as e:
        logger.error(f"Error starting product scraping: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_product_scraping(url: str, db: Session):
    """
    Background task to process product scraping
    This runs asynchronously and doesn't block the API response
    """
    try:
        logger.info(f"Starting background processing for: {url}")

        async with scraper:
            product_data = await scraper.scrape_product_page(url)

            logger.info(f"Scraped product: {product_data['title']}")
            logger.info(f"Found {len(product_data['reviews'])} reviews")

            cloud_images = await storage_service.upload_multiple_images(
                product_data['images'],
                folder="products"
            )

            product = Product(
                title=product_data['title'],
                url=url,
                images=cloud_images
            )
            db.add(product)
            db.commit()
            db.refresh(product)

            logger.info(f"Created product with ID: {product.id}")

            for i, review_data in enumerate(product_data['reviews']):
                try:
                    sentiment_result = await sentiment_service.analyze_sentiment(review_data['text'])

                    review = Review(
                        product_id=product.id,
                        text=review_data['text'],
                        sentiment=sentiment_result.sentiment,
                        sentiment_score=sentiment_result.score,
                        confidence=sentiment_result.confidence
                    )
                    db.add(review)
                    db.commit()
                    db.refresh(review)

                    vector_id = vector_db.add_review(
                        review_id=review.id,
                        text=review.text,
                        product_id=product.id,
                        sentiment=review.sentiment,
                        extra_metadata={'confidence': sentiment_result.confidence}
                    )

                    review.vector_id = vector_id
                    db.commit()

                    if review_data.get('media'):
                        cloud_media = await storage_service.upload_multiple_images(
                            review_data['media'],
                            folder="reviews"
                        )

                        for original_url, cloud_url in zip(review_data['media'], cloud_media):
                            media = ReviewMedia(
                                review_id=review.id,
                                media_type="image",
                                original_url=original_url,
                                cloud_url=cloud_url
                            )
                            db.add(media)

                        db.commit()

                    logger.info(f"Processed review {i+1}/{len(product_data['reviews'])}")

                except Exception as e:
                    logger.error(f"Error processing review {i+1}: {e}")
                    continue

            logger.info(f"Completed processing product: {product.id}")

    except Exception as e:
        logger.error(f"Error in background processing: {e}")

@router.get("/products/{product_id}")
async def get_product(product_id: int, db: Session = Depends(get_db)):
    """
    Get detailed information about a product including all reviews
    """
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    reviews = db.query(Review).filter(Review.product_id == product_id).all()

    review_list = []
    for review in reviews:
        media = db.query(ReviewMedia).filter(ReviewMedia.review_id == review.id).all()

        review_list.append({
            "id": review.id,
            "text": review.text,
            "sentiment": review.sentiment,
            "sentiment_score": review.sentiment_score,
            "confidence": review.confidence,
            "vector_id": review.vector_id,
            "media": [{"type": m.media_type, "url": m.cloud_url} for m in media],
            "created_at": review.created_at
        })

    return {
        "product": {
            "id": product.id,
            "title": product.title,
            "url": product.url,
            "images": product.images,
            "created_at": product.created_at
        },
        "reviews": review_list,
        "summary": {
            "total_reviews": len(review_list),
            "positive_reviews": len([r for r in review_list if r["sentiment"] == "positive"]),
            "negative_reviews": len([r for r in review_list if r["sentiment"] == "negative"]),
            "neutral_reviews": len([r for r in review_list if r["sentiment"] == "neutral"])
        }
    }

@router.get("/products")
async def get_products(
        limit: int = Query(10, description="Maximum number of products to return"),
        offset: int = Query(0, description="Number of products to skip"),
        db: Session = Depends(get_db)
):
    """
    Get list of all products with basic information
    """

    products = db.query(Product).offset(offset).limit(limit).all()

    result = []
    for product in products:
        total_reviews = db.query(Review).filter(Review.product_id == product.id).count()
        positive_reviews = db.query(Review).filter(
            Review.product_id == product.id,
            Review.sentiment == "positive"
        ).count()
        negative_reviews = db.query(Review).filter(
            Review.product_id == product.id,
            Review.sentiment == "negative"
        ).count()

        result.append({
            "id": product.id,
            "title": product.title,
            "url": product.url,
            "images": product.images[:3] if product.images else [],
            "review_summary": {
                "total": total_reviews,
                "positive": positive_reviews,
                "negative": negative_reviews,
                "neutral": total_reviews - positive_reviews - negative_reviews
            },
            "created_at": product.created_at
        })

    total_count = db.query(Product).count()

    return {
        "products": result,
        "pagination": {
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_count
        }
    }

@router.get("/search-reviews")
async def search_reviews(
        query: str = Query(..., description="Search query text"),
        product_id: Optional[int] = Query(None, description="Filter by product ID"),
        sentiment: Optional[str] = Query(None, description="Filter by sentiment (positive/negative/neutral)"),
        limit: int = Query(10, description="Maximum number of results"),
        min_similarity: float = Query(0.1, description="Minimum similarity threshold"),
        db: Session = Depends(get_db)
):
    """
    Search for reviews using semantic similarity
    """
    try:
        if sentiment and sentiment not in ['positive', 'negative', 'neutral']:
            raise HTTPException(status_code=400, detail="Invalid sentiment. Use: positive, negative, or neutral")

        similar_reviews = vector_db.search_similar_reviews(
            query_text=query,
            k=limit,
            product_id=product_id,
            sentiment_filter=sentiment,
            min_similarity=min_similarity
        )

        enriched_results = []
        for review_data in similar_reviews:
            review = db.query(Review).filter(Review.id == review_data['review_id']).first()
            if not review:
                continue

            product = db.query(Product).filter(Product.id == review.product_id).first()

            media = db.query(ReviewMedia).filter(ReviewMedia.review_id == review.id).all()

            enriched_results.append({
                "review_id": review.id,
                "text": review.text,
                "sentiment": review.sentiment,
                "sentiment_score": review.sentiment_score,
                "similarity_score": review_data['similarity_score'],
                "product": {
                    "id": product.id,
                    "title": product.title
                } if product else None,
                "media": [{"type": m.media_type, "url": m.cloud_url} for m in media],
                "created_at": review.created_at
            })

        return {
            "query": query,
            "filters": {
                "product_id": product_id,
                "sentiment": sentiment,
                "min_similarity": min_similarity
            },
            "results": enriched_results,
            "total_found": len(enriched_results)
        }

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reviews/by-sentiment/{sentiment}")
async def get_reviews_by_sentiment(
        sentiment: str,
        product_id: Optional[int] = Query(None, description="Filter by product ID"),
        limit: int = Query(50, description="Maximum number of results"),
        db: Session = Depends(get_db)
):
    """
    Get reviews filtered by sentiment
    """
    if sentiment not in ['positive', 'negative', 'neutral']:
        raise HTTPException(status_code=400, detail="Invalid sentiment. Use: positive, negative, or neutral")

    query = db.query(Review).filter(Review.sentiment == sentiment)
    if product_id:
        query = query.filter(Review.product_id == product_id)

    reviews = query.limit(limit).all()

    result = []
    for review in reviews:
        product = db.query(Product).filter(Product.id == review.product_id).first()

        media = db.query(ReviewMedia).filter(ReviewMedia.review_id == review.id).all()

        result.append({
            "review_id": review.id,
            "text": review.text,
            "sentiment_score": review.sentiment_score,
            "confidence": review.confidence,
            "product": {
                "id": product.id,
                "title": product.title
            } if product else None,
            "media": [{"type": m.media_type, "url": m.cloud_url} for m in media],
            "created_at": review.created_at
        })

    return {
        "sentiment": sentiment,
        "filters": {"product_id": product_id},
        "reviews": result,
        "total": len(result)
    }

@router.get("/analytics/sentiment-stats")
async def get_sentiment_stats(
        product_id: Optional[int] = Query(None, description="Filter by product ID"),
        db: Session = Depends(get_db)
):
    """
    Get sentiment analysis statistics
    """
    try:
        query = db.query(Review)
        if product_id:
            query = query.filter(Review.product_id == product_id)

        reviews = query.all()

        stats = {
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'total': len(reviews)
        }

        score_sum = {'positive': 0, 'negative': 0, 'neutral': 0}
        confidence_sum = {'positive': 0, 'negative': 0, 'neutral': 0}

        for review in reviews:
            sentiment = review.sentiment
            stats[sentiment] += 1
            score_sum[sentiment] += review.sentiment_score or 0
            confidence_sum[sentiment] += review.confidence or 0

        avg_scores = {}
        avg_confidence = {}
        for sentiment in ['positive', 'negative', 'neutral']:
            count = stats[sentiment]
            if count > 0:
                avg_scores[sentiment] = round(score_sum[sentiment] / count, 3)
                avg_confidence[sentiment] = round(confidence_sum[sentiment] / count, 3)
            else:
                avg_scores[sentiment] = 0
                avg_confidence[sentiment] = 0

        vector_stats = vector_db.get_stats()

        product_info = None
        if product_id:
            product = db.query(Product).filter(Product.id == product_id).first()
            if product:
                product_info = {
                    "id": product.id,
                    "title": product.title
                }

        return {
            "product": product_info,
            "sentiment_distribution": stats,
            "percentages": {
                sentiment: round((count / stats['total'] * 100), 1) if stats['total'] > 0 else 0
                for sentiment, count in stats.items() if sentiment != 'total'
            },
            "average_scores": avg_scores,
            "average_confidence": avg_confidence,
            "vector_database": vector_stats
        }

    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint to verify all services are working
    """
    try:
        db_status = "connected"
        try:
            from sqlalchemy import text
            db.execute(text("SELECT 1"))
        except Exception as e:
            db_status = f"error: {str(e)}"

        vector_status = f"ready ({vector_db.get_total_vectors()} vectors)"

        storage_stats = storage_service.get_stats()
        storage_status = f"{storage_stats.get('mode', 'unknown')} mode"

        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": {
                "database": db_status,
                "vector_database": vector_status,
                "storage": storage_status,
                "sentiment_analysis": "ready"
            },
            "version": "1.0.0"
        }

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
