from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.models.database import Base, get_db


SQLALCHEMY_DATABASE_URL = "sqlite:///./test_system.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

class TestAPI:

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "Product Review Analysis System" in data["message"]
        assert "version" in data
        assert "key_endpoints" in data

    def test_health_check(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert "timestamp" in data

    def test_products_empty_list(self):
        response = client.get("/api/v1/products")
        assert response.status_code == 200
        data = response.json()
        assert "products" in data
        assert "pagination" in data
        assert isinstance(data["products"], list)

    def test_search_reviews_empty(self):
        """Test search functionality with empty database"""
        response = client.get("/api/v1/search-reviews?query=test")
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert data["query"] == "test"
        assert isinstance(data["results"], list)

    def test_sentiment_endpoints(self):
        for sentiment in ["positive", "negative", "neutral"]:
            response = client.get(f"/api/v1/reviews/by-sentiment/{sentiment}")
            assert response.status_code == 200
            data = response.json()
            assert data["sentiment"] == sentiment
            assert "reviews" in data

    def test_invalid_sentiment(self):
        response = client.get("/api/v1/reviews/by-sentiment/invalid")
        assert response.status_code == 400

    def test_analytics_endpoint(self):
        response = client.get("/api/v1/analytics/sentiment-stats")
        assert response.status_code == 200
        data = response.json()

        required_fields = [
            "sentiment_distribution",
            "percentages",
            "average_scores",
            "average_confidence",
            "vector_database"
        ]

        for field in required_fields:
            assert field in data

    def test_products_pagination(self):
        response = client.get("/api/v1/products?limit=5&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert data["pagination"]["limit"] == 5
        assert data["pagination"]["offset"] == 0

    def test_nonexistent_product(self):
        response = client.get("/api/v1/products/999999")
        assert response.status_code == 404

    def test_search_missing_query(self):
        response = client.get("/api/v1/search-reviews")
        assert response.status_code == 422
