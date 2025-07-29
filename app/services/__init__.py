from .scraper import ProductScraper, MockProductScraper

try:
    from .vector_db_pinecone import vector_db
    print("Using Pinecone vector database")
except ImportError as e:
    print(f"Failed to import Pinecone: {e}")
    print("Install with: pip install pinecone-client sentence-transformers")

    class MockVectorDB:
        def add_review(self, *args, **kwargs): return "mock_id"
        def search_similar_reviews(self, *args, **kwargs): return []
        def get_reviews_by_sentiment(self, *args, **kwargs): return []
        def get_stats(self): return {"backend_type": "mock", "total_vectors": 0}
        def get_total_vectors(self): return 0

    vector_db = MockVectorDB()
    print("Using minimal mock vector database")

__all__ = ["ProductScraper", "MockProductScraper", "vector_db"]
