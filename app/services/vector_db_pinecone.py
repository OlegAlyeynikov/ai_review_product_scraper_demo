import logging
from typing import List, Dict, Optional
import numpy as np
import threading
from datetime import datetime

try:
    import pinecone
    from sentence_transformers import SentenceTransformer
    PINECONE_AVAILABLE = True
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("Pinecone and SentenceTransformers imported successfully")
except ImportError as e:
    PINECONE_AVAILABLE = False
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(f"Pinecone or SentenceTransformers not available: {e}")

from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PineconeVectorDatabase:
    """
    Pinecone-based vector database for semantic search of reviews
    Uses SentenceTransformers for embeddings and Pinecone for storage/search
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        """Initialize Pinecone vector database"""
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone not available. Install with: pip install pinecone-client")

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("SentenceTransformers not available. Install with: pip install sentence-transformers")

        self.model_name = model_name
        self.dimension = dimension
        self.index_name = getattr(settings, 'PINECONE_INDEX_NAME', 'product-reviews')

        self.mock_vectors = {}
        self.mock_metadata = {}
        self.pinecone_enabled = False

        try:
            logger.info(f"Loading SentenceTransformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info("SentenceTransformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
            raise e

        self._initialize_pinecone()

        self._lock = threading.Lock()

        logger.info(f"Pinecone vector database initialized with index: {self.index_name}")

    def _initialize_pinecone(self):
        """Initialize Pinecone client and index"""
        try:
            api_key = getattr(settings, 'PINECONE_API_KEY', None)
            environment = getattr(settings, 'PINECONE_ENVIRONMENT', 'us-west1-gcp-free')

            if not api_key or api_key == "your_pinecone_api_key_here":
                logger.warning("Pinecone API key not provided - using mock mode")
                self.pinecone_enabled = False
                return

            pinecone.init(api_key=api_key, environment=environment)

            if self.index_name not in pinecone.list_indexes():
                logger.info(f"Creating Pinecone index: {self.index_name}")
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine"
                )
                logger.info(f"Pinecone index created: {self.index_name}")

            self.index = pinecone.Index(self.index_name)
            self.pinecone_enabled = True

            logger.info(f"Connected to Pinecone index: {self.index_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            logger.warning("Using mock mode")
            self.pinecone_enabled = False

    def add_review(self, review_id: int, text: str, product_id: int, 
                   sentiment: str, extra_metadata: Optional[Dict] = None) -> str:
        """Add a review to the Pinecone vector database"""
        try:
            with self._lock:
                embedding = self._generate_embedding(text)
                if embedding is None:
                    raise Exception("Failed to generate embedding")

                vector_id = f"review_{review_id}_{int(datetime.utcnow().timestamp())}"

                metadata = {
                    'review_id': review_id,
                    'text': text[:1000],
                    'product_id': product_id,
                    'sentiment': sentiment,
                    'created_at': datetime.utcnow().isoformat(),
                    **(extra_metadata or {})
                }

                if self.pinecone_enabled:
                    self.index.upsert(
                        vectors=[(vector_id, embedding.tolist(), metadata)]
                    )
                else:
                    self.mock_vectors[vector_id] = embedding
                    self.mock_metadata[vector_id] = metadata

                logger.info(f"Added review {review_id} as vector {vector_id}")
                return vector_id

        except Exception as e:
            logger.error(f"Failed to add review {review_id}: {e}")
            raise e

    def search_similar_reviews(self, query_text: str, k: int = 10, 
                              product_id: Optional[int] = None,
                              sentiment_filter: Optional[str] = None,
                              min_similarity: float = 0.1) -> List[Dict]:
        """Search for similar reviews using Pinecone"""
        try:
            with self._lock:
                query_embedding = self._generate_embedding(query_text)
                if query_embedding is None:
                    return []

                if self.pinecone_enabled:
                    search_results = self._search_pinecone(
                        query_embedding, k * 2, product_id, sentiment_filter
                    )
                else:
                    search_results = self._search_mock(
                        query_embedding, k * 2, product_id, sentiment_filter
                    )

                results = []
                for result in search_results:
                    similarity_score = result.get('score', 0.0)

                    if similarity_score < min_similarity:
                        continue

                    metadata = result.get('metadata', {})
                    metadata['vector_id'] = result.get('id', '')
                    metadata['similarity_score'] = similarity_score

                    results.append(metadata)

                    if len(results) >= k:
                        break

                logger.info(f"Found {len(results)} similar reviews for query: '{query_text[:50]}...'")
                return results

        except Exception as e:
            logger.error(f"Search failed for query '{query_text}': {e}")
            return []

    def _search_pinecone(self, query_embedding: np.ndarray, k: int, 
                        product_id: Optional[int], sentiment_filter: Optional[str]) -> List[Dict]:
        """Search in Pinecone index"""
        try:
            filter_dict = {}
            if product_id:
                filter_dict['product_id'] = product_id
            if sentiment_filter:
                filter_dict['sentiment'] = sentiment_filter

            query_results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=k,
                filter=filter_dict if filter_dict else None,
                include_metadata=True
            )

            results = []
            for match in query_results.matches:
                results.append({
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata
                })

            return results

        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return []

    def _search_mock(self, query_embedding: np.ndarray, k: int,
                    product_id: Optional[int], sentiment_filter: Optional[str]) -> List[Dict]:
        """Search in mock storage using cosine similarity"""
        try:
            if not self.mock_vectors:
                return []

            results = []

            for vector_id, embedding in self.mock_vectors.items():
                metadata = self.mock_metadata.get(vector_id, {})

                if product_id and metadata.get('product_id') != product_id:
                    continue
                if sentiment_filter and metadata.get('sentiment') != sentiment_filter:
                    continue

                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )

                results.append({
                    'id': vector_id,
                    'score': float(similarity),
                    'metadata': metadata
                })

            results.sort(key=lambda x: x['score'], reverse=True)

            return results[:k]

        except Exception as e:
            logger.error(f"Mock search failed: {e}")
            return []

    def get_reviews_by_sentiment(self, sentiment: str, product_id: Optional[int] = None,
                                limit: int = 50) -> List[Dict]:
        """Get reviews filtered by sentiment"""
        try:
            if self.pinecone_enabled:
                filter_dict = {'sentiment': sentiment}
                if product_id:
                    filter_dict['product_id'] = product_id

                dummy_embedding = np.zeros(self.dimension)
                results = self._search_pinecone(dummy_embedding, limit, product_id, sentiment)

                return [result['metadata'] for result in results]

            else:
                results = []
                for vector_id, metadata in self.mock_metadata.items():
                    if metadata.get('sentiment') == sentiment:
                        if product_id is None or metadata.get('product_id') == product_id:
                            metadata_copy = metadata.copy()
                            metadata_copy['vector_id'] = vector_id
                            results.append(metadata_copy)

                            if len(results) >= limit:
                                break

                return results

        except Exception as e:
            logger.error(f"Failed to get reviews by sentiment: {e}")
            return []

    def remove_review(self, vector_id: str) -> bool:
        """Remove a review from the vector database"""
        try:
            with self._lock:
                if self.pinecone_enabled:
                    self.index.delete(ids=[vector_id])
                else:
                    if vector_id in self.mock_vectors:
                        del self.mock_vectors[vector_id]
                        del self.mock_metadata[vector_id]

                logger.info(f"Removed review vector {vector_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to remove vector {vector_id}: {e}")
            return False

    def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            if self.pinecone_enabled:
                index_stats = self.index.describe_index_stats()
                total_vectors = index_stats.total_vector_count

                return {
                    'backend_type': 'pinecone',
                    'total_vectors': total_vectors,
                    'index_name': self.index_name,
                    'dimension': self.dimension,
                    'model_name': self.model_name,
                    'pinecone_enabled': True
                }
            else:
                sentiment_counts = {}
                for metadata in self.mock_metadata.values():
                    sentiment = metadata.get('sentiment', 'unknown')
                    sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

                return {
                    'backend_type': 'pinecone_mock',
                    'total_vectors': len(self.mock_vectors),
                    'index_name': self.index_name,
                    'dimension': self.dimension,
                    'model_name': self.model_name,
                    'pinecone_enabled': False,
                    'sentiment_distribution': sentiment_counts
                }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'backend_type': 'pinecone', 'error': str(e)}

    def get_total_vectors(self) -> int:
        """Get total number of vectors"""
        try:
            if self.pinecone_enabled:
                stats = self.index.describe_index_stats()
                return stats.total_vector_count
            else:
                return len(self.mock_vectors)
        except Exception as e:
            logger.error(f"Failed to get total vectors: {e}")
            return 0

    def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text using SentenceTransformers"""
        try:
            text = text.strip()
            if not text:
                text = "empty text"

            embedding = self.model.encode([text])

            return embedding[0].astype('float32')

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

try:
    if PINECONE_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE:
        vector_db = PineconeVectorDatabase()
        logger.info("Pinecone vector database initialized successfully")
    else:

        class MockVectorDB:
            def add_review(self, *args, **kwargs): return "mock_id"
            def search_similar_reviews(self, *args, **kwargs): return []
            def get_reviews_by_sentiment(self, *args, **kwargs): return []
            def get_stats(self): return {"backend_type": "mock", "total_vectors": 0}
            def get_total_vectors(self): return 0
            def remove_review(self, *args): return True

        vector_db = MockVectorDB()
        logger.warning("Using mock vector database")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")

    class MockVectorDB:
        def add_review(self, *args, **kwargs): return "mock_id"
        def search_similar_reviews(self, *args, **kwargs): return []
        def get_reviews_by_sentiment(self, *args, **kwargs): return []
        def get_stats(self): return {"backend_type": "mock", "total_vectors": 0}
        def get_total_vectors(self): return 0
        def remove_review(self, *args): return True

    vector_db = MockVectorDB()
    logger.warning("Using mock vector database")
