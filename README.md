# AI Review & Product Scraper API

This project provides an API for scraping product data, analyzing reviews, storing and retrieving sentiment information,
and performing semantic searches using Pinecone.

---

## Getting Started

### 1. Start Services with Docker using "make"

Before starting, you need to change the name of .env_example file to .env everything will work in mock mode and then:

```bash
   make up
```

To see docker logs:

```bash
   make logs
```

This command launches the following services:

* FastAPI app ([http://localhost:8000](http://localhost:8000))
* PostgreSQL database
* Redis server

---

## 🔧 Makefile Commands

```bash
   make help             # Show help message
```
```bash
   make install          # Install dependencies
```
```bash
   make dev              # Run in development mode
```
```bash
   make test             # Run all tests
```
```bash
   make build            # Build Docker image
```
```bash
   make up               # Start services
```
```bash
   make down             # Stop services
```
```bash
   make logs             # Show logs
```
```bash
   make clean            # Remove volumes, cache, and local storage
```
```bash
   make status           # Check health endpoint
```

---

## 🔧 Manual Testing Using Mock Data

You can test this API manually using mock data generated in `scraper.py`. Here is a step-by-step test guide.

---

## 🔍 API Testing Flow

1. **Scrape Product Data**
   `POST /api/v1/scrape-product`
   Scrapes and ingests a mock product.

```bash
   curl -X 'POST' \
     'http://127.0.0.1:8000/api/v1/scrape-product?url=https%3A%2F%2Fexample.com%2Fsmartphone-galaxy-s23' \
     -H 'accept: application/json' \
     -d ''
```

2. **Get List of All Products**
   `GET /api/v1/products`

```bash
   curl -X 'GET' \
     'http://127.0.0.1:8000/api/v1/products?limit=10&offset=0' \
     -H 'accept: application/json'
```

3. **Get Detailed Product with Reviews**
   `GET /api/v1/products/{product_id}`

```bash
   curl -X 'GET' \
     'http://127.0.0.1:8000/api/v1/products/1' \
     -H 'accept: application/json'
```

4. **Semantic Review Search**
   `GET /api/v1/search-reviews?q=...`

```bash
   curl -X 'GET' \
     'http://localhost:8000/api/v1/search-reviews?query=camera&limit=10&min_similarity=0.1' \
     -H 'accept: application/json'
```

5. **Filter Reviews by Sentiment**
   `GET /api/v1/reviews/by-sentiment/{sentiment}`

Positive sentiment:

```bash
   curl -X 'GET' \
     'http://localhost:8000/api/v1/reviews/by-sentiment/positive?limit=50' \
     -H 'accept: application/json'
```

Negative sentiment:

```bash
   curl -X 'GET' \
     'http://localhost:8000/api/v1/reviews/by-sentiment/negative?limit=50' \
     -H 'accept: application/json'
```

Neutral sentiment:

```bash
   curl -X 'GET' \
     'http://localhost:8000/api/v1/reviews/by-sentiment/neutral?limit=50' \
     -H 'accept: application/json'
```

6. **Sentiment Statistics Analytics**
   `GET /api/v1/analytics/sentiment-stats`

```bash
   curl -X 'GET' \
     'http://localhost:8000/api/v1/analytics/sentiment-stats' \
     -H 'accept: application/json'
```

---

## Developer Notes

* Environment Variables to set in `.env` ready to test in mock mode:

  ```env
   API_HOST=0.0.0.0
   API_PORT=8000
   DATABASE_URL=postgresql://reviews_user:reviews_pass@localhost:5432/reviews_db
   VECTOR_DB_TYPE=pinecone
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=us-west1-gcp-free
   PINECONE_INDEX_NAME=product-reviews
   STORAGE_MODE=mock
   GOOGLE_CLOUD_BUCKET=your-bucket-name
   GOOGLE_APPLICATION_CREDENTIALS=./credentials.json
   LOCAL_STORAGE_PATH=./local_storage
   OPENAI_API_KEY=your_openai_api_key_here
  ```
  
* API routes are all prefixed under `/api/v1/`

---

## Shut Down

```bash
   make down
```

---

## Author

Created for demonstration and local testing purposes with mock and AI-integrated endpoints.
