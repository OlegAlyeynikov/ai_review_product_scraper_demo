import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import time
from urllib.parse import urljoin
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductScraper:
    """Web scraper for extracting product information and reviews"""

    def __init__(self, delay: float = 1.0):
        """
        Initialize scraper

        Args:
            delay: Delay between requests in seconds
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        pass

    async def scrape_product_page(self, url: str) -> Dict:
        """
        Scrape a product page and extract product info and reviews

        Args:
            url: URL of the product page to scrape

        Returns:
            Dictionary containing product data and reviews
        """
        try:
            logger.info(f"Scraping product page: {url}")

            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            product_data = {
                'title': self._extract_title(soup, url),
                'images': self._extract_images(soup, url),
                'reviews': self._extract_reviews(soup, url)
            }

            logger.info(f"Extracted product: {product_data['title']}")
            logger.info(f"Found {len(product_data['reviews'])} reviews")

            time.sleep(self.delay)

            return product_data

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            raise Exception(f"Failed to scrape product page: {str(e)}")

    def _extract_title(self, soup: BeautifulSoup, base_url: str) -> str:
        """Extract product title from page"""
        title_selectors = [
            'h1',
            '.product-title',
            '.product-name',
            '[data-testid="product-title"]',
            '.pdp-product-name',
            '.x-item-title-label',
            '#product-title',
            '.product_title'
        ]

        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text(strip=True)
                if len(title) > 5:
                    return title

        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)

        return "Unknown Product"

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract product images"""
        images = []

        img_selectors = [
            '.product-images img',
            '.product-gallery img',
            '.gallery img',
            '[data-testid="product-image"]',
            '.pdp-image img',
            '.product-photo img'
        ]

        for selector in img_selectors:
            elements = soup.select(selector)
            for img in elements:
                src = img.get('src') or img.get('data-src') or img.get('data-lazy')
                if src and self._is_valid_image_url(src):

                    absolute_url = urljoin(base_url, src)
                    if absolute_url not in images:
                        images.append(absolute_url)
        return images[:10]

    def _extract_reviews(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Extract reviews from page"""
        reviews = []

        review_selectors = [
            '.review',
            '.comment',
            '.user-review',
            '[data-testid="review"]',
            '.review-item',
            '.customer-review',
            '.review-content'
        ]

        for selector in review_selectors:
            elements = soup.select(selector)
            for element in elements:
                review_text = self._extract_review_text(element)
                if review_text and len(review_text) > 10:
                    review_data = {
                        'text': review_text,
                        'media': self._extract_review_media(element, base_url)
                    }
                    reviews.append(review_data)

        unique_reviews = []
        seen_texts = set()
        for review in reviews:
            if review['text'] not in seen_texts:
                unique_reviews.append(review)
                seen_texts.add(review['text'])

        return unique_reviews[:50]

    def _extract_review_text(self, element) -> Optional[str]:
        """Extract text from a review element"""
        text_selectors = [
            '.review-text',
            '.comment-text',
            '.review-body',
            '.review-content',
            'p',
            '.text'
        ]

        for selector in text_selectors:
            text_element = element.select_one(selector)
            if text_element:
                text = text_element.get_text(strip=True)
                if len(text) > 10:
                    return text

        text = element.get_text(strip=True)
        return text if len(text) > 10 else None

    def _extract_review_media(self, element, base_url: str) -> List[str]:
        """Extract media (images) from review element"""
        media = []
        images = element.select('img')

        for img in images:
            src = img.get('src') or img.get('data-src')
            if src and self._is_valid_image_url(src):
                absolute_url = urljoin(base_url, src)
                media.append(absolute_url)

        return media[:5]

    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL points to a valid image"""
        if not url or len(url) < 10:
            return False

        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
        url_lower = url.lower()

        if any(url_lower.endswith(ext) for ext in image_extensions):
            return True

        image_indicators = ['image', 'img', 'photo', 'picture', 'pic']
        if any(indicator in url_lower for indicator in image_indicators):
            return True

        return False

class MockProductScraper:
    """Mock scraper that returns sample data for testing"""

    def __init__(self):
        self.mock_products = {
            "smartphone": {
                "title": "Samsung Galaxy S23 Ultra",
                "images": [
                    "https://example.com/galaxy-s23-front.jpg",
                    "https://example.com/galaxy-s23-back.jpg",
                    "https://example.com/galaxy-s23-side.jpg"
                ],
                "reviews": [
                    {
                        "text": "Amazing phone! The camera quality is outstanding and battery lasts all day. Highly recommended!",
                        "media": ["https://example.com/review-photo-1.jpg"]
                    },
                    {
                        "text": "Good performance but the price is too high. The design feels slippery in hands.",
                        "media": []
                    },
                    {
                        "text": "Disappointed with this purchase. Battery drains quickly and the screen has issues.",
                        "media": ["https://example.com/review-photo-2.jpg"]
                    },
                    {
                        "text": "Perfect smartphone for professionals. Fast processing and excellent display quality.",
                        "media": []
                    },
                    {
                        "text": "Average product. Nothing special but does the job. Camera is decent.",
                        "media": []
                    }
                ]
            },
            "laptop": {
                "title": "MacBook Pro 16-inch M3",
                "images": [
                    "https://example.com/macbook-closed.jpg",
                    "https://example.com/macbook-open.jpg"
                ],
                "reviews": [
                    {
                        "text": "Excellent laptop for development work. The M3 chip is incredibly fast!",
                        "media": []
                    },
                    {
                        "text": "Too expensive for what it offers. You can get better value elsewhere.",
                        "media": []
                    },
                    {
                        "text": "Perfect build quality and amazing screen. Worth the investment.",
                        "media": ["https://example.com/macbook-review.jpg"]
                    }
                ]
            }
        }

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        pass

    async def scrape_product_page(self, url: str) -> Dict:
        """Return mock product data based on URL"""
        logger.info(f"Mock scraping: {url}")

        url_lower = url.lower()
        if "phone" in url_lower or "smartphone" in url_lower or "galaxy" in url_lower:
            return self.mock_products["smartphone"]
        elif "laptop" in url_lower or "macbook" in url_lower or "computer" in url_lower:
            return self.mock_products["laptop"]
        else:
            return self.mock_products["smartphone"]
