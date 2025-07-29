import asyncio
import logging
from typing import Dict, List, Optional
import re
import aiohttp
import json
from textblob import TextBlob
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentResult:
    """Class to represent sentiment analysis result"""

    def __init__(self, sentiment: str, score: float, confidence: float, method: str = "unknown"):
        self.sentiment = sentiment
        self.score = score
        self.confidence = confidence
        self.method = method

    def to_dict(self) -> Dict:
        return {
            "sentiment": self.sentiment,
            "score": self.score,
            "confidence": self.confidence,
            "method": self.method
        }


class OpenAISentimentAnalyzer:
    """Sentiment analyzer using OpenAI API"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.enabled = bool(self.api_key and self.api_key != "your_openai_api_key_here")

        if self.enabled:
            logger.info("OpenAI sentiment analyzer initialized")
        else:
            logger.warning("OpenAI API key not provided - using fallback analyzer")

    async def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment using OpenAI API"""
        if not self.enabled:
            raise Exception("OpenAI API key not configured")

        try:
            prompt = f"""
            Analyze the sentiment of this product review and respond with ONLY a JSON object:

            Review: "{text}"

            Response format:
            {{
                "sentiment": "positive|negative|neutral",
                "score": <float between -1.0 and 1.0>,
                "confidence": <float between 0.0 and 1.0>
            }}

            Guidelines:
            - sentiment: "positive" for good reviews, "negative" for bad reviews, "neutral" for mixed/neutral
            - score: -1.0 (very negative) to 1.0 (very positive), 0.0 for neutral
            - confidence: how certain you are about the analysis (0.0 to 1.0)
            """

            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                data = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a sentiment analysis expert. Always respond with valid JSON only."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 150
                }

                async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=30
                ) as response:
                    if response.status != 200:
                        raise Exception(f"OpenAI API error: {response.status}")

                    result = await response.json()
                    content = result['choices'][0]['message']['content'].strip()

                    try:
                        parsed = json.loads(content)
                        return SentimentResult(
                            sentiment=parsed['sentiment'],
                            score=float(parsed['score']),
                            confidence=float(parsed['confidence']),
                            method="openai"
                        )
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"Failed to parse OpenAI response: {content}")
                        raise Exception(f"Invalid API response format: {e}")

        except Exception as e:
            logger.error(f"OpenAI sentiment analysis failed: {e}")
            raise e


class FallbackSentimentAnalyzer:
    """Fallback sentiment analyzer using TextBlob and keyword analysis"""

    def __init__(self):
        self.positive_keywords = {
            'excellent', 'amazing', 'great', 'awesome', 'fantastic', 'wonderful',
            'perfect', 'outstanding', 'brilliant', 'superb', 'magnificent', 'incredible',
            'love', 'best', 'good', 'nice', 'happy', 'satisfied', 'pleased',
            'recommend', 'impressive', 'beautiful', 'quality', 'fast', 'reliable'
        }

        self.negative_keywords = {
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate', 'disappointed',
            'disgusting', 'useless', 'broken', 'defective', 'poor', 'cheap',
            'slow', 'expensive', 'overpriced', 'waste', 'regret', 'problem',
            'issue', 'failed', 'error', 'bug', 'crash', 'freeze'
        }

        logger.info("Fallback sentiment analyzer initialized")

    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment using TextBlob and keyword matching"""
        try:
            clean_text = self._clean_text(text)

            blob = TextBlob(clean_text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            keyword_score = self._analyze_keywords(clean_text)

            combined_score = (polarity * 0.7) + (keyword_score * 0.3)

            if combined_score > 0.1:
                sentiment = "positive"
            elif combined_score < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            confidence = min(abs(combined_score) + (subjectivity * 0.3), 1.0)
            confidence = max(confidence, 0.3)

            return SentimentResult(
                sentiment=sentiment,
                score=combined_score,
                confidence=confidence,
                method="fallback"
            )

        except Exception as e:
            logger.error(f"Fallback sentiment analysis failed: {e}")

            return SentimentResult(
                sentiment="neutral",
                score=0.0,
                confidence=0.1,
                method="fallback_error"
            )

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis"""

        text = text.lower()

        text = re.sub(r'\s+', ' ', text).strip()

        text = re.sub(r'[^\w\s.,!?-]', '', text)

        return text

    def _analyze_keywords(self, text: str) -> float:
        """Analyze sentiment based on keyword presence"""
        words = set(text.split())

        positive_count = len(words.intersection(self.positive_keywords))
        negative_count = len(words.intersection(self.negative_keywords))

        total_sentiment_words = positive_count + negative_count

        if total_sentiment_words == 0:
            return 0.0

        score = (positive_count - negative_count) / max(total_sentiment_words, 1)

        return max(-1.0, min(1.0, score))


class SentimentAnalysisService:
    """Main sentiment analysis service that tries multiple methods"""

    def __init__(self):
        self.openai_analyzer = OpenAISentimentAnalyzer()
        self.fallback_analyzer = FallbackSentimentAnalyzer()
        logger.info("Sentiment analysis service initialized")

    async def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Analyze sentiment with fallback strategy

        Args:
            text: Text to analyze

        Returns:
            SentimentResult object
        """
        if not text or len(text.strip()) < 5:
            return SentimentResult("neutral", 0.0, 0.1, "too_short")

        if self.openai_analyzer.enabled:
            try:
                result = await self.openai_analyzer.analyze_sentiment(text)
                logger.info(f"OpenAI analysis: {result.sentiment} ({result.score:.2f})")
                return result
            except Exception as e:
                logger.warning(f"OpenAI analysis failed, using fallback: {e}")

        result = self.fallback_analyzer.analyze_sentiment(text)
        logger.info(f"Fallback analysis: {result.sentiment} ({result.score:.2f})")
        return result

    async def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze multiple texts in parallel"""
        if not texts:
            return []

        tasks = [self.analyze_sentiment(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error analyzing text {i}: {result}")
                processed_results.append(SentimentResult("neutral", 0.0, 0.1, "error"))
            else:
                processed_results.append(result)

        return processed_results


sentiment_service = SentimentAnalysisService()
