import os
import shutil
import hashlib
import requests
from typing import List, Dict
from urllib.parse import urlparse
from PIL import Image
import logging
from abc import ABC, abstractmethod

from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StorageServiceInterface(ABC):
    """Abstract interface for storage services"""

    @abstractmethod
    async def upload_image_from_url(self, image_url: str, folder: str = "images") -> str:
        """Upload image from URL and return storage URL"""
        pass

    @abstractmethod
    async def upload_multiple_images(self, image_urls: List[str], folder: str = "images") -> List[str]:
        """Upload multiple images and return list of storage URLs"""
        pass

    @abstractmethod
    def delete_image(self, storage_url: str) -> bool:
        """Delete image from storage"""
        pass


class GoogleCloudStorageService(StorageServiceInterface):
    """Google Cloud Storage implementation"""

    def __init__(self):
        try:
            from google.cloud import storage
            self.client = storage.Client()
            self.bucket_name = settings.GOOGLE_CLOUD_BUCKET
            self.bucket = self.client.bucket(self.bucket_name)
            self.enabled = True
            logger.info(f"Google Cloud Storage initialized with bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud Storage: {e}")
            self.enabled = False

    async def upload_image_from_url(self, image_url: str, folder: str = "images") -> str:
        """Upload image from URL to Google Cloud Storage"""
        if not self.enabled:
            raise Exception("Google Cloud Storage is not properly configured")

        try:
            response = requests.get(image_url, timeout=30, stream=True)
            response.raise_for_status()
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                raise Exception(f"URL does not point to an image: {content_type}")

            url_hash = hashlib.md5(image_url.encode()).hexdigest()
            file_extension = self._get_file_extension(image_url, content_type)
            filename = f"{folder}/{url_hash}{file_extension}"

            blob = self.bucket.blob(filename)
            blob.upload_from_string(
                response.content,
                content_type=content_type
            )

            blob.make_public()

            logger.info(f"Uploaded to GCS: {filename}")
            return blob.public_url

        except Exception as e:
            logger.error(f"Failed to upload {image_url} to GCS: {e}")
            raise e

    async def upload_multiple_images(self, image_urls: List[str], folder: str = "images") -> List[str]:
        """Upload multiple images to Google Cloud Storage"""
        results = []
        for url in image_urls:
            try:
                result_url = await self.upload_image_from_url(url, folder)
                results.append(result_url)
            except Exception as e:
                logger.error(f"Failed to upload {url}: {e}")
                results.append(url)
        return results

    def delete_image(self, storage_url: str) -> bool:
        """Delete image from Google Cloud Storage"""
        try:
            blob_name = storage_url.split(f'{self.bucket_name}/')[-1]
            blob = self.bucket.blob(blob_name)
            blob.delete()
            logger.info(f"Deleted from GCS: {blob_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {storage_url}: {e}")
            return False

    def _get_file_extension(self, url: str, content_type: str) -> str:
        """Get appropriate file extension"""
        parsed = urlparse(url)
        path = parsed.path.lower()

        if path.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
            return os.path.splitext(path)[1]

        content_type_map = {
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'image/webp': '.webp'
        }

        return content_type_map.get(content_type, '.jpg')


class MockStorageService(StorageServiceInterface):
    """Mock storage service for development and testing"""

    def __init__(self):
        self.storage_path = settings.LOCAL_STORAGE_PATH
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, "reviews"), exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, "products"), exist_ok=True)

        self.base_url = f"http://localhost:{settings.API_PORT}/storage"
        logger.info(f"Mock storage initialized at: {self.storage_path}")

    async def upload_image_from_url(self, image_url: str, folder: str = "images") -> str:
        """Download and store image locally"""
        try:
            response = requests.get(image_url, timeout=30, stream=True)
            response.raise_for_status()

            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                logger.warning(f"Non-image content type: {content_type}, proceeding anyway")

            url_hash = hashlib.md5(image_url.encode()).hexdigest()
            file_extension = self._get_file_extension(image_url, content_type)
            filename = f"{url_hash}{file_extension}"

            folder_path = os.path.join(self.storage_path, folder)
            os.makedirs(folder_path, exist_ok=True)

            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)

            try:
                self._process_image(file_path)
            except Exception as e:
                logger.warning(f"Image processing failed: {e}")

            mock_url = f"{self.base_url}/{folder}/{filename}"
            logger.info(f"Stored locally: {file_path} -> {mock_url}")
            return mock_url

        except Exception as e:
            logger.error(f"Failed to store {image_url} locally: {e}")
            return image_url

    async def upload_multiple_images(self, image_urls: List[str], folder: str = "images") -> List[str]:
        """Upload multiple images locally"""
        results = []
        for url in image_urls:
            try:
                result_url = await self.upload_image_from_url(url, folder)
                results.append(result_url)
            except Exception as e:
                logger.error(f"Failed to store {url}: {e}")
                results.append(url)
        return results

    def delete_image(self, storage_url: str) -> bool:
        """Delete image from local storage"""
        try:
            if self.base_url in storage_url:
                relative_path = storage_url.replace(self.base_url + "/", "")
                file_path = os.path.join(self.storage_path, relative_path)

                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted local file: {file_path}")
                    return True
                else:
                    logger.warning(f"File not found: {file_path}")
                    return False
            else:
                logger.warning(f"Not a local storage URL: {storage_url}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete {storage_url}: {e}")
            return False

    def _get_file_extension(self, url: str, content_type: str) -> str:
        """Get appropriate file extension"""
        parsed = urlparse(url)
        path = parsed.path.lower()

        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            if path.endswith(ext):
                return ext

        content_type_map = {
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'image/webp': '.webp'
        }

        return content_type_map.get(content_type, '.jpg')

    def _process_image(self, file_path: str):
        """Process and optimize image"""
        try:
            with Image.open(file_path) as img:
                if img.mode == 'RGBA':
                    img = img.convert('RGB')

                max_size = (1920, 1920)
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    img.save(file_path, optimize=True, quality=85)
                    logger.info(f"Resized image: {file_path}")

        except Exception as e:
            logger.warning(f"Could not process image {file_path}: {e}")


class StorageService:
    """Main storage service that chooses the appropriate implementation"""

    def __init__(self):
        self.mode = settings.STORAGE_MODE.lower()

        if self.mode == "gcs":
            self.service = GoogleCloudStorageService()
            if not self.service.enabled:
                logger.warning("GCS failed to initialize, falling back to mock storage")
                self.service = MockStorageService()
        else:
            self.service = MockStorageService()

        logger.info(f"Storage service initialized in {self.mode} mode")

    async def upload_image_from_url(self, image_url: str, folder: str = "images") -> str:
        """Upload image from URL"""
        return await self.service.upload_image_from_url(image_url, folder)

    async def upload_multiple_images(self, image_urls: List[str], folder: str = "images") -> List[str]:
        """Upload multiple images"""
        return await self.service.upload_multiple_images(image_urls, folder)

    def delete_image(self, storage_url: str) -> bool:
        """Delete image"""
        return self.service.delete_image(storage_url)

    def get_stats(self) -> Dict:
        """Get storage statistics"""
        if isinstance(self.service, MockStorageService):
            try:
                total_files = 0
                total_size = 0

                for root, dirs, files in os.walk(self.service.storage_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        total_files += 1
                        total_size += os.path.getsize(file_path)

                return {
                    "mode": "mock",
                    "total_files": total_files,
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "storage_path": self.service.storage_path
                }
            except Exception as e:
                return {"mode": "mock", "error": str(e)}
        else:
            return {"mode": "gcs", "bucket": settings.GOOGLE_CLOUD_BUCKET}


storage_service = StorageService()
