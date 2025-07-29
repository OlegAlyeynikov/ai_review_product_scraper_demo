import uvicorn
from app.config import settings


if __name__ == "__main__":
    print(f"📡 Access in http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"📚 Docs: http://{settings.API_HOST}:{settings.API_PORT}/docs")

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
