from fastapi import FastAPI

from app.core.config import get_settings
from app.routers.health import router as health_router


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Decision Intelligence Assistant API",
        version="0.1.0",
        description=(
            "Backend API for retrieval-augmented answers and ticket "
            "priority comparisons."
        ),
    )

    app.include_router(health_router, prefix=settings.api_v1_prefix)

    return app


app = create_app()

