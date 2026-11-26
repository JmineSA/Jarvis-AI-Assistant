"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from jarvis import __version__
from jarvis.ai.ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from jarvis.api.routes import router
from jarvis.config import Settings, load_settings
from jarvis.wiring import build_brain, open_store

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, client: OllamaClient | None = None) -> FastAPI:
    """Build the application. `client` can be replaced in tests."""
    settings = settings or load_settings()
    client = client or OllamaClient.from_settings(settings.ollama)
    brain = build_brain(settings, client, store=open_store(settings))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        logger.info("JARVIS API starting (model: %s).", settings.ollama.model)
        yield
        await client.aclose()
        logger.info("JARVIS API stopped.")

    app = FastAPI(title="JARVIS", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.client = client
    app.state.brain = brain
    app.include_router(router)

    def _error(status_code: int):
        async def handler(_: Request, exc: OllamaError) -> JSONResponse:
            logger.error("Model error (%s): %s", type(exc).__name__, exc)
            return JSONResponse(status_code=status_code, content={"detail": str(exc)})

        return handler

    app.add_exception_handler(OllamaUnavailableError, _error(503))
    app.add_exception_handler(OllamaTimeoutError, _error(504))
    app.add_exception_handler(OllamaResponseError, _error(502))
    return app
