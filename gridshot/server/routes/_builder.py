"""Build FastAPI routers around handlers owned by the composition module."""

from __future__ import annotations

from typing import Any, TypeAlias

from fastapi import APIRouter

RouteSpec: TypeAlias = tuple[str, str, str]


def build_domain_router(
    owner: Any, *, tag: str, specs: tuple[RouteSpec, ...]
) -> APIRouter:
    """Bind an ordered domain route table to its existing handler functions."""
    router = APIRouter(tags=[tag])
    for method, path, handler_name in specs:
        router.add_api_route(
            path, getattr(owner, handler_name), methods=[method], name=handler_name
        )
    return router
