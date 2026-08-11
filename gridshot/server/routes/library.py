"""Saved-tool library CRUD and correction endpoints."""

from ._builder import RouteSpec, build_domain_router

ROUTES: tuple[RouteSpec, ...] = (
    ("POST", "/api/library/add/{project}", "library_add"),
    ("GET", "/api/library", "library_list"),
    ("GET", "/api/library/{tool_id}/outline", "library_outline"),
    ("GET", "/api/library/{tool_id}/thumb", "library_thumb"),
    ("GET", "/api/library/{tool_id}/photo", "library_photo"),
    ("GET", "/api/library/{tool_id}/photo-thumb", "library_photo_thumb"),
    ("GET", "/api/library/{tool_id}/photo-outline", "library_photo_outline"),
    ("POST", "/api/library/{tool_id}/outline-px", "library_save_outline_px"),
    ("POST", "/api/library/{tool_id}/edit", "library_edit_start"),
    ("POST", "/api/library/edit/{sid}/click", "library_edit_click"),
    ("POST", "/api/library/edit/{sid}/outline", "library_edit_outline"),
    ("POST", "/api/library/edit/{sid}/history/{direction}", "library_edit_history"),
    ("POST", "/api/library/edit/{sid}/save", "library_edit_save"),
    ("DELETE", "/api/library/{tool_id}", "library_delete"),
    ("PATCH", "/api/library/{tool_id}", "library_edit"),
)


def build_router(owner):
    return build_domain_router(owner, tag="library", specs=ROUTES)
