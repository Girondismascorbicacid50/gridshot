"""Resumable batch ingest, correction, review, and commit endpoints."""

from ._builder import RouteSpec, build_domain_router

ROUTES: tuple[RouteSpec, ...] = (
    ("POST", "/api/batch", "batch_upload"),
    ("GET", "/api/batch/{sid}", "batch_job"),
    ("POST", "/api/batch/{sid}/cancel", "batch_cancel"),
    ("POST", "/api/batch/{sid}/resume", "batch_resume"),
    ("GET", "/api/batch-jobs", "batch_jobs"),
    ("GET", "/api/batch/{sid}/image/{idx}/photo", "batch_edit_photo"),
    ("POST", "/api/batch/{sid}/image/{idx}/edit", "batch_edit_start"),
    ("POST", "/api/batch/edit/{edit_sid}/click", "batch_edit_click"),
    ("POST", "/api/batch/edit/{edit_sid}/outline", "batch_edit_outline"),
    ("POST", "/api/batch/edit/{edit_sid}/history/{direction}", "batch_edit_history"),
    ("POST", "/api/batch/edit/{edit_sid}/save", "batch_edit_save"),
    ("POST", "/api/batch/{sid}/review", "batch_review"),
    ("POST", "/api/batch/{sid}/commit", "batch_commit"),
)


def build_router(owner):
    return build_domain_router(owner, tag="batch", specs=ROUTES)
