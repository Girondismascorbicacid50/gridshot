"""Library backup, composition, preview, and manufacturing-export endpoints."""

from ._builder import RouteSpec, build_domain_router

ROUTES: tuple[RouteSpec, ...] = (
    ("GET", "/api/library-export.zip", "library_export_archive"),
    ("POST", "/api/library-backups", "library_create_backup"),
    ("POST", "/api/library/compose", "library_compose"),
    ("POST", "/api/library/compose/preview.glb", "library_compose_preview_glb"),
    ("POST", "/api/library/export", "library_export"),
    ("POST", "/api/library/combine/preview", "library_combine_preview"),
    ("POST", "/api/library/combine/preview.glb", "library_combine_preview_glb"),
    ("POST", "/api/library/combine", "library_combine"),
    ("GET", "/api/files/{project}/{name}", "get_file"),
)


def build_router(owner):
    return build_domain_router(owner, tag="export", specs=ROUTES)
