"""Exception hierarchy. See spec/API.md § Errors."""

from __future__ import annotations


class SpatialMemError(Exception):
    """Base for all SpatialMem errors."""


class SchemaMismatchError(SpatialMemError):
    """Store on disk disagrees with requested/known schema (version or embedding_dim)."""


class IngestError(SpatialMemError):
    """Ingest failed."""


class BadDetectionError(IngestError):
    """A Detection failed validation."""


class AdapterError(IngestError):
    """A perception adapter failed."""


class QueryError(SpatialMemError):
    """Query failed."""


class StoreError(SpatialMemError):
    """Persistence-layer failure."""
