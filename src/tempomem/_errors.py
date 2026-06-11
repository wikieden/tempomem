"""Exception hierarchy. See spec/API.md § Errors."""

from __future__ import annotations


class ChronotopeError(Exception):
    """Base for all Chronotope errors."""


class SchemaMismatchError(ChronotopeError):
    """Store on disk disagrees with requested/known schema (version or embedding_dim)."""


class IngestError(ChronotopeError):
    """Ingest failed."""


class BadDetectionError(IngestError):
    """A Detection failed validation."""


class AdapterError(IngestError):
    """A perception adapter failed."""


class QueryError(ChronotopeError):
    """Query failed."""


class StoreError(ChronotopeError):
    """Persistence-layer failure."""


class ToolError(ChronotopeError):
    """LLM tool dispatch failed (unknown tool or bad arguments)."""
