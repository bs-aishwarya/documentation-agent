"""Document Q&A Agent - Enterprise document processing and question answering system.

This package provides intelligent document processing, vector storage,
and natural language querying capabilities for PDF documents.
"""

__version__ = "1.0.0"
__author__ = "Document Agent Team"
__description__ = "Enterprise-ready AI-powered document Q&A system"

# Core modules (import lazily to avoid heavy imports on package import)
def get_config():
    from .utils.config import get_config as _get_config
    return _get_config()

def DocumentProcessor():
    from .ingestion.document_processor import DocumentProcessor as _DocumentProcessor
    return _DocumentProcessor()

def VectorStoreManager(*args, **kwargs):
    from .extraction.vector_store import VectorStoreManager as _VectorStoreManager
    return _VectorStoreManager(*args, **kwargs)

def DocumentQueryEngine(*args, **kwargs):
    from .query.query_engine import DocumentQueryEngine as _DocumentQueryEngine
    return _DocumentQueryEngine(*args, **kwargs)

__all__ = [
    "get_config",
    "DocumentProcessor",
    "VectorStoreManager",
    "DocumentQueryEngine",
]
