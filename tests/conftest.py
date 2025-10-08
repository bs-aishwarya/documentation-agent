"""
Test configuration and utilities for the test suite.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return {
        "llm": {
            "provider": "openai",
            "temperature": 0.1,
            "max_tokens": 1000
        },
        "vector_store": {
            "type": "chroma",
            "embedding_dimension": 384
        },
        "document_processing": {
            "chunk_size": 500,
            "chunk_overlap": 50
        }
    }


@pytest.fixture
def mock_embedding():
    """Provide a mock embedding vector."""
    return np.random.rand(384).astype(np.float32)


@pytest.fixture
def mock_document_chunks():
    """Provide mock document chunks for testing."""
    from src.ingestion.document_processor import DocumentChunk
    
    chunks = []
    for i in range(3):
        chunk = DocumentChunk(
            chunk_id=f"chunk_{i}",
            document_id="test_doc",
            content=f"This is test content for chunk {i}. " * 10,
            chunk_type="text",
            page_number=i + 1,
            metadata={"test": True, "chunk_index": i}
        )
        chunks.append(chunk)
    
    return chunks


@pytest.fixture
def sample_pdf_content():
    """Provide sample PDF content for testing."""
    return {
        "text": "This is a sample PDF document with multiple pages. " * 50,
        "pages": 3,
        "images": [],
        "tables": [
            {
                "page": 1,
                "content": [["Header 1", "Header 2"], ["Row 1 Col 1", "Row 1 Col 2"]],
                "bbox": [100, 100, 400, 200]
            }
        ],
        "metadata": {
            "title": "Sample Test Document",
            "author": "Test Author",
            "creation_date": "2023-01-01",
            "page_count": 3
        }
    }


@pytest.fixture
def mock_arxiv_papers():
    """Provide mock ArXiv papers for testing."""
    from datetime import datetime
    from src.agents.arxiv_agent import ArXivPaper
    
    papers = []
    for i in range(3):
        paper = ArXivPaper(
            arxiv_id=f"2023.{1000 + i:04d}",
            title=f"Sample Research Paper {i + 1}",
            authors=[f"Author {i + 1}A", f"Author {i + 1}B"],
            abstract=f"This is the abstract for paper {i + 1}. " * 10,
            published_date=datetime(2023, 1, i + 1),
            updated_date=None,
            categories=["cs.AI", "cs.LG"],
            doi=None,
            pdf_url=f"https://arxiv.org/pdf/2023.{1000 + i:04d}.pdf",
            abstract_url=f"https://arxiv.org/abs/2023.{1000 + i:04d}",
            comment=None,
            journal_ref=None,
            primary_category="cs.AI"
        )
        papers.append(paper)
    
    return papers


class MockLLMManager:
    """Mock LLM manager for testing."""
    
    def __init__(self, responses=None):
        self.responses = responses or {
            "classify": "question_answer",
            "answer": "This is a mock answer from the LLM.",
            "summarize": "This is a mock summary.",
            "extract": "Extracted data: value1, value2, value3"
        }
    
    def generate_text(self, prompt, system_message=None):
        """Generate mock text based on prompt content."""
        prompt_lower = prompt.lower()
        
        if "classify" in prompt_lower or "category" in prompt_lower:
            return self.responses["classify"]
        elif "summarize" in prompt_lower or "summary" in prompt_lower:
            return self.responses["summarize"]
        elif "extract" in prompt_lower or "data" in prompt_lower:
            return self.responses["extract"]
        else:
            return self.responses["answer"]


class MockVectorStore:
    """Mock vector store for testing."""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
    
    def add_document(self, processed_doc):
        """Add a document to the mock store."""
        self.documents.append(processed_doc)
    
    def search_documents(self, query, n_results=5, filters=None):
        """Return mock search results."""
        return [
            {
                "id": f"chunk_{i}",
                "content": f"Mock search result {i + 1} for query: {query[:30]}...",
                "metadata": {
                    "file_name": f"document_{i + 1}.pdf",
                    "page_number": i + 1,
                    "section_title": f"Section {i + 1}"
                },
                "score": 0.9 - (i * 0.1)
            }
            for i in range(min(n_results, 3))
        ]
    
    def get_stats(self):
        """Return mock statistics."""
        return {
            "total_chunks": len(self.documents) * 10,  # Assume 10 chunks per doc
            "total_documents": len(self.documents)
        }


@pytest.fixture
def mock_llm_manager():
    """Provide a mock LLM manager."""
    return MockLLMManager()


@pytest.fixture
def mock_vector_store():
    """Provide a mock vector store."""
    return MockVectorStore()


# Test markers
pytest_plugins = []

# Custom markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
pytest.mark.external = pytest.mark.external  # Tests requiring external APIs
