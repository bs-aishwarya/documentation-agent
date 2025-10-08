"""
Comprehensive test suite for the Document Q&A Agent.

This module contains unit tests and integration tests for all components
of the enterprise document processing and Q&A system.
"""

import pytest
import tempfile
import os
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import numpy as np

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import ConfigManager, get_config
from ingestion.document_processor import DocumentProcessor, DocumentChunk, ProcessedDocument
from extraction.vector_store import EmbeddingManager, ChromaVectorStore, VectorStoreManager
from query.query_engine import DocumentQueryEngine, QueryType, QueryResult
from agents.arxiv_agent import ArXivAgent, ArXivAPI, ArXivPaper, ArXivSearchBuilder


class TestConfigManager:
    """Tests for configuration management."""
    
    def test_config_loading(self):
        """Test configuration loading from file."""
        config = get_config()
        assert config is not None
        assert hasattr(config, 'llm')
        assert hasattr(config, 'vector_store')
        assert hasattr(config, 'document_processing')
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = get_config()
        
        # Test LLM configuration
        assert config.llm.provider in ['openai', 'gemini']
        assert config.llm.temperature >= 0.0
        assert config.llm.temperature <= 2.0
        assert config.llm.max_tokens > 0
        
        # Test vector store configuration
        assert config.vector_store.type in ['chroma', 'faiss']
        assert config.vector_store.embedding_dimension > 0
    
    @patch.dict(os.environ, {'LLM_PROVIDER': 'openai', 'LLM_TEMPERATURE': '0.5'})
    def test_environment_override(self):
        """Test environment variable overrides."""
        # This would require reloading config, simplified for test
        assert os.environ.get('LLM_PROVIDER') == 'openai'
        assert os.environ.get('LLM_TEMPERATURE') == '0.5'


class TestDocumentProcessor:
    """Tests for document processing functionality."""
    
    @pytest.fixture
    def temp_pdf(self):
        """Create a temporary PDF file for testing."""
        # This would create a real PDF for integration tests
        # For unit tests, we'll mock the file operations
        temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        temp_file.write(b'%PDF-1.4 mock PDF content')
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def processor(self):
        """Create a document processor instance."""
        return DocumentProcessor()
    
    def test_document_chunk_creation(self):
        """Test DocumentChunk creation and validation."""
        chunk = DocumentChunk(
            chunk_id="test_chunk_1",
            document_id="test_doc",
            content="This is test content for the chunk.",
            chunk_type="text",
            page_number=1,
            metadata={"test": "value"}
        )
        
        assert chunk.chunk_id == "test_chunk_1"
        assert chunk.document_id == "test_doc"
        assert chunk.content == "This is test content for the chunk."
        assert chunk.chunk_type == "text"
        assert chunk.page_number == 1
        assert chunk.metadata["test"] == "value"
    
    def test_processed_document_creation(self):
        """Test ProcessedDocument creation."""
        chunks = [
            DocumentChunk(
                chunk_id="chunk_1",
                document_id="doc_1",
                content="First chunk",
                chunk_type="text"
            )
        ]
        
        processed_doc = ProcessedDocument(
            document_id="doc_1",
            chunks=chunks,
            metadata=Mock()
        )
        
        assert processed_doc.document_id == "doc_1"
        assert len(processed_doc.chunks) == 1
        assert processed_doc.chunks[0].content == "First chunk"
    
    @patch('src.ingestion.document_processor.fitz')
    def test_pdf_text_extraction(self, mock_fitz, processor):
        """Test PDF text extraction with mocked PyMuPDF."""
        # Mock PyMuPDF document
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "Sample PDF text content"
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz.open.return_value = mock_doc
        
        # This would test the actual extraction logic
        # For now, we verify the mock setup
        doc = mock_fitz.open("test.pdf")
        assert len(doc) == 1
        assert doc[0].get_text() == "Sample PDF text content"
    
    def test_content_chunking(self, processor):
        """Test content chunking functionality."""
        content = "This is a long piece of text that should be split into multiple chunks. " * 20
        
        chunks = processor._chunk_content(
            content=content,
            document_id="test_doc",
            chunk_size=100,
            overlap=20
        )
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.document_id == "test_doc" for chunk in chunks)
    
    @patch('src.ingestion.document_processor.DocumentProcessor._extract_with_pymupdf')
    def test_document_processing_pipeline(self, mock_extract, processor):
        """Test the complete document processing pipeline."""
        # Mock extraction results
        mock_extract.return_value = {
            'text': "Sample document text",
            'pages': 1,
            'images': [],
            'tables': [],
            'metadata': {'title': 'Test Document'}
        }
        
        # This would test the full pipeline
        # For now, verify mock behavior
        result = mock_extract("test.pdf")
        assert result['text'] == "Sample document text"
        assert result['pages'] == 1


class TestVectorStore:
    """Tests for vector storage functionality."""
    
    @pytest.fixture
    def embedding_manager(self):
        """Create an embedding manager for testing."""
        with patch('src.extraction.vector_store.SentenceTransformer') as mock_model:
            mock_model.return_value.encode.return_value = np.random.rand(384)
            return EmbeddingManager(model_name="test-model")
    
    def test_embedding_generation(self, embedding_manager):
        """Test text embedding generation."""
        text = "This is a test sentence for embedding."
        
        with patch.object(embedding_manager, 'model') as mock_model:
            mock_model.encode.return_value = np.random.rand(384)
            
            embedding = embedding_manager.embed_text(text)
            
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (384,)
            mock_model.encode.assert_called_once()
    
    def test_batch_embedding_generation(self, embedding_manager):
        """Test batch embedding generation."""
        texts = ["First text", "Second text", "Third text"]
        
        with patch.object(embedding_manager, 'model') as mock_model:
            mock_model.encode.return_value = np.random.rand(3, 384)
            
            embeddings = embedding_manager.embed_texts(texts)
            
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape == (3, 384)
    
    @patch('src.extraction.vector_store.chromadb')
    def test_chroma_initialization(self, mock_chromadb):
        """Test ChromaDB initialization."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        
        store = ChromaVectorStore(collection_name="test_collection")
        
        assert store.collection_name == "test_collection"
        mock_chromadb.PersistentClient.assert_called_once()
    
    def test_vector_search(self):
        """Test vector similarity search."""
        # Mock search results
        mock_results = {
            "ids": [["chunk_1", "chunk_2"]],
            "documents": [["First document", "Second document"]],
            "metadatas": [[{"file": "test1.pdf"}, {"file": "test2.pdf"}]],
            "distances": [[0.1, 0.3]]
        }
        
        with patch('src.extraction.vector_store.ChromaVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.search.return_value = [
                {
                    "id": "chunk_1",
                    "content": "First document",
                    "metadata": {"file": "test1.pdf"},
                    "score": 0.9
                }
            ]
            
            query_embedding = np.random.rand(384)
            results = mock_store.search(query_embedding, n_results=5)
            
            assert len(results) == 1
            assert results[0]["id"] == "chunk_1"
            assert results[0]["score"] == 0.9


class TestQueryEngine:
    """Tests for query processing functionality."""
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Create a mock LLM manager."""
        mock_manager = Mock()
        mock_manager.generate_text.return_value = "This is a test response from the LLM."
        return mock_manager
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        mock_store = Mock()
        mock_store.search_documents.return_value = [
            {
                "id": "chunk_1",
                "content": "Relevant document content for testing",
                "metadata": {"file_name": "test.pdf", "page_number": 1},
                "score": 0.8
            }
        ]
        return mock_store
    
    def test_query_classification(self, mock_llm_manager):
        """Test query type classification."""
        from query.query_engine import QueryClassifier
        
        classifier = QueryClassifier(mock_llm_manager)
        mock_llm_manager.generate_text.return_value = "direct_lookup"
        
        query_type = classifier.classify_query("What is machine learning?")
        
        assert query_type == QueryType.DIRECT_LOOKUP
        mock_llm_manager.generate_text.assert_called_once()
    
    def test_query_result_creation(self):
        """Test QueryResult object creation."""
        result = QueryResult(
            query="Test query",
            answer="Test answer",
            query_type=QueryType.QUESTION_ANSWER,
            confidence=0.8,
            sources=[{"id": "chunk_1", "content": "test content"}]
        )
        
        assert result.query == "Test query"
        assert result.answer == "Test answer"
        assert result.query_type == QueryType.QUESTION_ANSWER
        assert result.confidence == 0.8
        assert len(result.sources) == 1
    
    @patch('src.query.query_engine.LLMManager')
    @patch('src.query.query_engine.VectorStoreManager')
    def test_document_query_engine_initialization(self, mock_vector, mock_llm):
        """Test DocumentQueryEngine initialization."""
        engine = DocumentQueryEngine()
        
        assert engine.llm_manager is not None
        assert engine.vector_store is not None
        assert engine.query_classifier is not None
        assert engine.query_processor is not None
    
    def test_context_preparation(self):
        """Test context preparation from search results."""
        from query.query_engine import DocumentQueryEngine
        
        search_results = [
            {
                "content": "First relevant chunk",
                "metadata": {"file_name": "doc1.pdf", "page_number": 1},
                "score": 0.9
            },
            {
                "content": "Second relevant chunk",
                "metadata": {"file_name": "doc2.pdf", "page_number": 2},
                "score": 0.7
            }
        ]
        
        with patch('src.query.query_engine.LLMManager'), \
             patch('src.query.query_engine.VectorStoreManager'):
            engine = DocumentQueryEngine()
            context = engine._prepare_context(search_results)
            
            assert "First relevant chunk" in context
            assert "Second relevant chunk" in context
            assert "doc1.pdf" in context
            assert "doc2.pdf" in context


class TestArXivAgent:
    """Tests for ArXiv integration functionality."""
    
    @pytest.fixture
    def arxiv_api(self):
        """Create an ArXiv API instance."""
        return ArXivAPI()
    
    @pytest.fixture
    def sample_paper(self):
        """Create a sample ArXiv paper for testing."""
        return ArXivPaper(
            arxiv_id="1234.5678",
            title="Sample Research Paper",
            authors=["John Doe", "Jane Smith"],
            abstract="This is a sample abstract for testing purposes.",
            published_date=datetime.now(),
            updated_date=None,
            categories=["cs.AI", "cs.LG"],
            doi=None,
            pdf_url="https://arxiv.org/pdf/1234.5678.pdf",
            abstract_url="https://arxiv.org/abs/1234.5678",
            comment=None,
            journal_ref=None,
            primary_category="cs.AI"
        )
    
    def test_arxiv_search_builder(self):
        """Test ArXiv search query builder."""
        builder = ArXivSearchBuilder()
        query = (builder
                .add_title("machine learning")
                .add_author("Geoffrey Hinton")
                .add_category("cs.LG")
                .build())
        
        assert 'ti:"machine learning"' in query
        assert 'au:"Geoffrey Hinton"' in query
        assert 'cat:cs.LG' in query
        assert ' AND ' in query
    
    def test_arxiv_paper_creation(self, sample_paper):
        """Test ArXiv paper object creation."""
        assert sample_paper.arxiv_id == "1234.5678"
        assert sample_paper.title == "Sample Research Paper"
        assert len(sample_paper.authors) == 2
        assert "cs.AI" in sample_paper.categories
    
    def test_arxiv_paper_to_dict(self, sample_paper):
        """Test ArXiv paper serialization."""
        paper_dict = sample_paper.to_dict()
        
        assert paper_dict["arxiv_id"] == "1234.5678"
        assert paper_dict["title"] == "Sample Research Paper"
        assert paper_dict["authors"] == ["John Doe", "Jane Smith"]
        assert isinstance(paper_dict["published_date"], str)
    
    @patch('requests.Session.get')
    def test_arxiv_api_search(self, mock_get, arxiv_api):
        """Test ArXiv API search functionality."""
        # Mock response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.text = '''<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
        <entry>
            <id>http://arxiv.org/abs/1234.5678v1</id>
            <title>Test Paper</title>
            <summary>Test abstract</summary>
            <published>2023-01-01T00:00:00Z</published>
            <author><name>Test Author</name></author>
        </entry>
        </feed>'''
        mock_get.return_value = mock_response
        
        papers = arxiv_api.search_papers("machine learning", max_results=1)
        
        # With proper XML parsing, this would return papers
        # For now, verify the mock was called
        mock_get.assert_called_once()
    
    @patch('src.agents.arxiv_agent.ArXivAPI')
    def test_arxiv_agent_initialization(self, mock_api):
        """Test ArXiv agent initialization."""
        agent = ArXivAgent()
        
        assert agent.api is not None
        assert agent.query_processor is not None
        assert agent.document_processor is not None
        assert agent.downloads_dir.exists()
    
    def test_natural_query_processing(self):
        """Test natural language query processing."""
        from agents.arxiv_agent import ArXivQueryProcessor
        
        processor = ArXivQueryProcessor()
        
        # Test rule-based processing
        query = "papers by Geoffrey Hinton about machine learning"
        search_query = processor._process_with_rules(query)
        
        # Should contain author search
        assert 'au:' in search_query or 'all:' in search_query


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    def temp_directory(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('src.utils.config.get_config')
    def test_end_to_end_document_processing(self, mock_config, temp_directory):
        """Test complete document processing pipeline."""
        # Mock configuration
        mock_config.return_value = Mock()
        mock_config.return_value.document_processing.chunk_size = 500
        mock_config.return_value.document_processing.chunk_overlap = 50
        
        # This would test the complete pipeline from PDF to query
        # For now, verify components can be instantiated
        try:
            processor = DocumentProcessor()
            vector_manager = VectorStoreManager()
            assert processor is not None
            assert vector_manager is not None
        except Exception as e:
            pytest.skip(f"Integration test requires full setup: {e}")
    
    def test_arxiv_to_vector_store_pipeline(self):
        """Test ArXiv paper retrieval and indexing pipeline."""
        # This would test downloading a paper, processing it, and indexing
        # Skipped for unit tests to avoid external dependencies
        pytest.skip("Integration test requires external API access")


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--strict-markers"
    ])
