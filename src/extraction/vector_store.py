"""
Vector storage and retrieval system for document chunks.

This module handles the storage and retrieval of document embeddings
using various vector database backends.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

# Vector storage
import chromadb
from chromadb.config import Settings
import faiss

# Embeddings
from sentence_transformers import SentenceTransformer

# Logging
from loguru import logger

# Configuration and types
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_config
from ingestion.document_processor import DocumentChunk, ProcessedDocument


class EmbeddingManager:
    """Manages text embeddings for document chunks."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.config = get_config()
        self.model_name = model_name or self.config.vector_store.embedding_model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            np.ndarray: Text embedding
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Generate embedding
        embedding = self.model.encode(cleaned_text, convert_to_numpy=True)
        # Normalize embedding to unit length to make similarity comparisons more stable
        try:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        except Exception:
            pass
        return embedding
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            np.ndarray: Array of text embeddings
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        # Clean texts
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        # Generate embeddings
        embeddings = self.model.encode(cleaned_texts, convert_to_numpy=True)
        # Normalize each embedding row-wise
        try:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
        except Exception:
            pass
        return embeddings
    
    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[Tuple[DocumentChunk, np.ndarray]]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List[Tuple[DocumentChunk, np.ndarray]]: Chunks with their embeddings
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Extract text from chunks
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Pair chunks with embeddings
        chunk_embeddings = list(zip(chunks, embeddings))
        
        logger.info("Embeddings generated successfully")
        return chunk_embeddings
    
    def _clean_text(self, text: str) -> str:
        """Clean text for embedding generation."""
        if not text:
            return ""
        
        # Basic cleaning
        cleaned = text.strip()
        
        # Remove excessive whitespace
        import re
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned


class ChromaVectorStore:
    """ChromaDB-based vector storage implementation."""
    
    def __init__(self, collection_name: Optional[str] = None):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the collection to use
        """
        self.config = get_config()
        self.collection_name = collection_name or self.config.vector_store.chroma.collection_name
        self.persist_directory = self.config.vector_store.chroma.persist_directory
        
        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            logger.info(f"Initializing ChromaDB client at {self.persist_directory}")
            
            # Create client with persistence
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document chunks for Q&A"}
            )
            
            logger.info(f"ChromaDB collection '{self.collection_name}' initialized")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def add_chunks(self, chunk_embeddings: List[Tuple[DocumentChunk, np.ndarray]]):
        """
        Add document chunks with embeddings to the vector store.
        
        Args:
            chunk_embeddings: List of (chunk, embedding) tuples
        """
        if not chunk_embeddings:
            return
        
        logger.info(f"Adding {len(chunk_embeddings)} chunks to ChromaDB")
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for chunk, embedding in chunk_embeddings:
            ids.append(chunk.chunk_id)
            documents.append(chunk.content)
            embeddings.append(embedding.tolist())
            
            # Prepare metadata
            metadata = {
                "document_id": chunk.document_id,
                "file_name": chunk.metadata.get("file_name", ""),
                "file_path": chunk.metadata.get("file_path", ""),
                "chunk_type": chunk.chunk_type,
                "created_at": datetime.now().isoformat()
            }
            
            if chunk.page_number is not None:
                metadata["page_number"] = chunk.page_number
            
            if chunk.section_title:
                metadata["section_title"] = chunk.section_title
            
            # Add custom metadata
            for key, value in chunk.metadata.items():
                if key not in metadata and isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
            
            metadatas.append(metadata)
        
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info("Chunks added to ChromaDB successfully")
            
        except Exception as e:
            logger.error(f"Error adding chunks to ChromaDB: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, n_results: int = 5,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List[Dict[str, Any]]: Search results with chunks and scores
        """
        try:
            # Prepare where clause for filtering
            where = None
            if filters:
                where = filters
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            search_results = []
            
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    result = {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        # Map distance to a bounded similarity in (0, 1]: similarity = 1 / (1 + distance)
                        "score": 1.0 / (1.0 + float(results["distances"][0][i]))
                    }
                    search_results.append(result)

            # Sort results by score descending to ensure highest relevance first
            search_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []
    
    def delete_document(self, document_id: str):
        """
        Delete all chunks for a specific document.
        
        Args:
            document_id: ID of the document to delete
        """
        try:
            self.collection.delete(where={"document_id": document_id})
            logger.info(f"Deleted chunks for document: {document_id}")
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}


class FAISSVectorStore:
    """FAISS-based vector storage implementation."""
    
    def __init__(self):
        """Initialize FAISS vector store."""
        self.config = get_config()
        self.index_path = self.config.vector_store.faiss.index_path
        self.dimension = self.config.vector_store.embedding_dimension
        
        # Ensure directory exists
        Path(os.path.dirname(self.index_path)).mkdir(parents=True, exist_ok=True)
        
        self.index = None
        self.chunks_metadata: List[Dict[str, Any]] = []
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index."""
        try:
            # Try to load existing index
            if os.path.exists(self.index_path) and os.path.exists(f"{self.index_path}.metadata"):
                logger.info(f"Loading existing FAISS index from {self.index_path}")
                self.index = faiss.read_index(self.index_path)
                
                # Load metadata
                with open(f"{self.index_path}.metadata", "rb") as f:
                    self.chunks_metadata = pickle.load(f)
                
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            else:
                # Create new index
                logger.info("Creating new FAISS index")
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
                self.chunks_metadata = []
                
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}")
            # Create new index as fallback
            self.index = faiss.IndexFlatIP(self.dimension)
            self.chunks_metadata = []
    
    def add_chunks(self, chunk_embeddings: List[Tuple[DocumentChunk, np.ndarray]]):
        """
        Add document chunks with embeddings to the FAISS index.
        
        Args:
            chunk_embeddings: List of (chunk, embedding) tuples
        """
        if not chunk_embeddings:
            return
        
        logger.info(f"Adding {len(chunk_embeddings)} chunks to FAISS index")
        
        # Prepare embeddings and metadata
        embeddings = []
        metadatas = []
        
        for chunk, embedding in chunk_embeddings:
            # Normalize embedding for cosine similarity
            normalized_embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(normalized_embedding)
            
            # Prepare metadata
            metadata = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "chunk_type": chunk.chunk_type,
                "file_name": chunk.metadata.get("file_name", ""),
                "file_path": chunk.metadata.get("file_path", ""),
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "created_at": datetime.now().isoformat()
            }
            
            # Add custom metadata
            metadata.update(chunk.metadata)
            metadatas.append(metadata)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Add to index
        self.index.add(embeddings_array)
        self.chunks_metadata.extend(metadatas)
        
        # Save index and metadata
        self._save_index()
        
        logger.info(f"Added chunks to FAISS index. Total vectors: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, n_results: int = 5,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using FAISS.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filters: Optional metadata filters (applied post-search)
            
        Returns:
            List[Dict[str, Any]]: Search results with chunks and scores
        """
        if self.index.ntotal == 0:
            return []
        
        try:
            # Normalize query embedding
            normalized_query = query_embedding / np.linalg.norm(query_embedding)
            query_array = np.array([normalized_query], dtype=np.float32)
            
            # Search
            scores, indices = self.index.search(query_array, min(n_results * 2, self.index.ntotal))
            
            # Process results
            search_results = []
            
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.chunks_metadata):
                    continue
                
                metadata = self.chunks_metadata[idx]
                score = float(scores[0][i])
                
                # Apply filters if provided
                if filters:
                    if not self._match_filters(metadata, filters):
                        continue
                
                result = {
                    "id": metadata["chunk_id"],
                    "content": metadata["content"],
                    "metadata": metadata,
                    "score": score,
                    "distance": 1 - score
                }
                
                search_results.append(result)
                
                if len(search_results) >= n_results:
                    break
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching FAISS index: {e}")
            return []
    
    def _match_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the provided filters."""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            faiss.write_index(self.index, self.index_path)
            
            with open(f"{self.index_path}.metadata", "wb") as f:
                pickle.dump(self.chunks_metadata, f)
                
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
    
    def delete_document(self, document_id: str):
        """
        Delete all chunks for a specific document.
        Note: FAISS doesn't support deletion, so we would need to rebuild the index.
        """
        logger.warning("FAISS doesn't support deletion. Consider using ChromaDB for this feature.")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_path": self.index_path
        }


class VectorStoreManager:
    """Main interface for vector storage operations."""
    
    def __init__(self, store_type: Optional[str] = None):
        """
        Initialize the vector store manager.
        
        Args:
            store_type: Type of vector store to use ('chroma' or 'faiss')
        """
        self.config = get_config()
        self.store_type = store_type or self.config.vector_store.type
        self.embedding_manager = EmbeddingManager()
        
        # Initialize the appropriate vector store
        if self.store_type == "chroma":
            self.vector_store = ChromaVectorStore()
        elif self.store_type == "faiss":
            self.vector_store = FAISSVectorStore()
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")
        
        logger.info(f"Initialized vector store manager with {self.store_type}")
    
    def add_document(self, processed_doc: ProcessedDocument):
        """
        Add a processed document to the vector store.
        
        Args:
            processed_doc: The processed document to add
        """
        logger.info(f"Adding document to vector store: {processed_doc.metadata.file_name}")
        
        # Generate embeddings for chunks
        chunk_embeddings = self.embedding_manager.embed_chunks(processed_doc.chunks)
        
        # Add to vector store
        self.vector_store.add_chunks(chunk_embeddings)
        
        logger.info(f"Document added successfully: {processed_doc.metadata.file_name}")
    
    def search_documents(self, query: str, n_results: int = 5,
                        filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List[Dict[str, Any]]: Search results
        """
        logger.info(f"Searching documents for query: '{query[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.embedding_manager.embed_text(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, n_results, filters)

        # Filter by similarity threshold
        threshold = self.config.query.similarity_threshold
        filtered_results = [r for r in results if r.get("score", 0) >= threshold]

        if filtered_results:
            logger.info(f"Found {len(filtered_results)} relevant chunks above threshold {threshold:.2f}")
            return filtered_results

        # Fallback: if nothing met the threshold but we still have results, return the top hits
        if results:
            scores = [r.get("score", 0.0) for r in results]
            top_score = max(scores)
            logger.info(
                "No chunks met similarity threshold {threshold:.2f}; highest score was {top_score:.3f}. Returning top {count} result(s) instead",
                threshold=threshold,
                top_score=top_score,
                count=min(n_results, len(results))
            )
            return results[:n_results]

        logger.info("No relevant chunks found for query")
        return []
    
    def delete_document(self, document_id: str):
        """
        Delete a document from the vector store.
        
        Args:
            document_id: ID of the document to delete
        """
        self.vector_store.delete_document(document_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        stats = self.vector_store.get_collection_stats()
        stats["store_type"] = self.store_type
        stats["embedding_model"] = self.embedding_manager.model_name
        return stats


if __name__ == "__main__":
    # Example usage
    from ingestion.document_processor import DocumentProcessor
    
    # Initialize components
    processor = DocumentProcessor()
    vector_manager = VectorStoreManager()
    
    # Process and add a document
    try:
        doc = processor.process_document("documents/sample.pdf")
        vector_manager.add_document(doc)
        
        # Search for relevant content
        results = vector_manager.search_documents("machine learning algorithms", n_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Score: {result['score']:.3f}):")
            print(f"Content: {result['content'][:200]}...")
            print(f"Metadata: {result['metadata']}")
            
        # Get stats
        stats = vector_manager.get_stats()
        print(f"\nVector Store Stats: {stats}")
        
    except Exception as e:
        print(f"Error: {e}")
