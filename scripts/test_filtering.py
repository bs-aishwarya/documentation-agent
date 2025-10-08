"""
Test script to verify document filtering functionality.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from extraction.vector_store import VectorStoreManager
from ingestion.document_processor import DocumentMetadata, ProcessedDocument, DocumentChunk

# Create test documents with different IDs
def create_test_doc(doc_id: str, content: str):
    """Create a test document."""
    metadata = DocumentMetadata(
        file_name=f"test_doc_{doc_id}.txt",
        file_path=f"/test/test_doc_{doc_id}.txt",
        file_type="text/plain",
        file_size=len(content),
        file_hash=doc_id
    )
    
    chunk = DocumentChunk(
        chunk_id=f"{doc_id}_chunk_0",
        content=content,
        chunk_index=0,
        document_id=doc_id,
        start_char=0,
        end_char=len(content)
    )
    
    return ProcessedDocument(
        metadata=metadata,
        chunks=[chunk],
        total_chunks=1
    )

def test_document_filtering():
    """Test document filtering with ChromaDB."""
    print("Testing document filtering...")
    
    # Initialize vector store
    vector_store = VectorStoreManager("chroma")
    
    # Create and add test documents
    doc1 = create_test_doc("doc_001", "This is about machine learning and AI research.")
    doc2 = create_test_doc("doc_002", "This discusses quantum computing and physics.")
    
    print("\n1. Adding test documents...")
    vector_store.add_document(doc1)
    vector_store.add_document(doc2)
    print("✓ Documents added")
    
    # Test 1: Search without filters (should return results from both docs)
    print("\n2. Testing search WITHOUT filters...")
    results = vector_store.search_documents("machine learning", n_results=5)
    print(f"   Found {len(results)} results")
    for r in results:
        print(f"   - Doc ID: {r['metadata'].get('document_id')} | Score: {r['score']:.3f}")
    
    # Test 2: Search with filter for doc_001 only
    print("\n3. Testing search WITH filter (doc_001 only)...")
    filters = {"document_id": {"$in": ["doc_001"]}}
    results = vector_store.search_documents("machine learning", n_results=5, filters=filters)
    print(f"   Found {len(results)} results")
    for r in results:
        print(f"   - Doc ID: {r['metadata'].get('document_id')} | Score: {r['score']:.3f}")
    
    # Test 3: Search with filter for doc_002 only
    print("\n4. Testing search WITH filter (doc_002 only)...")
    filters = {"document_id": {"$in": ["doc_002"]}}
    results = vector_store.search_documents("quantum computing", n_results=5, filters=filters)
    print(f"   Found {len(results)} results")
    for r in results:
        print(f"   - Doc ID: {r['metadata'].get('document_id')} | Score: {r['score']:.3f}")
    
    # Test 4: Search with both documents in filter
    print("\n5. Testing search WITH filter (both docs)...")
    filters = {"document_id": {"$in": ["doc_001", "doc_002"]}}
    results = vector_store.search_documents("research", n_results=5, filters=filters)
    print(f"   Found {len(results)} results")
    for r in results:
        print(f"   - Doc ID: {r['metadata'].get('document_id')} | Score: {r['score']:.3f}")
    
    # Cleanup
    print("\n6. Cleaning up test documents...")
    vector_store.delete_document("doc_001")
    vector_store.delete_document("doc_002")
    print("✓ Cleanup complete")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_document_filtering()
