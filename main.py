"""
Main entry point for the Document Q&A Agent.

This script provides a command-line interface to run different components
of the document processing and Q&A system.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(description="Document Q&A Agent")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Web interface command
    web_parser = subparsers.add_parser("web", help="Start web interface")
    web_parser.add_argument("--host", default="localhost", help="Host to bind to")
    web_parser.add_argument("--port", default=8501, type=int, help="Port to bind to")
    
    # Process document command
    process_parser = subparsers.add_parser("process", help="Process a document")
    process_parser.add_argument("file_path", help="Path to the document to process")
    process_parser.add_argument("--index", action="store_true", help="Add to vector store")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query documents")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--max-results", default=5, type=int, help="Max results to return")
    
    # ArXiv command
    arxiv_parser = subparsers.add_parser("arxiv", help="Search ArXiv papers")
    arxiv_parser.add_argument("query", help="Search query")
    arxiv_parser.add_argument("--download", action="store_true", help="Download papers")
    arxiv_parser.add_argument("--max-results", default=10, type=int, help="Max results")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Show configuration")
    
    args = parser.parse_args()
    
    if args.command == "web":
        start_web_interface(args.host, args.port)
    elif args.command == "process":
        process_document(args.file_path, args.index)
    elif args.command == "query":
        query_documents(args.question, args.max_results)
    elif args.command == "arxiv":
        search_arxiv(args.query, args.download, args.max_results)
    elif args.command == "config":
        show_config()
    else:
        parser.print_help()


def start_web_interface(host: str, port: int):
    """Start the Streamlit web interface."""
    import subprocess
    
    cmd = [
        "streamlit", "run", 
        "src/query/streamlit_app.py",
        "--server.address", host,
        "--server.port", str(port)
    ]
    
    print(f"🚀 Starting web interface at http://{host}:{port}")
    subprocess.run(cmd)


def process_document(file_path: str, index: bool = False):
    """Process a single document."""
    from ingestion.document_processor import DocumentProcessor
    from extraction.vector_store import VectorStoreManager
    
    print(f"📄 Processing document: {file_path}")
    
    try:
        # Process document
        processor = DocumentProcessor()
        doc = processor.process_document(file_path)
        
        print(f"✅ Document processed successfully:")
        print(f"   - Pages: {doc.metadata.page_count}")
        print(f"   - Chunks: {len(doc.chunks)}")
        print(f"   - File size: {doc.metadata.file_size} bytes")
        
        # Add to vector store if requested
        if index:
            print("🔍 Adding to vector store...")
            vector_store = VectorStoreManager()
            vector_store.add_document(doc)
            print("✅ Document indexed successfully")
            
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        sys.exit(1)


def query_documents(question: str, max_results: int = 5):
    """Query the document collection."""
    from query.query_engine import DocumentQueryEngine
    
    print(f"🤔 Processing question: {question}")
    
    try:
        engine = DocumentQueryEngine()
        result = engine.query(question, n_results=max_results)
        
        print(f"\n📝 Answer:")
        print(f"{result.answer}")
        print(f"\n📊 Confidence: {result.confidence:.2f}")
        print(f"🔍 Query Type: {result.query_type.value}")
        print(f"📚 Sources: {len(result.sources)}")
        
        if result.sources:
            print(f"\n📖 Source Details:")
            for i, source in enumerate(result.sources[:3], 1):  # Show top 3
                print(f"   {i}. {source['file_name']} (Score: {source['relevance_score']:.3f})")
                
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        sys.exit(1)


def search_arxiv(query: str, download: bool = False, max_results: int = 10):
    """Search ArXiv papers."""
    from agents.arxiv_agent import ArXivAgent
    
    print(f"🎓 Searching ArXiv for: {query}")
    
    try:
        agent = ArXivAgent()
        papers = agent.search_papers(query, max_results=max_results, auto_download=download)
        
        print(f"\n📚 Found {len(papers)} papers:")
        
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper.title}")
            print(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
            print(f"   ArXiv ID: {paper.arxiv_id}")
            print(f"   Categories: {', '.join(paper.categories)}")
            print(f"   Published: {paper.published_date.strftime('%Y-%m-%d')}")
            
        if download:
            print(f"\n📥 Papers downloaded to: downloads/arxiv/")
            
    except Exception as e:
        print(f"❌ Error searching ArXiv: {e}")
        sys.exit(1)


def show_config():
    """Show current configuration."""
    from utils.config import get_config
    
    try:
        config = get_config()
        print("⚙️ Current Configuration:")
        print(f"   LLM Provider: {config.llm.provider}")
        print(f"   Vector Store: {config.vector_store.type}")
        print(f"   Embedding Model: {config.vector_store.embedding_model}")
        print(f"   Chunk Size: {config.document_processing.chunk_size}")
        print(f"   Temperature: {config.llm.temperature}")
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
