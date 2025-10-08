"""
Web interface for the Document Q&A system using Streamlit.

This module provides a user-friendly web interface for interacting
with the document query engine.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
import json
from datetime import datetime, timedelta
import os
from pathlib import Path
import logging

# Configuration and components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from query.query_engine import DocumentQueryEngine, QueryResult, QueryType
from extraction.vector_store import VectorStoreManager
from ingestion.document_processor import DocumentProcessor
from utils.config import get_config

logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="Document Q&A Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .query-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .result-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    
    .source-item {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
    
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


class DocumentQAInterface:
    """Main interface class for the Document Q&A system."""
    
    def __init__(self):
        """Initialize the interface components."""
        self.config = get_config()
        
        # Initialize session state
        if 'query_engine' not in st.session_state:
            st.session_state.query_engine = None
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        if 'uploaded_documents' not in st.session_state:
            st.session_state.uploaded_documents = []
        if 'session_document_ids' not in st.session_state:
            st.session_state.session_document_ids = []  # Track document IDs for current session
    
    def render_header(self):
        """Render the main header."""
        st.markdown('<h1 class="main-header">📚 Enterprise Document Q&A Agent</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        Welcome to the **Enterprise Document Q&A Agent**! This intelligent system allows you to:
        
        - 📄 Upload and process PDF documents
        - 🔍 Ask natural language questions about your documents
        - 📊 Extract structured data and insights
        - 📚 Compare information across multiple documents
        - 🎯 Get precise answers with confidence scoring
        """)
    
    def render_sidebar(self):
        """Render the sidebar with controls and information."""
        with st.sidebar:
            st.header("🔧 System Controls")
            
            # Initialize system button
            if st.button("🚀 Initialize System", type="primary"):
                self.initialize_system()
            
            st.divider()
            
            # Document upload section
            st.header("📄 Document Management")
            
            uploaded_files = st.file_uploader(
                "Upload PDF documents",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload PDF documents to add to the knowledge base"
            )
            
            if uploaded_files:
                if st.button("📥 Process Documents"):
                    self.process_uploaded_documents(uploaded_files)
            
            # Session management
            st.markdown("---")
            st.subheader("🔄 Session Management")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Session", help="Clear current session documents"):
                    st.session_state.session_document_ids = []
                    st.success("Session cleared! Previously uploaded documents can still be queried.")
            
            with col2:
                if st.button("⚠️ Reset DB", type="secondary", 
                           help="Delete ALL documents from vector database"):
                    if st.session_state.query_engine:
                        try:
                            # Delete all documents
                            for doc in st.session_state.uploaded_documents:
                                doc_id = doc.get("document_id")
                                if doc_id:
                                    st.session_state.query_engine.vector_store.delete_document(doc_id)
                            
                            # Clear tracking
                            st.session_state.uploaded_documents = []
                            st.session_state.session_document_ids = []
                            st.success("✅ Vector database cleared!")
                        except Exception as e:
                            st.error(f"❌ Error clearing database: {str(e)}")
                    else:
                        st.warning("⚠️ Initialize system first")
            
            st.divider()
            
            # System configuration
            st.header("⚙️ Configuration")
            
            llm_provider = st.selectbox(
                "LLM Provider",
                options=["openai", "gemini", "hf", "local"],
                index=2,
                help="Select the language model provider"
            )

            # If using Hugging Face / local, allow model selection
            if llm_provider in ("hf", "local"):
                hf_model = st.text_input(
                    "HF Model Identifier",
                    value=(self.config.llm.hf.model if self.config.llm.hf else "gpt2"),
                    help="Hugging Face model id (e.g., gpt2, google/flan-t5-base) or local model path"
                )
                hf_use_local = st.checkbox(
                    "Use local transformers runtime",
                    value=(self.config.llm.hf.use_local if self.config.llm.hf else False),
                    help="Run the model with local transformers (requires transformers installed and model files available)"
                )
                # persist selections to config (session-level)
                st.session_state['llm_provider'] = llm_provider
                st.session_state['hf_model'] = hf_model
                st.session_state['hf_use_local'] = hf_use_local
            
            vector_store_type = st.selectbox(
                "Vector Store",
                options=["chroma", "faiss"],
                index=0,
                help="Select the vector database backend"
            )
            
            max_results = st.slider(
                "Max Search Results",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of relevant chunks to retrieve"
            )
            
            st.divider()
            
            # System statistics
            self.render_system_stats()
    
    def render_main_interface(self):
        """Render the main query interface."""
        # Check if system is initialized
        if st.session_state.query_engine is None:
            st.warning("⚠️ Please initialize the system first using the sidebar.")
            return
        
        # Query input section
        st.header("💬 Ask Your Questions")
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_input(
                "Enter your question:",
                placeholder="e.g., What are the main findings of the research?",
                help="Ask any question about your uploaded documents"
            )
        
        with col2:
            search_button = st.button("🔍 Search", type="primary")
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                query_type_filter = st.selectbox(
                    "Query Type (Optional)",
                    options=["Auto-detect"] + [qt.value for qt in QueryType],
                    help="Manually specify the query type"
                )
            
            with col2:
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.1,
                    help="Minimum confidence score for results"
                )
            
            # Document filters
            document_filter = []
            if st.session_state.uploaded_documents:
                # Default to only session documents if any exist
                default_docs = [doc["name"] for doc in st.session_state.uploaded_documents 
                               if doc.get("document_id") in st.session_state.session_document_ids]
                
                document_filter = st.multiselect(
                    "🔍 Query Documents",
                    options=[doc["name"] for doc in st.session_state.uploaded_documents],
                    default=default_docs,
                    help="Select which documents to search. Default: only documents uploaded this session."
                )
        
        # Process query
        if search_button and query:
            # Build document_id filters based on selected documents
            selected_doc_ids = None
            if document_filter:
                selected_doc_ids = [
                    doc["document_id"] for doc in st.session_state.uploaded_documents
                    if doc["name"] in document_filter
                ]
            
            self.process_query(query, confidence_threshold, selected_doc_ids)
        
        # Display query history
        if st.session_state.query_history:
            st.header("📝 Query History")
            
            # Show recent queries in tabs
            if len(st.session_state.query_history) > 0:
                recent_queries = st.session_state.query_history[-5:]  # Last 5 queries
                
                for i, result in enumerate(reversed(recent_queries), 1):
                    with st.expander(f"Query {i}: {result.query[:50]}..."):
                        self.render_query_result(result)
    
    def initialize_system(self):
        """Initialize the query engine and vector store."""
        try:
            with st.spinner("🔄 Initializing system..."):
                # Prepare overrides from UI selections (if any)
                overrides = {}
                provider = st.session_state.get('llm_provider') or self.config.llm.provider
                if provider in ("hf", "local"):
                    overrides['hf'] = {
                        'model': st.session_state.get('hf_model') or (self.config.llm.hf.model if self.config.llm.hf else None),
                        'use_local': bool(st.session_state.get('hf_use_local'))
                    }
                    overrides['provider'] = provider

                # Initialize query engine with overrides
                st.session_state.query_engine = DocumentQueryEngine(llm_provider=provider, overrides=overrides)
                
            st.success("✅ System initialized successfully!")
            
        except Exception as e:
            st.error(f"❌ Error initializing system: {str(e)}")
    
    def process_uploaded_documents(self, uploaded_files):
        """Process uploaded PDF documents."""
        if not st.session_state.query_engine:
            st.error("❌ Please initialize the system first.")
            return
        
        try:
            processor = DocumentProcessor()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # Process document
                    processed_doc = processor.process_document(temp_path)
                    
                    # Add to vector store
                    st.session_state.query_engine.vector_store.add_document(processed_doc)
                    
                    # Track uploaded document with document_id
                    doc_info = {
                        "name": uploaded_file.name,
                        "size": uploaded_file.size,
                        "upload_time": datetime.now(),
                        "chunks": len(processed_doc.chunks),
                        "document_id": processed_doc.metadata.file_hash  # Store document ID
                    }
                    st.session_state.uploaded_documents.append(doc_info)
                    st.session_state.session_document_ids.append(processed_doc.metadata.file_hash)
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("✅ All documents processed successfully!")
            st.success(f"📚 Processed {len(uploaded_files)} documents successfully!")
            
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")
    
    def process_query(self, query: str, confidence_threshold: float, 
                     document_ids: Optional[List[str]] = None):
        """Process a user query and display results."""
        try:
            with st.spinner("🤔 Processing your question..."):
                # Build filters for document IDs
                filters = None
                if document_ids:
                    # ChromaDB supports $in operator for filtering by multiple values
                    filters = {"document_id": {"$in": document_ids}}
                    logger_msg = f"Filtering by {len(document_ids)} document(s)"
                    st.info(f"🔍 {logger_msg}")
                
                # Execute query with filters
                result = st.session_state.query_engine.query(query, filters=filters)
                
                # Filter by confidence threshold
                if result.confidence >= confidence_threshold:
                    # Add to history
                    st.session_state.query_history.append(result)
                    
                    # Display result
                    self.render_query_result(result)
                else:
                    st.warning(f"⚠️ Result confidence ({result.confidence:.2f}) is below threshold ({confidence_threshold:.2f})")
                    st.info("💡 Try rephrasing your question or lowering the confidence threshold.")
        
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
    
    def render_query_result(self, result: QueryResult):
        """Render a query result with formatting."""
        # Main result container
        with st.container():
            # Query information
            st.markdown(f'<div class="query-box"><strong>Query:</strong> {result.query}</div>', 
                       unsafe_allow_html=True)
            
            # Result metadata
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Query Type", result.query_type.value.replace('_', ' ').title())
            
            with col2:
                confidence_class = self.get_confidence_class(result.confidence)
                st.markdown(f'<div class="metric-card">Confidence<br><span class="{confidence_class}">{result.confidence:.2f}</span></div>', 
                           unsafe_allow_html=True)
            
            with col3:
                st.metric("Sources Found", len(result.sources))
            
            with col4:
                st.metric("Timestamp", result.timestamp.strftime("%H:%M:%S"))
            
            # Answer
            st.markdown(f'<div class="result-box"><strong>Answer:</strong><br>{result.answer}</div>', 
                       unsafe_allow_html=True)
            
            # Sources
            if result.sources:
                st.subheader("📖 Sources")
                
                for i, source in enumerate(result.sources, 1):
                    with st.expander(f"Source {i}: {source['file_name']} (Score: {source['relevance_score']:.3f})"):
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Content Preview:**\n{source['content_preview']}")
                        
                        with col2:
                            if source.get('page_number'):
                                st.markdown(f"**Page:** {source['page_number']}")
                            if source.get('section_title'):
                                st.markdown(f"**Section:** {source['section_title']}")
                            st.markdown(f"**Relevance:** {source['relevance_score']:.3f}")
    
    def get_confidence_class(self, confidence: float) -> str:
        """Get CSS class for confidence score."""
        if confidence >= 0.7:
            return "confidence-high"
        elif confidence >= 0.4:
            return "confidence-medium"
        else:
            return "confidence-low"
    
    def render_system_stats(self):
        """Render system statistics in the sidebar."""
        st.header("📊 System Statistics")
        
        if st.session_state.query_engine:
            try:
                stats = st.session_state.query_engine.get_statistics()
                
                st.metric("Documents", len(st.session_state.uploaded_documents))
                st.metric("Total Chunks", stats.get("vector_store", {}).get("total_chunks", 0))
                st.metric("Queries Made", len(st.session_state.query_history))
                
                # Document breakdown
                if st.session_state.uploaded_documents:
                    st.subheader("📄 Documents")
                    for doc in st.session_state.uploaded_documents[-3:]:  # Show last 3
                        st.caption(f"• {doc['name']} ({doc['chunks']} chunks)")
                
            except Exception as e:
                st.caption(f"Stats unavailable: {str(e)}")
        else:
            st.caption("System not initialized")
    
    def render_analytics_page(self):
        """Render analytics and insights page."""
        st.header("📈 Analytics & Insights")
        
        if not st.session_state.query_history:
            st.info("📝 No queries yet. Start asking questions to see analytics!")
            return
        
        # Query statistics
        col1, col2 = st.columns(2)
        
        with col1:
            # Query types distribution
            query_types = [q.query_type.value for q in st.session_state.query_history]
            type_counts = pd.Series(query_types).value_counts()
            
            fig = px.pie(values=type_counts.values, names=type_counts.index, 
                        title="Query Types Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence scores over time
            timestamps = [q.timestamp for q in st.session_state.query_history]
            confidences = [q.confidence for q in st.session_state.query_history]
            
            fig = px.line(x=timestamps, y=confidences, 
                         title="Confidence Scores Over Time",
                         labels={"x": "Time", "y": "Confidence"})
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent query performance
        st.subheader("🎯 Recent Query Performance")
        
        recent_queries = st.session_state.query_history[-10:]  # Last 10 queries
        query_data = []
        
        for q in recent_queries:
            query_data.append({
                "Query": q.query[:50] + "..." if len(q.query) > 50 else q.query,
                "Type": q.query_type.value.replace('_', ' ').title(),
                "Confidence": q.confidence,
                "Sources": len(q.sources),
                "Time": q.timestamp.strftime("%H:%M:%S")
            })
        
        df = pd.DataFrame(query_data)
        st.dataframe(df, use_container_width=True)


def main():
    """Main application function."""
    # Initialize interface
    interface = DocumentQAInterface()
    
    # Render header
    interface.render_header()
    
    # Create tabs
    tab1, tab2 = st.tabs(["💬 Q&A Interface", "📈 Analytics"])
    
    # Render sidebar
    interface.render_sidebar()
    
    with tab1:
        interface.render_main_interface()
    
    with tab2:
        interface.render_analytics_page()


if __name__ == "__main__":
    main()
