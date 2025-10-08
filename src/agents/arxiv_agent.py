"""
ArXiv API integration for automatic paper lookup and retrieval.

This module provides functionality to search and retrieve academic papers
from ArXiv based on user descriptions and queries.
"""

import re
import json
import requests
import feedparser
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
import tempfile
import os

# Logging
from loguru import logger

# Configuration and other modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_config
from ingestion.document_processor import DocumentProcessor
from extraction.vector_store import VectorStoreManager


@dataclass
class ArXivPaper:
    """Represents an ArXiv paper with metadata."""
    
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published_date: datetime
    updated_date: Optional[datetime]
    categories: List[str]
    doi: Optional[str]
    pdf_url: str
    abstract_url: str
    comment: Optional[str]
    journal_ref: Optional[str]
    primary_category: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "published_date": self.published_date.isoformat(),
            "updated_date": self.updated_date.isoformat() if self.updated_date else None,
            "categories": self.categories,
            "doi": self.doi,
            "pdf_url": self.pdf_url,
            "abstract_url": self.abstract_url,
            "comment": self.comment,
            "journal_ref": self.journal_ref,
            "primary_category": self.primary_category
        }


class ArXivSearchBuilder:
    """Builder for constructing ArXiv search queries."""
    
    def __init__(self):
        """Initialize the search builder."""
        self.query_parts = []
    
    def add_title(self, title: str) -> 'ArXivSearchBuilder':
        """Add title search criteria."""
        if title:
            self.query_parts.append(f'ti:"{title}"')
        return self
    
    def add_author(self, author: str) -> 'ArXivSearchBuilder':
        """Add author search criteria."""
        if author:
            self.query_parts.append(f'au:"{author}"')
        return self
    
    def add_abstract(self, keywords: str) -> 'ArXivSearchBuilder':
        """Add abstract search criteria."""
        if keywords:
            self.query_parts.append(f'abs:"{keywords}"')
        return self
    
    def add_category(self, category: str) -> 'ArXivSearchBuilder':
        """Add category search criteria."""
        if category:
            self.query_parts.append(f'cat:{category}')
        return self
    
    def add_all_fields(self, keywords: str) -> 'ArXivSearchBuilder':
        """Add search across all fields."""
        if keywords:
            self.query_parts.append(f'all:"{keywords}"')
        return self
    
    def build(self) -> str:
        """Build the final search query."""
        if not self.query_parts:
            return ""
        
        return " AND ".join(self.query_parts)


class ArXivAPI:
    """Interface for ArXiv API operations."""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self):
        """Initialize the ArXiv API client."""
        self.config = get_config()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DocumentAgent/1.0 (Academic Research Tool)'
        })
    
    def search_papers(self, query: str, max_results: int = 10,
                     start: int = 0, sort_by: str = "relevance") -> List[ArXivPaper]:
        """
        Search for papers on ArXiv.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            start: Starting index for pagination
            sort_by: Sort order ('relevance', 'lastUpdatedDate', 'submittedDate')
            
        Returns:
            List[ArXivPaper]: List of matching papers
        """
        logger.info(f"Searching ArXiv for: '{query}' (max_results={max_results})")
        
        try:
            # Prepare parameters
            params = {
                'search_query': query,
                'start': start,
                'max_results': max_results,
                'sortBy': sort_by,
                'sortOrder': 'descending'
            }
            
            # Make request
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse results
            papers = self._parse_search_results(response.text)
            
            logger.info(f"Found {len(papers)} papers")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
            return []
    
    def get_paper_by_id(self, arxiv_id: str) -> Optional[ArXivPaper]:
        """
        Get a specific paper by ArXiv ID.
        
        Args:
            arxiv_id: ArXiv paper ID (e.g., "1234.5678" or "cs.AI/1234567")
            
        Returns:
            ArXivPaper: Paper details if found, None otherwise
        """
        logger.info(f"Fetching ArXiv paper: {arxiv_id}")
        
        try:
            # Clean ArXiv ID
            clean_id = self._clean_arxiv_id(arxiv_id)
            
            # Search for specific ID
            papers = self.search_papers(f"id:{clean_id}", max_results=1)
            
            if papers:
                return papers[0]
            else:
                logger.warning(f"Paper not found: {arxiv_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching paper {arxiv_id}: {e}")
            return None
    
    def download_pdf(self, paper: ArXivPaper, download_path: str) -> bool:
        """
        Download PDF for a paper.
        
        Args:
            paper: ArXiv paper object
            download_path: Path to save the PDF
            
        Returns:
            bool: True if download successful, False otherwise
        """
        logger.info(f"Downloading PDF for: {paper.title}")
        
        try:
            # Download PDF
            response = self.session.get(paper.pdf_url, timeout=60)
            response.raise_for_status()
            
            # Save to file
            os.makedirs(os.path.dirname(download_path), exist_ok=True)
            
            with open(download_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"PDF downloaded: {download_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            return False
    
    def _parse_search_results(self, xml_content: str) -> List[ArXivPaper]:
        """Parse XML response from ArXiv API."""
        papers = []
        
        try:
            # Parse XML using feedparser for better handling
            feed = feedparser.parse(xml_content)
            
            for entry in feed.entries:
                paper = self._parse_entry(entry)
                if paper:
                    papers.append(paper)
        
        except Exception as e:
            logger.error(f"Error parsing search results: {e}")
        
        return papers
    
    def _parse_entry(self, entry) -> Optional[ArXivPaper]:
        """Parse a single entry from the feed."""
        try:
            # Extract ArXiv ID
            arxiv_id = entry.id.split('/')[-1]
            
            # Extract title
            title = entry.title.strip()
            
            # Extract authors
            authors = []
            if hasattr(entry, 'authors'):
                authors = [author.name for author in entry.authors]
            elif hasattr(entry, 'author'):
                authors = [entry.author]
            
            # Extract abstract
            abstract = entry.summary.strip() if hasattr(entry, 'summary') else ""
            
            # Extract dates
            published_date = datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ")
            updated_date = None
            if hasattr(entry, 'updated'):
                updated_date = datetime.strptime(entry.updated, "%Y-%m-%dT%H:%M:%SZ")
            
            # Extract categories
            categories = []
            primary_category = ""
            
            if hasattr(entry, 'tags'):
                for tag in entry.tags:
                    if hasattr(tag, 'term'):
                        categories.append(tag.term)
                        if not primary_category:
                            primary_category = tag.term
            
            # Extract links
            pdf_url = ""
            abstract_url = ""
            
            if hasattr(entry, 'links'):
                for link in entry.links:
                    if link.type == 'application/pdf':
                        pdf_url = link.href
                    elif link.type == 'text/html':
                        abstract_url = link.href
            
            # Extract optional fields
            doi = getattr(entry, 'arxiv_doi', None)
            comment = getattr(entry, 'arxiv_comment', None)
            journal_ref = getattr(entry, 'arxiv_journal_ref', None)
            
            return ArXivPaper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                published_date=published_date,
                updated_date=updated_date,
                categories=categories,
                doi=doi,
                pdf_url=pdf_url,
                abstract_url=abstract_url,
                comment=comment,
                journal_ref=journal_ref,
                primary_category=primary_category
            )
        
        except Exception as e:
            logger.error(f"Error parsing entry: {e}")
            return None
    
    def _clean_arxiv_id(self, arxiv_id: str) -> str:
        """Clean and normalize ArXiv ID."""
        # Remove any whitespace
        clean_id = arxiv_id.strip()
        
        # Remove "arXiv:" prefix if present
        if clean_id.lower().startswith("arxiv:"):
            clean_id = clean_id[6:]
        
        # Remove version suffix if present (e.g., "v1", "v2")
        clean_id = re.sub(r'v\d+$', '', clean_id)
        
        return clean_id


class ArXivQueryProcessor:
    """Processes natural language queries to generate ArXiv searches."""
    
    def __init__(self, llm_manager=None):
        """Initialize the query processor."""
        self.llm_manager = llm_manager
        
        # Predefined category mappings
        self.category_keywords = {
            'computer science': ['cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'cs.IR'],
            'machine learning': ['cs.LG', 'stat.ML'],
            'artificial intelligence': ['cs.AI'],
            'natural language processing': ['cs.CL'],
            'computer vision': ['cs.CV'],
            'information retrieval': ['cs.IR'],
            'physics': ['physics', 'astro-ph', 'cond-mat', 'hep-ph', 'nucl-th'],
            'mathematics': ['math.AG', 'math.AT', 'math.CA', 'math.CO'],
            'statistics': ['stat.AP', 'stat.CO', 'stat.ME', 'stat.ML'],
            'economics': ['econ.EM', 'econ.GN', 'econ.TH'],
            'quantitative biology': ['q-bio.BM', 'q-bio.CB', 'q-bio.GN'],
            'quantitative finance': ['q-fin.CP', 'q-fin.EC', 'q-fin.GN']
        }
    
    def process_natural_query(self, query: str) -> str:
        """
        Convert a natural language query to ArXiv search syntax.
        
        Args:
            query: Natural language query
            
        Returns:
            str: ArXiv search query
        """
        logger.info(f"Processing natural query: '{query}'")
        
        # Use LLM if available for better query processing
        if self.llm_manager:
            return self._process_with_llm(query)
        else:
            return self._process_with_rules(query)
    
    def _process_with_llm(self, query: str) -> str:
        """Process query using LLM for better understanding."""
        prompt = f"""
        Convert the following natural language query into an ArXiv search query.
        
        ArXiv search syntax:
        - ti:"title keywords" for title search
        - au:"author name" for author search
        - abs:"abstract keywords" for abstract search
        - cat:category for category search
        - all:"general keywords" for all fields search
        
        Common categories: cs.AI, cs.LG, cs.CL, cs.CV, stat.ML, physics, math
        
        User query: "{query}"
        
        Provide only the ArXiv search query:
        """
        
        try:
            response = self.llm_manager.generate_text(prompt)
            search_query = response.strip()
            logger.info(f"LLM generated search: {search_query}")
            return search_query
        except Exception as e:
            logger.error(f"Error using LLM for query processing: {e}")
            return self._process_with_rules(query)
    
    def _process_with_rules(self, query: str) -> str:
        """Process query using rule-based approach."""
        builder = ArXivSearchBuilder()
        query_lower = query.lower()
        
        # Check for author searches
        author_patterns = [
            r'papers? by (.+)',
            r'author[:\s]+(.+)',
            r'(.+)\'s papers?'
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, query_lower)
            if match:
                author = match.group(1).strip()
                builder.add_author(author)
                break
        
        # Check for category searches
        for topic, categories in self.category_keywords.items():
            if topic in query_lower:
                builder.add_category(categories[0])  # Use primary category
                break
        
        # Check for title-specific searches
        title_patterns = [
            r'papers? titled? (.+)',
            r'title[:\s]+(.+)',
            r'"([^"]+)"'  # Quoted strings
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, query)
            if match:
                title = match.group(1).strip()
                builder.add_title(title)
                break
        
        # If no specific patterns found, use general search
        search_query = builder.build()
        if not search_query:
            # Remove common stop words and use as general search
            stop_words = {'papers', 'about', 'on', 'the', 'a', 'an', 'and', 'or', 'but'}
            keywords = [word for word in query.split() if word.lower() not in stop_words]
            if keywords:
                builder.add_all_fields(' '.join(keywords))
                search_query = builder.build()
        
        return search_query


class ArXivAgent:
    """Main agent for ArXiv paper lookup and retrieval."""
    
    def __init__(self, llm_manager=None, vector_store_manager=None):
        """
        Initialize the ArXiv agent.
        
        Args:
            llm_manager: LLM manager for query processing
            vector_store_manager: Vector store for paper storage
        """
        self.config = get_config()
        self.api = ArXivAPI()
        self.query_processor = ArXivQueryProcessor(llm_manager)
        self.document_processor = DocumentProcessor()
        self.vector_store = vector_store_manager
        
        # Create downloads directory
        self.downloads_dir = Path("downloads/arxiv")
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ArXiv agent initialized")
    
    def search_papers(self, query: str, max_results: int = 10,
                     auto_download: bool = False) -> List[ArXivPaper]:
        """
        Search for papers using natural language query.
        
        Args:
            query: Natural language search query
            max_results: Maximum number of results
            auto_download: Whether to automatically download PDFs
            
        Returns:
            List[ArXivPaper]: List of matching papers
        """
        logger.info(f"Searching papers for: '{query}'")
        
        # Convert natural language to ArXiv query
        search_query = self.query_processor.process_natural_query(query)
        
        if not search_query:
            logger.warning("Could not generate search query")
            return []
        
        # Search papers
        papers = self.api.search_papers(search_query, max_results)
        
        # Auto-download if requested
        if auto_download:
            self.download_papers(papers)
        
        return papers
    
    def get_paper_by_id(self, arxiv_id: str, auto_download: bool = False) -> Optional[ArXivPaper]:
        """
        Get a specific paper by ArXiv ID.
        
        Args:
            arxiv_id: ArXiv paper ID
            auto_download: Whether to automatically download PDF
            
        Returns:
            ArXivPaper: Paper if found, None otherwise
        """
        paper = self.api.get_paper_by_id(arxiv_id)
        
        if paper and auto_download:
            self.download_papers([paper])
        
        return paper
    
    def download_papers(self, papers: List[ArXivPaper]) -> List[str]:
        """
        Download PDFs for multiple papers.
        
        Args:
            papers: List of papers to download
            
        Returns:
            List[str]: Paths to downloaded files
        """
        downloaded_paths = []
        
        for paper in papers:
            # Generate filename
            safe_title = re.sub(r'[^\w\s-]', '', paper.title).strip()[:50]
            filename = f"{paper.arxiv_id}_{safe_title}.pdf"
            filepath = self.downloads_dir / filename
            
            # Download PDF
            if self.api.download_pdf(paper, str(filepath)):
                downloaded_paths.append(str(filepath))
                logger.info(f"Downloaded: {paper.title}")
            else:
                logger.error(f"Failed to download: {paper.title}")
        
        return downloaded_paths
    
    def process_and_index_papers(self, papers: List[ArXivPaper]) -> Dict[str, Any]:
        """
        Download, process, and index papers in vector store.
        
        Args:
            papers: List of papers to process
            
        Returns:
            Dict[str, Any]: Processing results
        """
        if not self.vector_store:
            logger.error("Vector store not available for indexing")
            return {"error": "Vector store not available"}
        
        results = {
            "processed": 0,
            "failed": 0,
            "errors": []
        }
        
        # Download papers
        downloaded_paths = self.download_papers(papers)
        
        # Process each downloaded paper
        for i, filepath in enumerate(downloaded_paths):
            try:
                # Process document
                processed_doc = self.document_processor.process_document(filepath)
                
                # Add ArXiv metadata
                paper = papers[i] if i < len(papers) else None
                if paper:
                    processed_doc.metadata.arxiv_id = paper.arxiv_id
                    processed_doc.metadata.authors = paper.authors
                    processed_doc.metadata.categories = paper.categories
                    processed_doc.metadata.published_date = paper.published_date.isoformat()
                    processed_doc.metadata.abstract = paper.abstract
                
                # Add to vector store
                self.vector_store.add_document(processed_doc)
                
                results["processed"] += 1
                logger.info(f"Processed and indexed: {os.path.basename(filepath)}")
                
            except Exception as e:
                error_msg = f"Error processing {filepath}: {str(e)}"
                logger.error(error_msg)
                results["failed"] += 1
                results["errors"].append(error_msg)
        
        return results
    
    def recommend_papers(self, query: str, based_on_papers: List[str] = None) -> List[ArXivPaper]:
        """
        Recommend papers based on query and existing papers.
        
        Args:
            query: Research interest query
            based_on_papers: List of ArXiv IDs to base recommendations on
            
        Returns:
            List[ArXivPaper]: Recommended papers
        """
        logger.info(f"Getting recommendations for: '{query}'")
        
        # Start with basic search
        papers = self.search_papers(query, max_results=20)
        
        # If we have base papers, try to find related work
        if based_on_papers:
            related_papers = []
            
            for arxiv_id in based_on_papers:
                base_paper = self.api.get_paper_by_id(arxiv_id)
                if base_paper:
                    # Search for papers with similar keywords
                    keywords = self._extract_keywords(base_paper.abstract)
                    if keywords:
                        related = self.search_papers(keywords, max_results=5)
                        related_papers.extend(related)
            
            # Combine and deduplicate
            all_papers = papers + related_papers
            unique_papers = {}
            for paper in all_papers:
                unique_papers[paper.arxiv_id] = paper
            
            papers = list(unique_papers.values())
        
        # Sort by relevance (published date as proxy)
        papers.sort(key=lambda p: p.published_date, reverse=True)
        
        return papers[:10]  # Return top 10 recommendations
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> str:
        """Extract keywords from text for search."""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Remove common words
        stop_words = {
            'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been',
            'were', 'said', 'each', 'which', 'their', 'time', 'these', 'than',
            'many', 'some', 'very', 'when', 'much', 'such', 'most', 'even'
        }
        
        keywords = [word for word in words if word not in stop_words]
        
        # Get most frequent keywords
        from collections import Counter
        word_counts = Counter(keywords)
        top_keywords = [word for word, count in word_counts.most_common(max_keywords)]
        
        return ' '.join(top_keywords)


if __name__ == "__main__":
    # Example usage
    agent = ArXivAgent()
    
    # Search for papers
    papers = agent.search_papers("machine learning transformers", max_results=5)
    
    for paper in papers:
        print(f"\nTitle: {paper.title}")
        print(f"Authors: {', '.join(paper.authors)}")
        print(f"ArXiv ID: {paper.arxiv_id}")
        print(f"Categories: {', '.join(paper.categories)}")
        print(f"Abstract: {paper.abstract[:200]}...")
        print(f"PDF URL: {paper.pdf_url}")
    
    # Get specific paper
    specific_paper = agent.get_paper_by_id("1706.03762")  # Attention is All You Need
    if specific_paper:
        print(f"\nSpecific paper: {specific_paper.title}")
    
    # Download papers
    if papers:
        downloaded = agent.download_papers(papers[:2])
        print(f"\nDownloaded {len(downloaded)} papers")
