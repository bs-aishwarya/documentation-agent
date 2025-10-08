"""
Document ingestion and processing pipeline.

This module handles the ingestion of PDF documents and other file types,
extracting content while preserving document structure.
"""

import os
import io
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from datetime import datetime

# Text processing
import re
from sentence_transformers import SentenceTransformer
import spacy

# Logging
from loguru import logger

# Configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_config


@dataclass
class DocumentMetadata:
    """Metadata for a processed document."""
    file_path: str
    file_name: str
    file_size: int
    file_hash: str
    created_at: datetime
    modified_at: datetime
    page_count: int
    document_type: str
    language: str
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[List[str]] = None


@dataclass
class DocumentChunk:
    """A chunk of processed document content."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    document_id: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    chunk_type: str = "text"  # text, table, figure, equation
    bbox: Optional[Tuple[float, float, float, float]] = None  # Bounding box coordinates


@dataclass
class ProcessedDocument:
    """A fully processed document with all extracted content."""
    metadata: DocumentMetadata
    chunks: List[DocumentChunk]
    raw_text: str
    structured_content: Dict[str, Any]
    images: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    equations: List[Dict[str, Any]]


class DocumentExtractor:
    """Handles extraction of content from various document types."""
    
    def __init__(self):
        """Initialize the document extractor."""
        self.config = get_config()
        self.nlp = None
        self._load_nlp_model()
    
    def _load_nlp_model(self):
        """Load spaCy NLP model for text processing."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not found. Some features may be limited.")
            self.nlp = None
    
    def extract_pdf_content(self, file_path: str) -> ProcessedDocument:
        """
        Extract content from a PDF file with structure preservation.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ProcessedDocument: Extracted content and metadata
        """
        logger.info(f"Extracting content from PDF: {file_path}")
        
        # Get document metadata
        metadata = self._extract_metadata(file_path)
        
        # Extract content using multiple methods for best results
        pymupdf_content = self._extract_with_pymupdf(file_path)
        pdfplumber_content = self._extract_with_pdfplumber(file_path)
        
        # Combine and structure the content
        combined_content = self._combine_extraction_results(
            pymupdf_content, pdfplumber_content
        )
        
        # Create document chunks
        chunks = self._create_chunks(combined_content, metadata)
        
        # Extract structured elements
        images = self._extract_images(file_path) if self.config.document_processing.extract_images else []
        tables = self._extract_tables(file_path) if self.config.document_processing.extract_tables else []
        equations = self._extract_equations(combined_content) if self.config.document_processing.extract_equations else []
        
        # Create structured content
        structured_content = self._create_structured_content(combined_content)
        
        return ProcessedDocument(
            metadata=metadata,
            chunks=chunks,
            raw_text=combined_content.get("full_text", ""),
            structured_content=structured_content,
            images=images,
            tables=tables,
            equations=equations
        )
    
    def _extract_metadata(self, file_path: str) -> DocumentMetadata:
        """Extract metadata from a document file."""
        file_stats = os.stat(file_path)
        file_hash = self._calculate_file_hash(file_path)
        
        # Try to extract PDF metadata
        title, author, subject, keywords = None, None, None, None
        page_count = 0
        
        try:
            with fitz.open(file_path) as doc:
                page_count = len(doc)
                meta = doc.metadata
                title = meta.get("title")
                author = meta.get("author")
                subject = meta.get("subject")
                keywords_str = meta.get("keywords")
                if keywords_str:
                    keywords = [k.strip() for k in keywords_str.split(",")]
        except Exception as e:
            logger.warning(f"Could not extract PDF metadata: {e}")
        
        return DocumentMetadata(
            file_path=file_path,
            file_name=Path(file_path).name,
            file_size=file_stats.st_size,
            file_hash=file_hash,
            created_at=datetime.fromtimestamp(file_stats.st_ctime),
            modified_at=datetime.fromtimestamp(file_stats.st_mtime),
            page_count=page_count,
            document_type="pdf",
            language="en",  # Could be detected automatically
            title=title,
            author=author,
            subject=subject,
            keywords=keywords
        )
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of the file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _extract_with_pymupdf(self, file_path: str) -> Dict[str, Any]:
        """Extract content using PyMuPDF (fitz)."""
        content = {
            "pages": [],
            "full_text": "",
            "fonts": set(),
            "structure": []
        }
        
        try:
            with fitz.open(file_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    # Extract text with formatting information
                    blocks = page.get_text("dict")
                    page_content = self._process_pymupdf_blocks(blocks, page_num)
                    
                    content["pages"].append(page_content)
                    content["full_text"] += page_content["text"] + "\n\n"
                    content["fonts"].update(page_content["fonts"])
                    content["structure"].extend(page_content["structure"])
        
        except Exception as e:
            logger.error(f"Error extracting with PyMuPDF: {e}")
        
        content["fonts"] = list(content["fonts"])
        return content
    
    def _process_pymupdf_blocks(self, blocks: Dict, page_num: int) -> Dict[str, Any]:
        """Process PyMuPDF text blocks to extract structured content."""
        page_content = {
            "text": "",
            "fonts": set(),
            "structure": [],
            "page_number": page_num
        }
        
        for block in blocks.get("blocks", []):
            if "lines" in block:  # Text block
                for line in block["lines"]:
                    line_text = ""
                    line_fonts = []
                    
                    for span in line["spans"]:
                        text = span["text"]
                        font = span["font"]
                        size = span["size"]
                        flags = span["flags"]  # Bold, italic, etc.
                        
                        line_text += text
                        line_fonts.append((font, size, flags))
                        page_content["fonts"].add(f"{font}_{size}")
                    
                    if line_text.strip():
                        # Determine if this is a heading based on font size/style
                        is_heading = self._is_heading(line_fonts)
                        
                        page_content["structure"].append({
                            "text": line_text.strip(),
                            "type": "heading" if is_heading else "paragraph",
                            "page": page_num,
                            "bbox": line["bbox"],
                            "fonts": line_fonts
                        })
                        
                        page_content["text"] += line_text.strip() + "\n"
        
        return page_content
    
    def _is_heading(self, fonts: List[Tuple]) -> bool:
        """Determine if text is likely a heading based on font properties."""
        if not fonts:
            return False
        
        # Simple heuristic: larger font size or bold text
        avg_size = sum(font[1] for font in fonts) / len(fonts)
        has_bold = any(font[2] & 2**4 for font in fonts)  # Bold flag
        
        return avg_size > 12 or has_bold
    
    def _extract_with_pdfplumber(self, file_path: str) -> Dict[str, Any]:
        """Extract content using pdfplumber for better table detection."""
        content = {
            "pages": [],
            "tables": [],
            "full_text": ""
        }
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    page_tables = page.extract_tables()
                    
                    content["pages"].append({
                        "text": page_text,
                        "page_number": page_num,
                        "tables": page_tables
                    })
                    
                    content["full_text"] += page_text + "\n\n"
                    
                    # Process tables
                    for table_idx, table in enumerate(page_tables or []):
                        if table and len(table) > 0:
                            content["tables"].append({
                                "page": page_num,
                                "table_id": f"table_{page_num}_{table_idx}",
                                "data": table,
                                "text": self._table_to_text(table)
                            })
        
        except Exception as e:
            logger.error(f"Error extracting with pdfplumber: {e}")
        
        return content
    
    def _table_to_text(self, table: List[List[str]]) -> str:
        """Convert table data to readable text."""
        if not table:
            return ""
        
        text_lines = []
        for row in table:
            if row:  # Skip empty rows
                cleaned_row = [cell.strip() if cell else "" for cell in row]
                if any(cleaned_row):  # Skip completely empty rows
                    text_lines.append(" | ".join(cleaned_row))
        
        return "\n".join(text_lines)
    
    def _combine_extraction_results(self, pymupdf_content: Dict, pdfplumber_content: Dict) -> Dict[str, Any]:
        """Combine results from different extraction methods."""
        combined = {
            "full_text": pdfplumber_content["full_text"],  # pdfplumber usually better for text
            "structure": pymupdf_content.get("structure", []),  # PyMuPDF better for structure
            "tables": pdfplumber_content.get("tables", []),  # pdfplumber better for tables
            "fonts": pymupdf_content.get("fonts", []),
            "pages": []
        }
        
        # Combine page-level information
        pymupdf_pages = pymupdf_content.get("pages", [])
        pdfplumber_pages = pdfplumber_content.get("pages", [])
        
        max_pages = max(len(pymupdf_pages), len(pdfplumber_pages))
        
        for i in range(max_pages):
            page_data = {"page_number": i}
            
            if i < len(pdfplumber_pages):
                page_data["text"] = pdfplumber_pages[i]["text"]
                page_data["tables"] = pdfplumber_pages[i].get("tables", [])
            
            if i < len(pymupdf_pages):
                page_data["structure"] = pymupdf_pages[i].get("structure", [])
                page_data["fonts"] = pymupdf_pages[i].get("fonts", set())
            
            combined["pages"].append(page_data)
        
        return combined
    
    def _create_chunks(self, content: Dict[str, Any], metadata: DocumentMetadata) -> List[DocumentChunk]:
        """Create document chunks for vector storage."""
        chunks = []
        chunk_size = self.config.document_processing.chunk_size
        chunk_overlap = self.config.document_processing.chunk_overlap
        
        full_text = content["full_text"]
        
        # Split text into chunks
        text_chunks = self._split_text(full_text, chunk_size, chunk_overlap)
        
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = f"{metadata.file_hash}_{i}"
            
            # Find which page this chunk belongs to
            page_number = self._find_page_for_chunk(chunk_text, content["pages"])
            
            # Determine chunk type based on content
            chunk_type = self._determine_chunk_type(chunk_text)
            
            chunk = DocumentChunk(
                content=chunk_text,
                metadata={
                    "file_name": metadata.file_name,
                    "file_path": metadata.file_path,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks)
                },
                chunk_id=chunk_id,
                document_id=metadata.file_hash,
                page_number=page_number,
                chunk_type=chunk_type
            )
            
            chunks.append(chunk)
        
        # Add table chunks
        for table in content.get("tables", []):
            chunk_id = f"{metadata.file_hash}_table_{table.get('table_id', 'unknown')}"
            
            chunk = DocumentChunk(
                content=table["text"],
                metadata={
                    "file_name": metadata.file_name,
                    "file_path": metadata.file_path,
                    "table_id": table.get("table_id"),
                    "table_data": table.get("data")
                },
                chunk_id=chunk_id,
                document_id=metadata.file_hash,
                page_number=table.get("page"),
                chunk_type="table"
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []
        
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to break at a sentence boundary
            chunk = text[start:end]
            
            # Look for sentence endings near the end of the chunk
            last_sentence_end = max(
                chunk.rfind('. '),
                chunk.rfind('! '),
                chunk.rfind('? '),
                chunk.rfind('\n\n')
            )
            
            if last_sentence_end > chunk_size * 0.5:  # Only break if we find a good boundary
                chunk = text[start:start + last_sentence_end + 1]
                start = start + last_sentence_end + 1 - overlap
            else:
                start = end - overlap
            
            chunks.append(chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _find_page_for_chunk(self, chunk_text: str, pages: List[Dict]) -> Optional[int]:
        """Find which page a chunk belongs to."""
        # Simple approach: find the page with the most overlap
        best_match_page = None
        best_overlap = 0
        
        for page in pages:
            page_text = page.get("text", "")
            if not page_text:
                continue
            
            # Calculate rough overlap
            chunk_words = set(chunk_text.lower().split())
            page_words = set(page_text.lower().split())
            overlap = len(chunk_words.intersection(page_words))
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_match_page = page.get("page_number")
        
        return best_match_page
    
    def _determine_chunk_type(self, text: str) -> str:
        """Determine the type of content in a chunk."""
        # Simple heuristics to classify content
        if re.search(r'\b(table|column|row)\b', text.lower()) and '|' in text:
            return "table"
        elif re.search(r'\b(figure|fig\.|image|chart|graph)\b', text.lower()):
            return "figure"
        elif re.search(r'[∑∫∂∇α-ωΑ-Ω]|\\[a-zA-Z]+', text):
            return "equation"
        else:
            return "text"
    
    def _extract_images(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract images from PDF."""
        images = []
        
        try:
            with fitz.open(file_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    image_list = page.get_images()
                    
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            
                            images.append({
                                "page": page_num,
                                "image_id": f"img_{page_num}_{img_index}",
                                "xref": xref,
                                "size": len(img_data),
                                "data": img_data,
                                "bbox": img[1:5] if len(img) > 4 else None
                            })
                        
                        pix = None  # Free memory
        
        except Exception as e:
            logger.error(f"Error extracting images: {e}")
        
        return images
    
    def _extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract tables from PDF (already done in pdfplumber extraction)."""
        # This is handled in _extract_with_pdfplumber
        return []
    
    def _extract_equations(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract mathematical equations from content."""
        equations = []
        
        # Look for mathematical content in text
        full_text = content.get("full_text", "")
        
        # Simple regex patterns for mathematical expressions
        math_patterns = [
            r'[∑∫∂∇α-ωΑ-Ω]+[^.]*',  # Greek letters and math symbols
            r'\\[a-zA-Z]+\{[^}]*\}',   # LaTeX commands
            r'\$[^$]+\$',               # Inline math
            r'\$\$[^$]+\$\$',           # Display math
        ]
        
        for pattern in math_patterns:
            matches = re.finditer(pattern, full_text)
            for match in matches:
                equations.append({
                    "equation": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "type": "mathematical_expression"
                })
        
        return equations
    
    def _create_structured_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create structured representation of document content."""
        structured = {
            "sections": [],
            "headings": [],
            "paragraphs": [],
            "tables": content.get("tables", []),
            "metadata": {
                "total_pages": len(content.get("pages", [])),
                "fonts_detected": content.get("fonts", []),
                "structure_elements": len(content.get("structure", []))
            }
        }
        
        # Extract headings and sections from structure
        current_section = None
        
        for element in content.get("structure", []):
            if element["type"] == "heading":
                structured["headings"].append(element)
                
                # Start new section
                if current_section:
                    structured["sections"].append(current_section)
                
                current_section = {
                    "title": element["text"],
                    "page": element["page"],
                    "content": []
                }
            
            elif element["type"] == "paragraph" and current_section:
                current_section["content"].append(element["text"])
                structured["paragraphs"].append(element)
        
        # Add last section
        if current_section:
            structured["sections"].append(current_section)
        
        return structured


class DocumentProcessor:
    """Main document processing pipeline."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.config = get_config()
        self.extractor = DocumentExtractor()
        self.processed_docs: Dict[str, ProcessedDocument] = {}
    
    def process_document(self, file_path: str) -> ProcessedDocument:
        """
        Process a single document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ProcessedDocument: Processed document with all extracted content
        """
        logger.info(f"Processing document: {file_path}")
        
        # Validate file
        if not self._validate_file(file_path):
            raise ValueError(f"Invalid file: {file_path}")
        
        # Check if already processed (by hash)
        file_hash = self.extractor._calculate_file_hash(file_path)
        if file_hash in self.processed_docs:
            logger.info(f"Document already processed: {file_path}")
            return self.processed_docs[file_hash]
        
        # Extract content based on file type
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == ".pdf":
            processed_doc = self.extractor.extract_pdf_content(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Cache the processed document
        self.processed_docs[file_hash] = processed_doc
        
        logger.info(f"Successfully processed document: {file_path}")
        logger.info(f"Extracted {len(processed_doc.chunks)} chunks, "
                   f"{len(processed_doc.tables)} tables, "
                   f"{len(processed_doc.images)} images")
        
        return processed_doc
    
    def process_directory(self, directory_path: str, recursive: bool = True) -> List[ProcessedDocument]:
        """
        Process all documents in a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to process subdirectories
            
        Returns:
            List[ProcessedDocument]: List of processed documents
        """
        logger.info(f"Processing directory: {directory_path}")
        
        # Find all supported files
        supported_extensions = [f".{ext}" for ext in self.config.document_processing.supported_formats]
        file_paths = []
        
        directory = Path(directory_path)
        if recursive:
            for ext in supported_extensions:
                file_paths.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in supported_extensions:
                file_paths.extend(directory.glob(f"*{ext}"))
        
        # Process files
        processed_docs = []
        
        # Use thread pool for parallel processing
        max_workers = min(4, len(file_paths))  # Limit to avoid overwhelming system
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_document, str(file_path)): file_path
                for file_path in file_paths
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    processed_doc = future.result()
                    processed_docs.append(processed_doc)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        logger.info(f"Processed {len(processed_docs)} documents from directory")
        return processed_docs
    
    def _validate_file(self, file_path: str) -> bool:
        """Validate if file can be processed."""
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False
        
        # Check file extension
        ext = path.suffix.lower()
        supported_exts = [f".{fmt}" for fmt in self.config.document_processing.supported_formats]
        if ext not in supported_exts:
            logger.error(f"Unsupported file type: {ext}")
            return False
        
        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        max_size_mb = self.config.document_processing.max_file_size
        if file_size_mb > max_size_mb:
            logger.error(f"File too large: {file_size_mb:.1f}MB > {max_size_mb}MB")
            return False
        
        return True
    
    def get_processed_document(self, file_hash: str) -> Optional[ProcessedDocument]:
        """Get a processed document by its hash."""
        return self.processed_docs.get(file_hash)
    
    def list_processed_documents(self) -> List[DocumentMetadata]:
        """Get metadata for all processed documents."""
        return [doc.metadata for doc in self.processed_docs.values()]


if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor()
    
    # Process a single document
    try:
        doc = processor.process_document("documents/sample.pdf")
        print(f"Processed: {doc.metadata.file_name}")
        print(f"Chunks: {len(doc.chunks)}")
        print(f"Tables: {len(doc.tables)}")
        print(f"Images: {len(doc.images)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Process a directory
    try:
        docs = processor.process_directory("documents/")
        print(f"Processed {len(docs)} documents from directory")
    except Exception as e:
        print(f"Error: {e}")
