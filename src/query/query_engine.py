"""
Query engine for intelligent document Q&A using LLMs.

This module provides natural language querying capabilities for documents
with support for different query types and LLM backends.
"""

import json
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from enum import Enum
import re

# LLM integrations
import openai
import google.generativeai as genai
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
# Optional: transformers for local / HF inference
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False
    pipeline = None
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Logging
from loguru import logger

# Configuration and other modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_config
from extraction.vector_store import VectorStoreManager


class QueryType(Enum):
    """Types of queries supported by the system."""
    DIRECT_LOOKUP = "direct_lookup"
    SUMMARIZATION = "summarization"
    DATA_EXTRACTION = "data_extraction"
    COMPARISON = "comparison"
    QUESTION_ANSWER = "question_answer"
    CITATION_LOOKUP = "citation_lookup"


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"
    HUGGINGFACE = "hf"
    LOCAL = "local"


class QueryResult:
    """Structured result from a query operation."""
    
    def __init__(self, query: str, answer: str, query_type: QueryType,
                 confidence: float, sources: List[Dict[str, Any]],
                 metadata: Optional[Dict[str, Any]] = None):
        self.query = query
        self.answer = answer
        self.query_type = query_type
        self.confidence = confidence
        self.sources = sources
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "answer": self.answer,
            "query_type": self.query_type.value,
            "confidence": self.confidence,
            "sources": self.sources,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class PromptTemplates:
    """Collection of prompt templates for different query types."""
    
    QUERY_CLASSIFICATION = """
    Analyze the following user query and classify it into one of these categories:
    
    1. direct_lookup: Looking for specific information, facts, or definitions
    2. summarization: Requesting a summary of content or concepts
    3. data_extraction: Extracting structured data like tables, lists, or specific values
    4. comparison: Comparing concepts, approaches, or items
    5. question_answer: General questions requiring reasoning and explanation
    6. citation_lookup: Looking for references, citations, or source information
    
    Query: "{query}"
    
    Respond with only the category name (e.g., "direct_lookup").
    """
    
    DIRECT_LOOKUP = """
    Based on the following document excerpts, provide a direct and accurate answer to the user's query.
    Focus on factual information and specific details from the documents.
    
    Query: {query}
    
    Document Excerpts:
    {context}
    
    Instructions:
    - Provide a clear, direct answer
    - Use specific information from the documents
    - If the information is not available, state this clearly
    - Include relevant details and examples where available
    
    Answer:
    """
    
    SUMMARIZATION = """
    Create a comprehensive summary based on the following document excerpts related to the user's query.
    
    Query: {query}
    
    Document Excerpts:
    {context}
    
    Instructions:
    - Provide a well-structured summary
    - Include key points and main concepts
    - Organize information logically
    - Highlight important insights or findings
    - Keep the summary focused on the query topic
    
    Summary:
    """
    
    DATA_EXTRACTION = """
    Extract and structure the relevant data from the following document excerpts based on the user's query.
    
    Query: {query}
    
    Document Excerpts:
    {context}
    
    Instructions:
    - Extract specific data points, numbers, or structured information
    - Present data in a clear, organized format (tables, lists, etc.)
    - Include units, dates, and context where relevant
    - If extracting multiple items, organize them systematically
    
    Extracted Data:
    """
    
    COMPARISON = """
    Compare and analyze the concepts, approaches, or items mentioned in the user's query based on the following document excerpts.
    
    Query: {query}
    
    Document Excerpts:
    {context}
    
    Instructions:
    - Identify key similarities and differences
    - Present comparison in a structured format
    - Include advantages and disadvantages where relevant
    - Use specific examples from the documents
    - Provide a balanced analysis
    
    Comparison:
    """
    
    QUESTION_ANSWER = """
    Answer the following question based on the provided document excerpts. Use reasoning and analysis to provide a comprehensive response.
    
    Question: {query}
    
    Document Excerpts:
    {context}
    
    Instructions:
    - Provide a thorough, well-reasoned answer
    - Use evidence from the documents to support your response
    - Explain the reasoning behind your answer
    - Address different aspects of the question
    - If there are limitations or uncertainties, mention them
    
    Answer:
    """
    
    CITATION_LOOKUP = """
    Find and present citation and reference information based on the user's query from the following document excerpts.
    
    Query: {query}
    
    Document Excerpts:
    {context}
    
    Instructions:
    - Extract relevant citations, references, and source information
    - Include authors, titles, publication details where available
    - Organize citations in a standard format
    - Provide context for why each citation is relevant
    
    Citations:
    """


class LLMManager:
    """Manages interactions with different LLM providers."""
    
    def __init__(self, provider: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM manager.
        
        Args:
            provider: LLM provider to use ('openai', 'gemini', etc.)
        """
        self.config = get_config()
        # Allow runtime overrides dictionary to temporarily override config values
        self.overrides = overrides or {}
        # provider string precedence: explicit arg -> overrides -> config file
        provider_str = provider or (self.overrides.get('provider') if isinstance(self.overrides, dict) else None) or self.config.llm.provider
        self.provider = LLMProvider(provider_str)
        self.model = None
        self.chat_model = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM based on the configured provider."""
        try:
            if self.provider == LLMProvider.OPENAI:
                self._initialize_openai()
            elif self.provider == LLMProvider.GEMINI:
                self._initialize_gemini()
            elif self.provider == LLMProvider.HUGGINGFACE or self.provider == LLMProvider.LOCAL:
                self._initialize_huggingface()
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
                
            logger.info(f"LLM initialized: {self.provider.value}")
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise
    
    def _initialize_openai(self):
        """Initialize OpenAI models."""
        # Set API key
        openai.api_key = self.config.llm.openai.api_key
        
        # Initialize models
        self.model = OpenAI(
            model_name=self.config.llm.openai.model,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens
        )
        
        self.chat_model = ChatOpenAI(
            model_name=self.config.llm.openai.model,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens
        )
    
    def _initialize_gemini(self):
        """Initialize Google Gemini models."""
        # Configure API key
        genai.configure(api_key=self.config.llm.gemini.api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(self.config.llm.gemini.model)

    def _initialize_huggingface(self):
        """Initialize Hugging Face / local transformer model pipeline."""
        # Prefer HF config under llm.hf, fallback to llm.local
        # Resolve HF configuration with overrides taking precedence
        hf_override = self.overrides.get('hf') if isinstance(self.overrides, dict) else None

        class _SimpleCfg:
            pass

        hf_cfg = _SimpleCfg()

        # base values from config if present
        base = None
        if self.config.llm.hf:
            base = self.config.llm.hf
        elif self.config.llm.local:
            base = self.config.llm.local

        if base is not None:
            hf_cfg.model = getattr(base, 'model', None)
            hf_cfg.auth_token = getattr(base, 'auth_token', None) if hasattr(base, 'auth_token') else getattr(base, 'auth_token', None) if hasattr(base, 'auth_token') else getattr(base, 'api_key', None)
            hf_cfg.use_local = getattr(base, 'use_local', False)
            hf_cfg.max_tokens = getattr(base, 'max_tokens', 4000)
            hf_cfg.temperature = getattr(base, 'temperature', 0.1)
        else:
            hf_cfg.model = None
            hf_cfg.auth_token = None
            hf_cfg.use_local = False
            hf_cfg.max_tokens = 4000
            hf_cfg.temperature = 0.1

        # apply overrides (if provided)
        if hf_override and isinstance(hf_override, dict):
            if 'model' in hf_override:
                hf_cfg.model = hf_override['model']
            if 'auth_token' in hf_override:
                hf_cfg.auth_token = hf_override['auth_token']
            if 'use_local' in hf_override:
                hf_cfg.use_local = bool(hf_override['use_local'])
            if 'max_tokens' in hf_override:
                hf_cfg.max_tokens = int(hf_override['max_tokens'])
            if 'temperature' in hf_override:
                hf_cfg.temperature = float(hf_override['temperature'])

        if not hf_cfg.model:
            raise ValueError("Hugging Face model id not specified in config or overrides")

        model_id = hf_cfg.model
        self._hf_config = hf_cfg

        # If auth token provided and not using local runtime, prefer HF Inference API via transformers pipelines
        if hf_cfg.auth_token and not hf_cfg.use_local:
            # Use the transformers pipeline with auth token (requires huggingface_hub-backed inference)
            try:
                self.model = pipeline("text-generation", model=model_id, use_auth_token=hf_cfg.auth_token)
            except Exception as e:
                logger.warning(f"Failed to initialize HF pipeline with auth token: {e}; falling back to local if available")
                if _HAS_TRANSFORMERS:
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    model = AutoModelForCausalLM.from_pretrained(model_id)
                    self.model = pipeline("text-generation", model=model, tokenizer=tokenizer)
                else:
                    raise
        else:
            # Local inference using transformers
            if not _HAS_TRANSFORMERS:
                raise RuntimeError("transformers not available; install `transformers` to use local or HF models")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            self.model = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    def generate_text(self, prompt: str, system_message: Optional[str] = None) -> str:
        """
        Generate text using the configured LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            system_message: Optional system message for context
            
        Returns:
            str: Generated text response
        """
        try:
            if self.provider == LLMProvider.OPENAI:
                return self._generate_openai(prompt, system_message)
            elif self.provider == LLMProvider.GEMINI:
                return self._generate_gemini(prompt, system_message)
            elif self.provider == LLMProvider.HUGGINGFACE or self.provider == LLMProvider.LOCAL:
                return self._generate_huggingface(prompt, system_message)
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating response: {str(e)}"
    
    def _generate_openai(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Generate text using OpenAI."""
        if system_message and self.chat_model:
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=prompt)
            ]
            response = self.chat_model(messages)
            return response.content
        else:
            return self.model(prompt)
    
    def _generate_gemini(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Generate text using Gemini."""
        full_prompt = prompt
        if system_message:
            full_prompt = f"{system_message}\n\n{prompt}"
        
        response = self.model.generate_content(full_prompt)
        return response.text

    def _generate_huggingface(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Generate text using the Hugging Face pipeline or local transformers pipeline."""
        full_prompt = prompt
        if system_message:
            full_prompt = f"{system_message}\n\n{prompt}"

        if not self.model:
            raise RuntimeError("Hugging Face model pipeline not initialized")

        tokenizer = getattr(self.model, "tokenizer", None)
        raw_model = getattr(self.model, "model", None)
        pad_token_id = None
        if tokenizer is not None:
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None and getattr(tokenizer, "eos_token_id", None) is not None:
                pad_token_id = tokenizer.eos_token_id

        max_new_tokens = min(self._hf_config.max_tokens, 256)

        # Prefer direct generation via underlying model for more control
        if tokenizer is not None and raw_model is not None:
            try:
                import torch  # type: ignore

                inputs = tokenizer(full_prompt, return_tensors="pt")
                if pad_token_id is not None:
                    raw_model.config.pad_token_id = pad_token_id
                outputs = raw_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                cleaned = decoded[len(full_prompt):] if decoded.startswith(full_prompt) else decoded
                if cleaned.strip():
                    return cleaned.strip()
            except Exception as gen_err:
                logger.debug(f"Direct HF generation failed, falling back to pipeline: {gen_err}")

        # Fallback to pipeline generation
        try:
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "return_full_text": False,
            }
            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = pad_token_id

            output = self.model(full_prompt, **generation_kwargs)
            if isinstance(output, list) and len(output) > 0:
                text = output[0].get('generated_text') or output[0].get('text') or str(output[0])
                cleaned = text[len(full_prompt):] if text.startswith(full_prompt) else text
                if cleaned.strip():
                    return cleaned.strip()
            # Final fallback: allow sampling and full text
            sampling_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "return_full_text": True,
            }
            if pad_token_id is not None:
                sampling_kwargs["pad_token_id"] = pad_token_id
            fallback = self.model(full_prompt, **sampling_kwargs)
            if isinstance(fallback, list) and fallback:
                fb_text = fallback[0].get("generated_text") or fallback[0].get("text") or str(fallback[0])
                fb_cleaned = fb_text[len(full_prompt):] if fb_text.startswith(full_prompt) else fb_text
                return fb_cleaned.strip()
            return ""
        except TypeError:
            # Some pipelines accept different params depending on model
            out = self.model(full_prompt)
            if isinstance(out, list) and len(out) > 0:
                text = out[0].get('generated_text') or out[0].get('text') or str(out[0])
                cleaned = text[len(full_prompt):] if text.startswith(full_prompt) else text
                return cleaned.strip()
            return str(out)


class QueryClassifier:
    """Classifies queries into different types for appropriate processing."""
    
    def __init__(self, llm_manager: LLMManager):
        """
        Initialize the query classifier.
        
        Args:
            llm_manager: LLM manager instance
        """
        self.llm_manager = llm_manager
        self.classification_cache = {}
    
    def classify_query(self, query: str) -> QueryType:
        """
        Classify a query into the appropriate type.
        
        Args:
            query: User query to classify
            
        Returns:
            QueryType: The classified query type
        """
        # Check cache first
        if query in self.classification_cache:
            return self.classification_cache[query]
        
        # Use LLM for classification
        prompt = PromptTemplates.QUERY_CLASSIFICATION.format(query=query)
        
        try:
            response = self.llm_manager.generate_text(prompt)
            classification = response.strip().lower()
            
            # Map response to QueryType
            type_mapping = {
                "direct_lookup": QueryType.DIRECT_LOOKUP,
                "summarization": QueryType.SUMMARIZATION,
                "data_extraction": QueryType.DATA_EXTRACTION,
                "comparison": QueryType.COMPARISON,
                "question_answer": QueryType.QUESTION_ANSWER,
                "citation_lookup": QueryType.CITATION_LOOKUP
            }
            
            query_type = type_mapping.get(classification, QueryType.QUESTION_ANSWER)
            
            # Cache the result
            self.classification_cache[query] = query_type
            
            logger.info(f"Classified query as: {query_type.value}")
            return query_type
            
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            return QueryType.QUESTION_ANSWER  # Default fallback


class QueryProcessor:
    """Processes queries using appropriate templates and strategies."""
    
    def __init__(self, llm_manager: LLMManager):
        """
        Initialize the query processor.
        
        Args:
            llm_manager: LLM manager instance
        """
        self.llm_manager = llm_manager
        self.templates = PromptTemplates()
    
    def process_query(self, query: str, query_type: QueryType, 
                     context: str, **kwargs) -> Tuple[str, float]:
        """
        Process a query with the appropriate template and strategy.
        
        Args:
            query: User query
            query_type: Type of query
            context: Relevant document context
            **kwargs: Additional parameters
            
        Returns:
            Tuple[str, float]: (answer, confidence_score)
        """
        # Select appropriate template
        template = self._get_template(query_type)

        # Truncate context to avoid exceeding provider limits
        context = self._truncate_context(context)

        # Format prompt
        prompt = template.format(query=query, context=context)

        # Generate response
        response = self.llm_manager.generate_text(prompt)

        # Clean up formatting artifacts but otherwise return model output as-is
        response = self._clean_response(response)

        # Calculate confidence score
        confidence = self._calculate_confidence(query, response, context)

        return response, confidence

    def _truncate_context(self, context: str) -> str:
        """Reduce context length to stay within model limits."""
        if not context:
            return context

        max_chars = 4000
        if self.llm_manager.provider in (LLMProvider.HUGGINGFACE, LLMProvider.LOCAL):
            max_chars = 2000

        if len(context) <= max_chars:
            return context

        sections = context.split("\n---\n")
        truncated_sections = []
        total = 0
        sep_len = len("\n---\n")

        for section in sections:
            section = section.strip()
            if not section:
                continue

            section_len = len(section)
            if total + section_len <= max_chars:
                truncated_sections.append(section)
                total += section_len + sep_len
            else:
                remaining = max_chars - total
                if remaining > 0:
                    truncated_sections.append(section[:remaining])
                break

        return "\n---\n".join(truncated_sections)

    def _clean_response(self, response: str) -> str:
        """Strip context headers and artifacts from the model response."""
        if not response:
            return response

        lines = response.splitlines()
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append("")
                continue
            if stripped.lower().startswith("source "):
                continue
            if stripped == "---":
                continue
            cleaned_lines.append(line)

        cleaned_text = "\n".join(cleaned_lines).strip()

        # Reduce duplicated "Answer:" prefixes
        while cleaned_text.lower().startswith("answer:"):
            cleaned_text = cleaned_text[len("answer:"):].strip()

        # Collapse repeated sentences (e.g., the model repeating the same sentence many times)
        try:
            sentences = [s.strip() for s in cleaned_text.split('.') if s.strip()]
            seen = set()
            unique_sentences = []
            for s in sentences:
                # Limit sentence length considered for deduping to avoid huge memory use
                key = s[:200]
                if key not in seen:
                    unique_sentences.append(s)
                    seen.add(key)
            if unique_sentences:
                cleaned_text = '. '.join(unique_sentences).strip()
                if cleaned_text and not cleaned_text.endswith('.'):
                    cleaned_text += '.'
        except Exception:
            # If anything goes wrong, fall back to original cleaned text
            pass

        return cleaned_text or response.strip()
    
    def _get_template(self, query_type: QueryType) -> str:
        """Get the appropriate prompt template for the query type."""
        template_mapping = {
            QueryType.DIRECT_LOOKUP: self.templates.DIRECT_LOOKUP,
            QueryType.SUMMARIZATION: self.templates.SUMMARIZATION,
            QueryType.DATA_EXTRACTION: self.templates.DATA_EXTRACTION,
            QueryType.COMPARISON: self.templates.COMPARISON,
            QueryType.QUESTION_ANSWER: self.templates.QUESTION_ANSWER,
            QueryType.CITATION_LOOKUP: self.templates.CITATION_LOOKUP
        }
        
        return template_mapping.get(query_type, self.templates.QUESTION_ANSWER)
    
    def _calculate_confidence(self, query: str, response: str, context: str) -> float:
        """
        Calculate confidence score for the response.
        
        Args:
            query: Original query
            response: Generated response
            context: Document context used
            
        Returns:
            float: Confidence score between 0 and 1
        """
        confidence = 0.5  # Base confidence
        
        # Factor 1: Response length and completeness
        if len(response) > 50:
            confidence += 0.1
        if len(response) > 200:
            confidence += 0.1
        
        # Factor 2: Presence of specific information
        if any(word in response.lower() for word in ["specific", "according to", "based on"]):
            confidence += 0.1
        
        # Factor 3: Context relevance
        query_words = set(query.lower().split())
        context_words = set(context.lower().split())
        response_words = set(response.lower().split())
        
        # Overlap between query and response
        query_response_overlap = len(query_words & response_words) / max(len(query_words), 1)
        confidence += query_response_overlap * 0.2
        
        # Factor 4: Avoid uncertainty indicators
        uncertainty_indicators = ["not sure", "unclear", "don't know", "uncertain", "maybe"]
        if any(indicator in response.lower() for indicator in uncertainty_indicators):
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))


class DocumentQueryEngine:
    """Main query engine for document Q&A."""
    
    def __init__(self, llm_provider: Optional[str] = None, 
                 vector_store_type: Optional[str] = None,
                 overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize the document query engine.
        
        Args:
            llm_provider: LLM provider to use
            vector_store_type: Vector store type to use
        """
        self.config = get_config()
        
        # Initialize components
        self.llm_manager = LLMManager(llm_provider, overrides=overrides)
        self.vector_store = VectorStoreManager(vector_store_type)
        self.query_classifier = QueryClassifier(self.llm_manager)
        self.query_processor = QueryProcessor(self.llm_manager)

        # Conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        logger.info("Document query engine initialized")
    
    def query(self, query: str, filters: Optional[Dict[str, Any]] = None,
              n_results: int = 5) -> QueryResult:
        """
        Process a natural language query against the document collection.
        
        Args:
            query: Natural language query
            filters: Optional metadata filters for document search
            n_results: Number of relevant chunks to retrieve
            
        Returns:
            QueryResult: Structured query result
        """
        logger.info(f"Processing query: '{query[:100]}...'")
        
        try:
            # Step 1: Classify the query
            query_type = self.query_classifier.classify_query(query)
            
            # Step 2: Retrieve relevant documents
            search_results = self.vector_store.search_documents(
                query=query,
                n_results=n_results,
                filters=filters
            )
            
            # Step 3: Prepare context
            context = self._prepare_context(search_results)
            
            # Step 4: Process the query
            answer, confidence = self.query_processor.process_query(
                query=query,
                query_type=query_type,
                context=context
            )
            
            # Step 5: Prepare sources
            sources = self._prepare_sources(search_results)
            
            # Step 6: Create result
            result = QueryResult(
                query=query,
                answer=answer,
                query_type=query_type,
                confidence=confidence,
                sources=sources,
                metadata={
                    "n_sources": len(sources),
                    "filters_applied": filters or {},
                    "processing_time": datetime.now().isoformat()
                }
            )
            
            # Update conversation memory
            self.memory.chat_memory.add_user_message(query)
            self.memory.chat_memory.add_ai_message(answer)
            
            logger.info(f"Query processed successfully. Confidence: {confidence:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            
            # Return error result
            return QueryResult(
                query=query,
                answer=f"Error processing query: {str(e)}",
                query_type=QueryType.QUESTION_ANSWER,
                confidence=0.0,
                sources=[],
                metadata={"error": str(e)}
            )
    
    def _prepare_context(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Prepare context from search results for LLM processing.
        
        Args:
            search_results: Results from vector search
            
        Returns:
            str: Formatted context string
        """
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            score = result.get("score", 0)
            
            # Format source information
            source_info = f"Source {i}"
            if metadata.get("file_name"):
                source_info += f" ({metadata['file_name']}"
                if metadata.get("page_number"):
                    source_info += f", Page {metadata['page_number']}"
                source_info += ")"
            
            if metadata.get("section_title"):
                source_info += f" - {metadata['section_title']}"
            
            source_info += f" [Relevance: {score:.3f}]"
            
            context_part = f"{source_info}:\n{content}\n"
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)
    
    def _prepare_sources(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare source information for the query result.
        
        Args:
            search_results: Results from vector search
            
        Returns:
            List[Dict[str, Any]]: Formatted source information
        """
        sources = []
        
        for result in search_results:
            metadata = result.get("metadata", {})
            
            source = {
                "content_preview": result.get("content", "")[:200] + "...",
                "file_name": metadata.get("file_name", "Unknown"),
                "file_path": metadata.get("file_path", ""),
                "page_number": metadata.get("page_number"),
                "section_title": metadata.get("section_title"),
                "relevance_score": result.get("score", 0),
                "chunk_id": result.get("id", "")
            }
            
            sources.append(source)
        
        return sources
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        messages = self.memory.chat_memory.messages
        history = []
        
        for message in messages:
            if isinstance(message, HumanMessage):
                history.append({"type": "human", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"type": "ai", "content": message.content})
        
        return history
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.memory.clear()
        logger.info("Conversation history cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get query engine statistics."""
        vector_stats = self.vector_store.get_stats()
        
        return {
            "llm_provider": self.llm_manager.provider.value,
            "vector_store": vector_stats,
            "conversation_length": len(self.memory.chat_memory.messages),
            "classification_cache_size": len(self.query_classifier.classification_cache)
        }


if __name__ == "__main__":
    # Example usage
    engine = DocumentQueryEngine()
    
    # Example queries
    queries = [
        "What is machine learning?",
        "Summarize the main findings of the research",
        "Extract all the performance metrics mentioned",
        "Compare the different algorithms discussed",
        "What are the references for deep learning papers?"
    ]
    
    for query in queries:
        result = engine.query(query)
        print(f"\nQuery: {result.query}")
        print(f"Type: {result.query_type.value}")
        print(f"Answer: {result.answer[:200]}...")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Sources: {len(result.sources)}")
