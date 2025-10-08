"""
Configuration management for the Document Q&A AI Agent.

This module handles loading and validation of configuration settings
from YAML files and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    provider: str = Field(..., description="LLM provider (openai, gemini, hf, local)")
    
    class OpenAIConfig(BaseModel):
        api_key: str = Field(..., description="OpenAI API key")
        model: str = Field(default="gpt-4-turbo-preview", description="Model name")
        temperature: float = Field(default=0.1, ge=0.0, le=2.0)
        max_tokens: int = Field(default=4000, gt=0)
    
    class GeminiConfig(BaseModel):
        api_key: str = Field(..., description="Gemini API key")
        model: str = Field(default="gemini-pro", description="Model name")
        temperature: float = Field(default=0.1, ge=0.0, le=2.0)
        max_tokens: int = Field(default=4000, gt=0)

    class HuggingFaceConfig(BaseModel):
        model: str = Field(default="tiiuae/falcon-7b-instruct", description="Hugging Face or local model identifier")
        auth_token: Optional[str] = Field(default=None, description="Hugging Face auth token if using Hugging Face API")
        use_local: bool = Field(default=False, description="If true, use local model runtime (transformers/accelerate) instead of Hugging Face hosted inference")
        temperature: float = Field(default=0.1, ge=0.0, le=2.0)
        max_tokens: int = Field(default=4000, gt=0)
    
    openai: Optional[OpenAIConfig] = None
    gemini: Optional[GeminiConfig] = None
    hf: Optional[HuggingFaceConfig] = None
    local: Optional[HuggingFaceConfig] = None
    
    @validator('provider')
    def validate_provider(cls, v):
        allowed = ['openai', 'gemini', 'hf', 'local']
        if v not in allowed:
            raise ValueError(f'Provider must be one of {allowed}')
        return v


class DocumentProcessingConfig(BaseModel):
    """Configuration for document processing."""
    supported_formats: list = Field(default=["pdf", "docx", "txt"])
    max_file_size: int = Field(default=100, description="Max file size in MB")
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    ocr_enabled: bool = Field(default=True)
    ocr_language: str = Field(default="eng")
    extract_images: bool = Field(default=True)
    extract_tables: bool = Field(default=True)
    extract_equations: bool = Field(default=True)


class VectorStoreConfig(BaseModel):
    """Configuration for vector storage."""
    type: str = Field(default="chroma", description="Vector store type")
    
    class ChromaConfig(BaseModel):
        persist_directory: str = Field(default="storage/vectordb")
        collection_name: str = Field(default="documents")
    
    class FAISSConfig(BaseModel):
        index_path: str = Field(default="storage/faiss_index")
    
    chroma: Optional[ChromaConfig] = ChromaConfig()
    faiss: Optional[FAISSConfig] = FAISSConfig()
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dimension: int = Field(default=384, gt=0)


class QueryConfig(BaseModel):
    """Configuration for query processing."""
    max_relevant_chunks: int = Field(default=5, gt=0)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_response_length: int = Field(default=2000, gt=0)
    include_sources: bool = Field(default=True)
    
    class QueryTypes(BaseModel):
        class DirectLookup(BaseModel):
            enabled: bool = True
        
        class Summarization(BaseModel):
            enabled: bool = True
            max_summary_length: int = 500
        
        class DataExtraction(BaseModel):
            enabled: bool = True
            structured_output: bool = True
        
        direct_lookup: DirectLookup = DirectLookup()
        summarization: Summarization = Summarization()
        data_extraction: DataExtraction = DataExtraction()
    
    query_types: QueryTypes = QueryTypes()


class ArxivConfig(BaseModel):
    """Configuration for ArXiv integration."""
    enabled: bool = Field(default=True)
    max_results: int = Field(default=10, gt=0)
    download_papers: bool = Field(default=True)
    download_directory: str = Field(default="documents/arxiv")


class StorageConfig(BaseModel):
    """Configuration for storage."""
    base_directory: str = Field(default="storage")
    documents_directory: str = Field(default="documents")
    processed_directory: str = Field(default="storage/processed")
    cache_enabled: bool = Field(default=True)
    cache_directory: str = Field(default="storage/cache")
    cache_expiry_days: int = Field(default=30, gt=0)


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = Field(default="INFO")
    file: str = Field(default="logs/document_agent.log")
    format: str = Field(default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}")
    rotation: str = Field(default="10 MB")
    retention: str = Field(default="1 month")


class PerformanceConfig(BaseModel):
    """Configuration for performance settings."""
    enable_async: bool = Field(default=True)
    max_concurrent_requests: int = Field(default=10, gt=0)
    batch_size: int = Field(default=100, gt=0)
    max_memory_usage: str = Field(default="4GB")
    
    class RateLimit(BaseModel):
        requests_per_minute: int = Field(default=60, gt=0)
        requests_per_hour: int = Field(default=1000, gt=0)
    
    rate_limit: RateLimit = RateLimit()


class SecurityConfig(BaseModel):
    """Configuration for security settings."""
    encrypt_api_keys: bool = Field(default=False)
    validate_inputs: bool = Field(default=True)
    sanitize_queries: bool = Field(default=True)
    allowed_file_extensions: list = Field(default=[".pdf", ".docx", ".txt"])
    scan_uploads: bool = Field(default=True)


class UIConfig(BaseModel):
    """Configuration for user interface."""
    type: str = Field(default="streamlit")
    host: str = Field(default="localhost")
    port: int = Field(default=8501, gt=0, lt=65536)
    theme: str = Field(default="light")
    title: str = Field(default="Document Q&A AI Agent")
    enable_file_upload: bool = Field(default=True)
    enable_arxiv_search: bool = Field(default=True)
    max_file_uploads: int = Field(default=10, gt=0)


class DevelopmentConfig(BaseModel):
    """Configuration for development settings."""
    debug: bool = Field(default=False)
    reload: bool = Field(default=True)
    testing: bool = Field(default=False)
    mock_llm: bool = Field(default=False)
    mock_embeddings: bool = Field(default=False)


class Config(BaseModel):
    """Main configuration class."""
    llm: LLMConfig
    document_processing: DocumentProcessingConfig = DocumentProcessingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    query: QueryConfig = QueryConfig()
    arxiv: ArxivConfig = ArxivConfig()
    storage: StorageConfig = StorageConfig()
    logging: LoggingConfig = LoggingConfig()
    performance: PerformanceConfig = PerformanceConfig()
    security: SecurityConfig = SecurityConfig()
    ui: UIConfig = UIConfig()
    development: DevelopmentConfig = DevelopmentConfig()


class ConfigManager:
    """Manages configuration loading and access."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the config manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path or self._find_config_file()
        self._config: Optional[Config] = None
    
    def _find_config_file(self) -> str:
        """Find the configuration file in standard locations."""
        possible_paths = [
            "config/config.yaml",
            "config.yaml",
            "../config/config.yaml",
            os.path.expanduser("~/.document_agent/config.yaml")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(
            "Configuration file not found. Please create config/config.yaml "
            "or set the config path explicitly."
        )
    
    def load_config(self) -> Config:
        """
        Load configuration from file and environment variables.
        
        Returns:
            Config: Loaded configuration object
        """
        if self._config is not None:
            return self._config
        
        # Load from YAML file
        with open(self.config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        
        # Override with environment variables
        config_data = self._apply_env_overrides(config_data)
        
        # Validate and create config object
        self._config = Config(**config_data)
        
        # Create necessary directories
        self._create_directories()
        
        return self._config
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to config data."""
        env_mappings = {
            'OPENAI_API_KEY': ['llm', 'openai', 'api_key'],
            'GEMINI_API_KEY': ['llm', 'gemini', 'api_key'],
            'LLM_PROVIDER': ['llm', 'provider'],
            'VECTOR_STORE_TYPE': ['vector_store', 'type'],
            'DEBUG': ['development', 'debug'],
            'LOG_LEVEL': ['logging', 'level'],
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Navigate to the nested config location
                current = config_data
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Set the value (convert boolean strings)
                if env_value.lower() in ('true', 'false'):
                    current[config_path[-1]] = env_value.lower() == 'true'
                else:
                    current[config_path[-1]] = env_value
        
        return config_data
    
    def _create_directories(self):
        """Create necessary directories based on configuration."""
        if self._config is None:
            return
        
        directories = [
            self._config.storage.base_directory,
            self._config.storage.documents_directory,
            self._config.storage.processed_directory,
            self._config.storage.cache_directory,
            self._config.vector_store.chroma.persist_directory,
            os.path.dirname(self._config.logging.file),
        ]
        
        if self._config.arxiv.enabled and self._config.arxiv.download_papers:
            directories.append(self._config.arxiv.download_directory)
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @property
    def config(self) -> Config:
        """Get the loaded configuration."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def reload_config(self) -> Config:
        """Reload configuration from file."""
        self._config = None
        return self.load_config()
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration for the selected provider."""
        config = self.config
        provider = config.llm.provider
        
        if provider == "openai" and config.llm.openai:
            return {
                "provider": "openai",
                "api_key": config.llm.openai.api_key,
                "model": config.llm.openai.model,
                "temperature": config.llm.openai.temperature,
                "max_tokens": config.llm.openai.max_tokens,
            }
        elif provider == "gemini" and config.llm.gemini:
            return {
                "provider": "gemini",
                "api_key": config.llm.gemini.api_key,
                "model": config.llm.gemini.model,
                "temperature": config.llm.gemini.temperature,
                "max_tokens": config.llm.gemini.max_tokens,
            }
        else:
            raise ValueError(f"Invalid or incomplete LLM configuration for provider: {provider}")


# Global config manager instance
config_manager = ConfigManager()

def get_config() -> Config:
    """Get the global configuration instance."""
    return config_manager.config

def reload_config() -> Config:
    """Reload the global configuration."""
    return config_manager.reload_config()


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = get_config()
        print("Configuration loaded successfully!")
        print(f"LLM Provider: {config.llm.provider}")
        print(f"Vector Store: {config.vector_store.type}")
        print(f"ArXiv Integration: {'Enabled' if config.arxiv.enabled else 'Disabled'}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Please ensure config/config.yaml exists and is properly formatted.")
