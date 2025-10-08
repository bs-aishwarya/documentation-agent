#!/usr/bin/env python3
"""
Development setup script for the Document Q&A Agent.

This script helps set up the development environment and verify the installation.
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def create_directories():
    """Create necessary directories."""
    directories = [
        "data/chroma",
        "data/faiss", 
        "data/cache",
        "logs",
        "downloads/arxiv",
        "temp",
        "uploads"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    
    return True

def download_spacy_model():
    """Download required spaCy model."""
    print("🔄 Downloading spaCy model...")
    
    try:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                      check=True, capture_output=True)
        print("✅ spaCy model downloaded")
    except subprocess.CalledProcessError:
        print("⚠️ spaCy model download failed - some features may not work")

def verify_installation():
    """Verify the installation by importing key modules."""
    print("🔍 Verifying installation...")
    
    try:
        # Add src to path
        sys.path.append("src")
        
        from utils.config import get_config
        config = get_config()
        print("✅ Configuration system working")
        
        from ingestion.document_processor import DocumentProcessor
        print("✅ Document processor available")
        
        from extraction.vector_store import VectorStoreManager  
        print("✅ Vector store available")
        
        from query.query_engine import DocumentQueryEngine
        print("✅ Query engine available")
        
        from agents.arxiv_agent import ArXivAgent
        print("✅ ArXiv agent available")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def create_env_file():
    """Create .env file from template if it doesn't exist."""
    if not Path(".env").exists():
        if Path(".env.template").exists():
            print("📝 Creating .env file from template...")
            with open(".env.template", "r") as template:
                content = template.read()
            
            with open(".env", "w") as env_file:
                env_file.write(content)
            
            print("✅ .env file created - please update with your API keys")
        else:
            print("⚠️ .env.template not found")
    else:
        print("✅ .env file already exists")

def main():
    """Main setup function."""
    print("🚀 Setting up Document Q&A Agent development environment...\n")
    
    # Check prerequisites
    check_python_version()
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        sys.exit(1)
    
    # Download models
    print("\n🤖 Setting up AI models...")
    download_spacy_model()
    
    # Create environment file
    print("\n⚙️ Setting up configuration...")
    create_env_file()
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    if verify_installation():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Update .env file with your API keys")
        print("2. Run: python main.py web")
        print("3. Open http://localhost:8501 in your browser")
    else:
        print("\n❌ Setup completed with errors")
        print("Please check the error messages above")
        sys.exit(1)

if __name__ == "__main__":
    main()
