#!/usr/bin/env python3
"""
Enhanced setup script for the RAG Web Scraper application.
Installs all enhanced dependencies and sets up the system.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout and len(result.stdout.strip()) > 0:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        if e.stdout:
            print(f"Output: {e.stdout.strip()}")
        if e.stderr:
            print(f"Error: {e.stderr.strip()}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def setup_virtual_environment():
    """Create and activate virtual environment"""
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print("📁 Virtual environment already exists")
        return True
    
    if not run_command("python -m venv .venv", "Creating virtual environment"):
        return False
    
    return True

def install_enhanced_dependencies():
    """Install all enhanced dependencies"""
    print("\n📦 Installing Enhanced Dependencies...")
    
    # First, upgrade pip
    if not run_command("python -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install enhanced requirements
    if not run_command("pip install -r requirements_enhanced.txt", "Installing enhanced packages"):
        return False
    
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    
    nltk_script = """
import nltk
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    print("✅ NLTK data downloaded successfully")
except Exception as e:
    print(f"❌ Error downloading NLTK data: {e}")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", nltk_script], 
                              capture_output=True, text=True, timeout=60)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        return True
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")
        return False

def test_enhanced_features():
    """Test if enhanced features are working"""
    print("\n🧪 Testing Enhanced Features...")
    
    test_script = """
import sys
sys.path.append('.')

# Test sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Sentence Transformers working")
except Exception as e:
    print(f"❌ Sentence Transformers error: {e}")

# Test ChromaDB
try:
    import chromadb
    client = chromadb.Client()
    print("✅ ChromaDB working")
except Exception as e:
    print(f"❌ ChromaDB error: {e}")

# Test document processing
try:
    import PyPDF2
    from docx import Document
    import openpyxl
    print("✅ Document processors working")
except Exception as e:
    print(f"❌ Document processors error: {e}")

# Test NLTK
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    sent_tokenize("This is a test. Another sentence.")
    print("✅ NLTK working")
except Exception as e:
    print(f"❌ NLTK error: {e}")

# Test evaluation
try:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    print("✅ Evaluation metrics working")
except Exception as e:
    print(f"❌ Evaluation metrics error: {e}")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_script], 
                              capture_output=True, text=True, timeout=120)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        return True
    except Exception as e:
        print(f"❌ Feature testing failed: {e}")
        return False

def check_env_file():
    """Check and create .env file if needed"""
    env_file = Path('.env')
    
    if env_file.exists():
        print("✅ .env file exists")
        return True
    
    print("📝 Creating .env file template...")
    env_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Additional API configurations
# ANTHROPIC_API_KEY=your_anthropic_key_here
# COHERE_API_KEY=your_cohere_key_here
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ .env file created")
        print("⚠️  Please add your actual OpenAI API key to the .env file")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def create_sample_documents():
    """Create sample documents for testing"""
    samples_dir = Path("sample_documents")
    samples_dir.mkdir(exist_ok=True)
    
    # Create sample text file
    sample_txt = samples_dir / "sample_article.txt"
    if not sample_txt.exists():
        content = """Sample Article: The Future of AI

Artificial Intelligence (AI) is rapidly transforming various industries and aspects of human life. 
From healthcare to transportation, AI technologies are being implemented to solve complex problems 
and improve efficiency.

Key developments in AI include:
- Machine Learning algorithms that can learn from data
- Natural Language Processing for human-computer interaction
- Computer Vision for image and video analysis
- Robotics for automation and assistance

The future of AI holds great promise, but also presents challenges that need to be addressed, 
including ethical considerations, job displacement, and the need for proper regulation.

As we move forward, it's crucial to develop AI systems that are beneficial, safe, and aligned 
with human values.
"""
        try:
            with open(sample_txt, 'w') as f:
                f.write(content)
            print(f"✅ Created sample document: {sample_txt}")
        except Exception as e:
            print(f"❌ Failed to create sample document: {e}")

def main():
    """Main setup function"""
    print("🚀 Enhanced RAG Web Scraper Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Setup virtual environment
    if not setup_virtual_environment():
        print("❌ Failed to setup virtual environment")
        return False
    
    # Check if we're in virtual environment (Windows/Linux compatible)
    venv_active = (
        hasattr(sys, 'real_prefix') or 
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
        'VIRTUAL_ENV' in os.environ
    )
    
    if not venv_active:
        print("⚠️  Virtual environment not activated")
        print("💡 Please activate it manually:")
        if os.name == 'nt':  # Windows
            print("   .venv\\Scripts\\activate")
        else:  # Linux/Mac
            print("   source .venv/bin/activate")
        print("   Then run this script again")
        return False
    
    print("✅ Virtual environment is active")
    
    # Install dependencies
    if not install_enhanced_dependencies():
        print("❌ Failed to install dependencies")
        return False
    
    # Download NLTK data
    if not download_nltk_data():
        print("⚠️  NLTK data download failed, but continuing...")
    
    # Test enhanced features
    print("\n🧪 Testing enhanced features (this may take a moment)...")
    if not test_enhanced_features():
        print("⚠️  Some features may not work correctly")
    
    # Check/create .env file
    if not check_env_file():
        print("❌ Failed to setup .env file")
        return False
    
    # Create sample documents
    create_sample_documents()
    
    print("\n" + "=" * 50)
    print("🎉 Enhanced RAG Web Scraper Setup Complete!")
    print("\n📋 Next Steps:")
    print("1. Add your OpenAI API key to the .env file")
    print("2. Start the enhanced backend: python main.py")
    print("3. In another terminal, start the frontend: streamlit run app.py")
    print("\n🆕 New Features Available:")
    print("- Vector embeddings for semantic search")
    print("- Multi-format document support (PDF, DOCX, XLSX)")
    print("- Conversation memory")
    print("- Quality evaluation metrics")
    print("- Enhanced analytics dashboard")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ Setup failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏸️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)