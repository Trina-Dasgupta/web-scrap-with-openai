#!/usr/bin/env python3
"""
Bulletproof installation script that handles dependency conflicts
"""

import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def install_packages():
    """Install packages in the correct order to avoid conflicts"""
    
    # Step 1: Remove all LangChain packages to start clean
    print("🧹 Cleaning up existing LangChain packages...")
    cleanup_commands = [
        "pip uninstall langchain langchain-community langchain-openai langchain-core -y",
        "pip uninstall openai tiktoken -y"
    ]
    
    for cmd in cleanup_commands:
        run_command(cmd, "Cleanup")
    
    # Step 2: Install core dependencies first
    print("\n📦 Installing core dependencies...")
    core_packages = [
        "streamlit==1.28.1",
        "fastapi==0.104.1", 
        "uvicorn==0.24.0",
        "requests==2.31.0",
        "beautifulsoup4==4.12.2",
        "python-dotenv==1.0.0",
        "lxml==4.9.3",
        "newspaper3k==0.2.8",
        "python-multipart==0.0.6",
        "pydantic==2.5.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0"
    ]
    
    for package in core_packages:
        if not run_command(f"pip install {package}", f"Installing {package.split('==')[0]}"):
            return False
    
    # Step 3: Install OpenAI and tiktoken
    print("\n🤖 Installing OpenAI dependencies...")
    openai_packages = [
        "tiktoken==0.7.0",
        "openai==1.35.13"
    ]
    
    for package in openai_packages:
        if not run_command(f"pip install {package}", f"Installing {package.split('==')[0]}"):
            return False
    
    # Step 4: Install LangChain in dependency order
    print("\n🔗 Installing LangChain ecosystem...")
    langchain_packages = [
        "langchain-core==0.2.24",
        "langchain-community==0.2.10", 
        "langchain-openai==0.1.19",
        "langchain==0.2.11"
    ]
    
    for package in langchain_packages:
        if not run_command(f"pip install {package}", f"Installing {package.split('==')[0]}"):
            return False
    
    # Step 5: Install ChromaDB
    print("\n🗄️ Installing ChromaDB...")
    if not run_command("pip install chromadb==0.4.24", "Installing ChromaDB"):
        return False
    
    return True

def test_installation():
    """Test that everything works"""
    print("\n🧪 Testing installation...")
    
    test_imports = [
        "streamlit",
        "fastapi",
        "openai", 
        "langchain",
        "langchain_core",
        "langchain_community",
        "langchain_openai",
        "chromadb"
    ]
    
    failed = []
    for package in test_imports:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed.append(package)
    
    return len(failed) == 0

def main():
    """Main installation process"""
    print("🔧 RAG Web Scraper - Bulletproof Installation")
    print("=" * 60)
    
    if not install_packages():
        print("\n❌ Installation failed!")
        return False
    
    if not test_installation():
        print("\n❌ Installation test failed!")
        return False
    
    print("\n🎉 Installation completed successfully!")
    print("\n📝 Next steps:")
    print("1. Make sure your .env file has: OPENAI_API_KEY=your_key_here")
    print("2. Run: python debug.py")
    print("3. Run: python run.py")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Installation cancelled")
    except Exception as e:
        print(f"\n❌ Installation crashed: {e}")