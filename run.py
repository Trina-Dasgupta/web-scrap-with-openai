#!/usr/bin/env python3
"""
Quick start script for the RAG Web Scraper application.
This script helps you set up and run both the FastAPI backend and Streamlit frontend.
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path
import threading

def check_dependencies():
    """Check if required packages are installed"""
    # Map pip package names to import names
    required_packages = {
        'streamlit': 'streamlit',
        'fastapi': 'fastapi', 
        'uvicorn': 'uvicorn',
        'requests': 'requests',
        'beautifulsoup4': 'bs4',  # beautifulsoup4 imports as bs4
        'langchain': 'langchain',
        'langchain-openai': 'langchain_openai',
        'chromadb': 'chromadb'
    }
    
    missing_packages = []
    for pip_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {pip_name}")
        except ImportError:
            missing_packages.append(pip_name)
            print(f"❌ {pip_name}")
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def check_env_file():
    """Check if .env file exists and has OpenAI API key"""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("❌ .env file not found!")
        print("💡 Create a .env file with your OpenAI API key:")
        print("   OPENAI_API_KEY=your_api_key_here")
        return False
    
    # Read .env file
    with open(env_file, 'r') as f:
        content = f.read()
    
    if 'OPENAI_API_KEY=' not in content or 'your_openai_api_key_here' in content:
        print("❌ OpenAI API key not configured in .env file!")
        print("💡 Add your OpenAI API key to the .env file:")
        print("   OPENAI_API_KEY=your_actual_api_key")
        return False
    
    print("✅ Environment configuration looks good!")
    return True

def run_fastapi_server():
    """Run the FastAPI server"""
    print("🚀 Starting FastAPI server...")
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ FastAPI server failed to start: {e}")
    except KeyboardInterrupt:
        print("\n⏹️ FastAPI server stopped")

def run_streamlit_app():
    """Run the Streamlit application"""
    print("🌐 Starting Streamlit app...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Streamlit app failed to start: {e}")
    except KeyboardInterrupt:
        print("\n⏹️ Streamlit app stopped")

def check_port_availability(port):
    """Check if a port is available"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def main():
    """Main function to run the application"""
    print("🔍 RAG Web Scraper - Quick Start")
    print("=" * 40)
    
    # Check current directory for required files
    required_files = ['main.py', 'app.py', 'requirements.txt']
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("💡 Make sure you're in the correct directory with all project files")
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check environment configuration
    if not check_env_file():
        return
    
    # Check if ports are available
    if not check_port_availability(8000):
        print("❌ Port 8000 is already in use (FastAPI)")
        print("💡 Stop any running FastAPI servers or change the port")
        return
    
    if not check_port_availability(8501):
        print("❌ Port 8501 is already in use (Streamlit)")
        print("💡 Stop any running Streamlit apps or change the port")
        return
    
    print("✅ All checks passed! Starting the application...")
    print("\n" + "=" * 40)
    
    # Start FastAPI server in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi_server, daemon=True)
    fastapi_thread.start()
    
    # Wait a moment for FastAPI to start
    print("⏳ Waiting for FastAPI server to start...")
    time.sleep(3)
    
    # Check if FastAPI is running
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ FastAPI server is running on http://localhost:8000")
        else:
            print("❌ FastAPI server is not responding properly")
            return
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to FastAPI server")
        return
    
    # Open browser
    print("🌐 Opening web browser...")
    webbrowser.open("http://localhost:8501")
    
    # Start Streamlit (this will block)
    try:
        print("🚀 Starting Streamlit frontend...")
        run_streamlit_app()
    except KeyboardInterrupt:
        print("\n⏹️ Application stopped by user")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("💡 Please check the setup instructions in README.md")