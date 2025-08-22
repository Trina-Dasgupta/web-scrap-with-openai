#!/usr/bin/env python3
"""
Debug script to test configuration and dependencies
"""

import os
from pathlib import Path

def check_env_file():
    """Check .env file configuration"""
    print("🔍 Checking .env file...")
    
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found!")
        print("💡 Create it with: echo 'OPENAI_API_KEY=your_key_here' > .env")
        return False
    
    with open(env_file, 'r') as f:
        content = f.read()
    
    print(f"✅ .env file found")
    
    if 'OPENAI_API_KEY=' not in content:
        print("❌ OPENAI_API_KEY not found in .env")
        return False
    
    # Extract the key
    for line in content.split('\n'):
        if line.startswith('OPENAI_API_KEY='):
            key = line.split('=', 1)[1].strip()
            if key == 'your_openai_api_key_here' or len(key) < 10:
                print("❌ OpenAI API key looks like placeholder or too short")
                print(f"   Current value: {key}")
                return False
            else:
                print(f"✅ OpenAI API key found: {key[:10]}...{key[-4:]}")
                return True
    
    return False

def test_imports():
    """Test all required imports"""
    print("\n🔍 Testing imports...")
    
    imports_to_test = [
        ('streamlit', 'streamlit'),
        ('fastapi', 'fastapi'),
        ('langchain', 'langchain'),
        ('langchain_openai', 'langchain_openai'),
        ('langchain_community', 'langchain_community'),
        ('chromadb', 'chromadb'),
        ('requests', 'requests'),
        ('bs4', 'beautifulsoup4'),
        ('newspaper', 'newspaper3k'),
    ]
    
    failed_imports = []
    
    for import_name, package_name in imports_to_test:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError as e:
            print(f"❌ {package_name}: {e}")
            failed_imports.append(package_name)
    
    return len(failed_imports) == 0

def test_openai_connection():
    """Test OpenAI API connection"""
    print("\n🔍 Testing OpenAI connection...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ No OpenAI API key found in environment")
            return False
        
        print(f"🔑 Using API key: {api_key[:10]}...{api_key[-4:]}")
        
        # Test 1: Basic OpenAI client
        print("📡 Testing basic OpenAI client...")
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            # Simple completion test
            response = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt="Hello",
                max_tokens=5
            )
            print("✅ OpenAI client working!")
            
        except Exception as e:
            print(f"❌ OpenAI client failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            if "authentication" in str(e).lower() or "invalid" in str(e).lower():
                print("   💡 This looks like an API key issue")
                print("   💡 Check if your API key is correct and has credits")
            return False
        
        # Test 2: LangChain OpenAI embeddings
        print("🔗 Testing LangChain OpenAI embeddings...")
        try:
            from langchain_openai.embeddings import OpenAIEmbeddings
            
            embeddings = OpenAIEmbeddings(
                openai_api_key=api_key,
                model="text-embedding-ada-002"
            )
            
            # Test with a simple text
            test_text = ["Hello world"]
            result = embeddings.embed_documents(test_text)
            
            if result and len(result) > 0 and len(result[0]) > 0:
                print(f"✅ OpenAI embeddings working! Vector dimension: {len(result[0])}")
                return True
            else:
                print("❌ OpenAI embeddings returned empty result")
                return False
                
        except Exception as e:
            print(f"❌ LangChain OpenAI embeddings failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            if "rate limit" in str(e).lower():
                print("   💡 Rate limit exceeded - try again in a moment")
            elif "quota" in str(e).lower():
                print("   💡 API quota exceeded - check your OpenAI billing")
            elif "authentication" in str(e).lower():
                print("   💡 Authentication failed - check your API key")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI connection test crashed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

def check_openai_account():
    """Check OpenAI account status"""
    print("\n💳 Checking OpenAI account status...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ No API key found")
            return
        
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        # Try to get account information
        try:
            models = client.models.list()
            available_models = [m.id for m in models.data if 'gpt' in m.id or 'embedding' in m.id]
            print(f"✅ Account active. Available models: {len(available_models)}")
            print(f"   Key models: {', '.join(available_models[:5])}")
            
        except Exception as e:
            print(f"❌ Cannot access account: {e}")
            if "authentication" in str(e).lower():
                print("   💡 Invalid API key")
            elif "quota" in str(e).lower():
                print("   💡 Quota exceeded - add billing to your OpenAI account")
            elif "rate" in str(e).lower():
                print("   💡 Rate limited - try again later")
                
    except Exception as e:
        print(f"❌ Account check failed: {e}")

def main():
    """Run all diagnostic tests"""
    print("🔍 RAG Web Scraper - Diagnostic Tool")
    print("=" * 50)
    
    tests = [
        ("Environment Configuration", check_env_file),
        ("Package Imports", test_imports),
        ("OpenAI Connection", test_openai_connection),
        ("Web Scraping", test_web_scraping),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Additional OpenAI account check
    check_openai_account()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Your application should work correctly.")
        print("💡 If you're still having issues, restart the application:")
        print("   python run.py")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues above before running the application.")
        print("\n🔧 Common fixes:")
        print("   1. Check your OpenAI API key is correct")
        print("   2. Ensure you have credits in your OpenAI account")
        print("   3. Verify your API key has the right permissions")
        print("   4. Try generating a new API key from OpenAI dashboard")

def test_web_scraping():
    """Test web scraping functionality"""
    print("\n🔍 Testing web scraping...")
    
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # Test with a simple URL
        test_url = "https://httpbin.org/html"
        
        response = requests.get(test_url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        
        if len(text) > 10:
            print(f"✅ Web scraping working! Extracted {len(text)} characters")
            return True
        else:
            print("❌ Web scraping returned too little content")
            return False
            
    except Exception as e:
        print(f"❌ Web scraping failed: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("🔍 RAG Web Scraper - Diagnostic Tool")
    print("=" * 50)
    
    tests = [
        ("Environment Configuration", check_env_file),
        ("Package Imports", test_imports),
        ("OpenAI Connection", test_openai_connection),
        ("Web Scraping", test_web_scraping),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Your application should work correctly.")
        print("💡 If you're still having issues, restart the application:")
        print("   python run.py")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues above before running the application.")

if __name__ == "__main__":
    main()