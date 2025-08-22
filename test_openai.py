"""
Simple test script to debug OpenAI API connection issues.
Run this to test your OpenAI setup before using the main application.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openai_setup():
    """Test OpenAI API setup and connection"""
    print("🔍 Testing OpenAI API Setup")
    print("=" * 40)
    
    # Check if OpenAI package is installed
    try:
        import openai
        print("✅ OpenAI package is installed")
        print(f"   Version: {openai.__version__}")
    except ImportError:
        print("❌ OpenAI package not found")
        print("💡 Install with: pip install openai==0.28.1")
        return False
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found")
        print("💡 Create a .env file with: OPENAI_API_KEY=your_key_here")
        return False
    
    if api_key == "your_openai_api_key_here":
        print("❌ Default API key placeholder found")
        print("💡 Replace with your actual OpenAI API key")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test API connection
    print("\n🔌 Testing API Connection...")
    
    try:
        openai.api_key = api_key
        
        # Simple test request
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt="Hello, this is a test. Please respond with 'API connection successful'.",
            max_tokens=20,
            temperature=0.1
        )
        
        result = response.choices[0].text.strip()
        print(f"✅ API connection successful!")
        print(f"   Response: {result}")
        return True
        
    except openai.error.AuthenticationError as e:
        print(f"❌ Authentication failed: {e}")
        print("💡 Check if your API key is valid and active")
        return False
        
    except openai.error.RateLimitError as e:
        print(f"❌ Rate limit exceeded: {e}")
        print("💡 Your API quota may be exhausted or you're making too many requests")
        return False
        
    except openai.error.InvalidRequestError as e:
        print(f"❌ Invalid request: {e}")
        if "model" in str(e).lower():
            print("💡 Trying alternative model...")
            try:
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt="Hello, this is a test. Please respond with 'API connection successful'.",
                    max_tokens=20,
                    temperature=0.1
                )
                result = response.choices[0].text.strip()
                print(f"✅ Alternative model works!")
                print(f"   Response: {result}")
                return True
            except Exception as fallback_error:
                print(f"❌ Alternative model also failed: {fallback_error}")
                return False
        return False
        
    except openai.error.APIConnectionError as e:
        print(f"❌ Connection error: {e}")
        print("💡 Check your internet connection")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        return False

def test_simple_qa():
    """Test a simple question-answering scenario"""
    print("\n🤖 Testing Question Answering...")
    print("=" * 40)
    
    try:
        import openai
        
        api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = api_key
        
        context = "The capital of France is Paris. Paris is known for the Eiffel Tower and great cuisine."
        question = "What is the capital of France?"
        
        prompt = f"""Based on the following context, answer the question.

Context: {context}

Question: {question}

Answer:"""
        
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=100,
            temperature=0.1
        )
        
        answer = response.choices[0].text.strip()
        print(f"✅ Question answering test successful!")
        print(f"   Question: {question}")
        print(f"   Answer: {answer}")
        return True
        
    except Exception as e:
        print(f"❌ Question answering test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 OpenAI API Test Suite")
    print("=" * 50)
    
    # Test basic setup
    if not test_openai_setup():
        print("\n❌ Basic setup failed. Please fix the issues above before proceeding.")
        return
    
    # Test question answering
    if not test_simple_qa():
        print("\n❌ Question answering test failed.")
        return
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! Your OpenAI setup is working correctly.")
    print("💡 You can now use the RAG Web Scraper application.")

if __name__ == "__main__":
    main()