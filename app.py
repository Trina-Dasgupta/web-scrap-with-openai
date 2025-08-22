import streamlit as st
import requests
import json
from typing import Dict, Any
import time
import uuid

# Configure page
st.set_page_config(
    page_title="RAG Web Scraper",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI (Fixed version - no duplicates)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #4CAF50, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .info-card {
        background: #ffffff !important;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .info-card small {
        color: #495057 !important;
        line-height: 1.6;
        display: block;
        white-space: pre-wrap;
        word-wrap: break-word;
        font-size: 0.9rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        animation: slideIn 0.3s ease-in;
    }
    
    .user-message {
        background: #e3f2fd !important;
        border-left: 4px solid #2196F3;
        color: #1565c0 !important;
    }
    
    .bot-message {
        background: #f1f8e9 !important;
        border-left: 4px solid #4CAF50;
        color: #2e7d32 !important;
    }
    
    .metric-card {
        background: white !important;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        border: 1px solid #e9ecef;
        color: #333 !important;
    }
    
    .metric-card strong {
        color: #1a1a1a !important;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "scraped_url" not in st.session_state:
    st.session_state.scraped_url = None
if "scraped_content_preview" not in st.session_state:
    st.session_state.scraped_content_preview = None
if "word_count" not in st.session_state:
    st.session_state.word_count = 0
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

def make_api_request(endpoint: str, method: str = "GET", data: Dict[Any, Any] = None) -> Dict[Any, Any]:
    """Make API request to FastAPI backend"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "POST":
            response = requests.post(url, json=data, timeout=30)
        elif method == "DELETE":
            response = requests.delete(url, timeout=30)
        else:
            response = requests.get(url, timeout=30)
        
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to API server. Please ensure the FastAPI server is running on http://localhost:8000"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. The website might be taking too long to load."}
    except requests.exceptions.RequestException as e:
        try:
            error_detail = response.json().get("detail", str(e))
        except:
            error_detail = str(e)
        return {"success": False, "error": f"API Error: {error_detail}"}

def scrape_website(url: str):
    """Scrape website content"""
    with st.spinner("🔍 Scraping website content... This may take a moment."):
        result = make_api_request("/scrape", "POST", {"url": url})
        
        if result["success"]:
            st.session_state.scraped_url = url
            st.session_state.scraped_content_preview = result["data"]["content_preview"]
            st.session_state.word_count = result["data"]["word_count"]
            st.session_state.messages = []  # Clear previous chat
            return True, result["data"]["message"]
        else:
            return False, result["error"]

def ask_question(question: str):
    """Ask question about scraped content"""
    print(f"DEBUG: Sending question: {question}")
    
    payload = {
        "question": question,
        "session_id": "default" 
    }
    print(f"DEBUG: Payload: {payload}")
    
    result = make_api_request("/ask", "POST", payload)
    print(f"DEBUG: API result: {result}")
    
    if result["success"]:
        return True, result["data"]["answer"], result["data"]["sources"]
    else:
        return False, result["error"], []

def clear_session():
    """Clear current session"""
    result = make_api_request(f"/clear/{st.session_state.session_id}", "DELETE")
    st.session_state.scraped_url = None
    st.session_state.scraped_content_preview = None
    st.session_state.word_count = 0
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())

# Main UI
st.markdown('<h1 class="main-header">🔍 RAG Web Scraper</h1>', unsafe_allow_html=True)
st.markdown("**Extract knowledge from any website and ask questions about its content using AI**")

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Session Info")
    
    if st.session_state.scraped_url:
        st.markdown(f'<div class="metric-card"><strong>📄 Scraped URL</strong><br><small>{st.session_state.scraped_url}</small></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><strong>📝 Word Count</strong><br>{st.session_state.word_count:,} words</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><strong>💬 Questions Asked</strong><br>{len([m for m in st.session_state.messages if m["role"] == "user"])}</div>', unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Session", type="secondary", use_container_width=True):
            clear_session()
            st.rerun()
    else:
        st.markdown('<div class="metric-card">No content scraped yet</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Paste any website URL to start
    - Works with articles, blogs, documentation
    - Ask specific questions about the content
    - Use natural language queries
    """)
    
    st.markdown("### 🔧 Requirements")
    st.markdown("""
    - FastAPI server running on port 8000
    - OpenAI API key configured
    - Internet connection for scraping
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # URL Input Section
    st.markdown("### 🌐 Website URL")
    url_input = st.text_input(
        "Enter the URL you want to scrape:",
        placeholder="https://example.com/article",
        help="Enter any website URL (articles, blogs, documentation, etc.)"
    )
    
    col_scrape, col_example = st.columns([1, 1])
    
    with col_scrape:
        if st.button("🚀 Scrape Website", type="primary", use_container_width=True):
            if url_input.strip():
                success, message = scrape_website(url_input.strip())
                if success:
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
            else:
                st.warning("⚠️ Please enter a URL")
    
    with col_example:
        example_urls = [
            "https://httpbin.org/html",
            "https://simple.wikipedia.org/wiki/Computer",
            "https://docs.python.org/3/tutorial/",
        ]
        
        selected_example = st.selectbox(
            "Or try an example:",
            ["Select an example..."] + example_urls,
            key="example_selector"
        )
        
        if selected_example and selected_example != "Select an example...":
            if st.button("Use This Example", use_container_width=True):
                success, message = scrape_website(selected_example)
                if success:
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")

with col2:
    # Content Preview
    if st.session_state.scraped_content_preview:
        st.markdown("### 📄 Content Preview")
        st.markdown(f'<div class="info-card"><small>{st.session_state.scraped_content_preview}</small></div>', unsafe_allow_html=True)

# Chat Interface
if st.session_state.scraped_url:
    st.markdown("---")
    st.markdown("### 💬 Ask Questions About The Content")
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message"><strong>AI:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            if "sources" in message:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")
    
    # Question input
    question_input = st.text_input(
        "Ask a question about the content:",
        placeholder="What is the main topic of this article?",
        key="question_input"
    )
    
    col_ask, col_examples = st.columns([1, 2])
    
    with col_ask:
        if st.button("🤔 Ask Question", type="primary", use_container_width=True):
            if question_input.strip():
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": question_input
                })
                
                with st.spinner("🤖 Thinking... Please wait."):
                    success, answer, sources = ask_question(question_input)
                
                if success:
                    # Add AI response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"❌ {answer}")
                
                st.rerun()
            else:
                st.warning("⚠️ Please enter a question")
    
    with col_examples:
        st.markdown("**Example questions:**")
        example_questions = [
            "What is the main topic?",
            "Summarize the key points",
            "What are the conclusions?",
            "Who are the main people mentioned?"
        ]
        
        for i, eq in enumerate(example_questions):
            if st.button(f"💡 {eq}", key=f"example_q_{i}", use_container_width=True):
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": eq
                })
                
                with st.spinner("🤖 Thinking... Please wait."):
                    success, answer, sources = ask_question(eq)
                
                if success:
                    # Add AI response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"❌ {answer}")
                
                st.rerun()

else:
    st.markdown("---")
    st.markdown('<div class="info-card">👆 Please scrape a website first to start asking questions!</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("**Powered by FastAPI, Streamlit & OpenAI** | Built with ❤️")

# Health check button
if st.button("🔍 Check API Health", help="Test connection to FastAPI backend"):
    health_result = make_api_request("/health")
    if health_result["success"]:
        st.success("✅ API Server is running")
        st.json(health_result["data"])
    else:
        st.error(f"❌ API Server issue: {health_result['error']}")