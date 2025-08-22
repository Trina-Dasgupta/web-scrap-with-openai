import streamlit as st
import requests
import json
from typing import Dict, Any
import time
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Enhanced RAG Web Scraper",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #4CAF50, #2196F3, #9C27B0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .feature-badge {
        display: inline-block;
        background: linear-gradient(45deg, #4CAF50, #2196F3);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-weight: bold;
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
    
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .conversation-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .upload-section {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background: #f9f9f9;
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
if "document_type" not in st.session_state:
    st.session_state.document_type = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "use_conversation_memory" not in st.session_state:
    st.session_state.use_conversation_memory = True

def make_api_request(endpoint: str, method: str = "GET", data: Dict[Any, Any] = None, files=None) -> Dict[Any, Any]:
    """Enhanced API request handler"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "POST":
            if files:
                response = requests.post(url, files=files, timeout=60)
            else:
                response = requests.post(url, json=data, timeout=60)
        elif method == "DELETE":
            response = requests.delete(url, timeout=30)
        else:
            response = requests.get(url, timeout=30)
        
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to API server. Please ensure the enhanced FastAPI server is running on http://localhost:8000"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. Large files may take longer to process."}
    except requests.exceptions.RequestException as e:
        try:
            error_detail = response.json().get("detail", str(e))
        except:
            error_detail = str(e)
        return {"success": False, "error": f"API Error: {error_detail}"}

def scrape_website(url: str):
    """Scrape website content with enhanced processing"""
    with st.spinner("🔍 Scraping website with enhanced vector processing... This may take a moment."):
        result = make_api_request("/scrape", "POST", {"url": url})
        
        if result["success"]:
            st.session_state.scraped_url = url
            st.session_state.scraped_content_preview = result["data"]["content_preview"]
            st.session_state.word_count = result["data"]["word_count"]
            st.session_state.document_type = result["data"].get("document_type", "webpage")
            st.session_state.messages = []  # Clear previous chat
            st.session_state.conversation_history = []
            return True, result["data"]["message"]
        else:
            return False, result["error"]

def upload_document(uploaded_file):
    """Upload and process document"""
    with st.spinner(f"📄 Processing {uploaded_file.name} with vector embeddings..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        result = make_api_request("/upload", "POST", files=files)
        
        if result["success"]:
            st.session_state.scraped_url = uploaded_file.name
            st.session_state.scraped_content_preview = result["data"]["content_preview"]
            st.session_state.word_count = result["data"]["word_count"]
            st.session_state.document_type = result["data"].get("document_type", "unknown")
            st.session_state.messages = []  # Clear previous chat
            st.session_state.conversation_history = []
            return True, result["data"]["message"]
        else:
            return False, result["error"]

def ask_question(question: str):
    """Ask question with enhanced features"""
    result = make_api_request("/ask", "POST", {
        "question": question,
        "session_id": "default",
        "use_conversation_history": st.session_state.use_conversation_memory
    })
    
    if result["success"]:
        data = result["data"]
        # Update conversation history
        if data.get("conversation_context"):
            st.session_state.conversation_history = data["conversation_context"]
        
        return True, data["answer"], data["sources"], data.get("relevance_scores", []), data.get("evaluation_metrics", {})
    else:
        return False, result["error"], [], [], {}

def clear_session():
    """Clear current session"""
    result = make_api_request("/clear/default", "DELETE")
    st.session_state.scraped_url = None
    st.session_state.scraped_content_preview = None
    st.session_state.word_count = 0
    st.session_state.document_type = None
    st.session_state.messages = []
    st.session_state.conversation_history = []

def get_conversation_history():
    """Get full conversation history"""
    result = make_api_request("/conversation/default")
    if result["success"]:
        return result["data"]["conversation"]
    return []

# Main UI
st.markdown('<h1 class="main-header">🔍 Enhanced RAG Web Scraper</h1>', unsafe_allow_html=True)

# Feature badges
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <span class="feature-badge">🧠 Vector Embeddings</span>
    <span class="feature-badge">📚 Multi-Format Support</span>
    <span class="feature-badge">💭 Conversation Memory</span>
    <span class="feature-badge">📊 Quality Metrics</span>
    <span class="feature-badge">🎯 Semantic Search</span>
</div>
""", unsafe_allow_html=True)

st.markdown("**Next-generation RAG with vector embeddings, conversation memory, and advanced document processing**")

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Session Information")
    
    if st.session_state.scraped_url:
        st.markdown(f'<div class="metric-card"><strong>📄 Source</strong><br><small>{st.session_state.scraped_url}</small></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><strong>📝 Words</strong><br>{st.session_state.word_count:,}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><strong>📋 Type</strong><br>{st.session_state.document_type}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><strong>💬 Q&A Exchanges</strong><br>{len([m for m in st.session_state.messages if m["role"] == "user"])}</div>', unsafe_allow_html=True)
        
        # Conversation memory toggle
        st.session_state.use_conversation_memory = st.toggle(
            "🧠 Conversation Memory",
            value=st.session_state.use_conversation_memory,
            help="Remember previous questions for context"
        )
        
        if st.button("🗑️ Clear Session", type="secondary", use_container_width=True):
            clear_session()
            st.rerun()
    else:
        st.markdown('<div class="metric-card">No content loaded yet</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🆕 Enhanced Features")
    st.markdown("""
    - **Vector Embeddings**: Semantic search using sentence transformers
    - **Multi-Format**: PDF, DOCX, XLSX, TXT, MD support
    - **Conversation Memory**: Contextual follow-up questions
    - **Quality Metrics**: Response evaluation and relevance scores
    - **Semantic Chunking**: Intelligent text segmentation
    """)
    
    st.markdown("### 🔧 System Status")
    # Health check
    health_result = make_api_request("/health")
    if health_result["success"]:
        health_data = health_result["data"]
        st.success("✅ Enhanced API Online")
        st.markdown(f"**Search**: {health_data.get('search_method', 'Unknown')}")
        st.markdown(f"**Embeddings**: {health_data.get('embeddings_status', 'Unknown')}")
        st.markdown(f"**Version**: {health_data.get('version', 'Unknown')}")
    else:
        st.error("❌ API Offline")

# Main content area
tab1, tab2, tab3 = st.tabs(["📄 Document Input", "💬 Chat Interface", "📊 Analytics"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🌐 Web Scraping")
        url_input = st.text_input(
            "Enter website URL:",
            placeholder="https://example.com/article",
            help="Scrape content from any website with enhanced processing"
        )
        
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
        
        # Example URLs
        st.markdown("**Quick Examples:**")
        example_urls = [
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://docs.python.org/3/tutorial/",
            "https://arxiv.org/abs/1706.03762",  # Attention paper
        ]
        
        for i, url in enumerate(example_urls):
            if st.button(f"📖 {url.split('/')[-1][:30]}...", key=f"example_{i}"):
                success, message = scrape_website(url)
                if success:
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
    
    with col2:
        st.markdown("### 📤 Document Upload")
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload your document",
            type=['pdf', 'docx', 'xlsx', 'txt', 'md'],
            help="Supports PDF, Word, Excel, Text, and Markdown files"
        )
        
        if uploaded_file is not None:
            st.write(f"📄 **File**: {uploaded_file.name}")
            st.write(f"📊 **Size**: {uploaded_file.size:,} bytes")
            st.write(f"🔖 **Type**: {uploaded_file.type}")
            
            if st.button("🔄 Process Document", type="primary", use_container_width=True):
                success, message = upload_document(uploaded_file)
                if success:
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Content Preview
    if st.session_state.scraped_content_preview:
        st.markdown("### 📄 Content Preview")
        st.markdown(f'<div class="info-card"><small>{st.session_state.scraped_content_preview}</small></div>', unsafe_allow_html=True)

with tab2:
    if st.session_state.scraped_url:
        st.markdown("### 💬 Enhanced Q&A Chat")
        
        # Display chat messages with enhanced info
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot-message"><strong>AI:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                
                # Show sources
                if "sources" in message:
                    with st.expander("📚 Sources & Metrics"):
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown("**Sources:**")
                            for source in message["sources"]:
                                st.markdown(f"- {source}")
                        
                        with col2:
                            if "relevance_scores" in message and message["relevance_scores"]:
                                st.markdown("**Relevance Scores:**")
                                for j, score in enumerate(message["relevance_scores"]):
                                    st.markdown(f"Chunk {j+1}: {score:.3f}")
                            
                            if "metrics" in message and message["metrics"]:
                                st.markdown("**Quality Metrics:**")
                                metrics = message["metrics"]
                                for key, value in metrics.items():
                                    if isinstance(value, float):
                                        st.markdown(f"{key}: {value:.3f}")
                                    else:
                                        st.markdown(f"{key}: {value}")
        
        # Question input with enhanced options
        col1, col2 = st.columns([3, 1])
        
        with col1:
            question_input = st.text_input(
                "Ask a question about the content:",
                placeholder="What are the key insights from this document?",
                key="question_input"
            )
        
        with col2:
            if st.button("🤔 Ask", type="primary", use_container_width=True):
                if question_input.strip():
                    # Add user message
                    st.session_state.messages.append({
                        "role": "user",
                        "content": question_input
                    })
                    
                    with st.spinner("🧠 Processing with vector search and conversation context..."):
                        success, answer, sources, relevance_scores, metrics = ask_question(question_input)
                    
                    if success:
                        # Add AI response with enhanced data
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "relevance_scores": relevance_scores,
                            "metrics": metrics
                        })
                    else:
                        st.error(f"❌ {answer}")
                    
                    st.rerun()
                else:
                    st.warning("⚠️ Please enter a question")
        
        # Enhanced example questions
        st.markdown("### 💡 Intelligent Question Suggestions")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        question_categories = {
            "📋 Summary": [
                "Summarize the main points",
                "What are the key takeaways?",
                "Give me an executive summary"
            ],
            "🔍 Analysis": [
                "What are the implications?",
                "How does this relate to current trends?",
                "What are the strengths and weaknesses?"
            ],
            "❓ Details": [
                "Who are the main people mentioned?",
                "What specific data is provided?",
                "What methodology was used?"
            ]
        }
        
        for i, (category, questions) in enumerate(question_categories.items()):
            with [col1, col2, col3][i]:
                st.markdown(f"**{category}**")
                for j, eq in enumerate(questions):
                    if st.button(eq, key=f"cat_{i}_q_{j}", use_container_width=True):
                        # Add user message
                        st.session_state.messages.append({
                            "role": "user",
                            "content": eq
                        })
                        
                        with st.spinner("🧠 Processing..."):
                            success, answer, sources, relevance_scores, metrics = ask_question(eq)
                        
                        if success:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": sources,
                                "relevance_scores": relevance_scores,
                                "metrics": metrics
                            })
                        else:
                            st.error(f"❌ {answer}")
                        
                        st.rerun()
    else:
        st.markdown("---")
        st.markdown('<div class="info-card">👆 Please load content first using the Document Input tab!</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("### 📊 Analytics & Insights")
    
    if st.session_state.messages:
        # Conversation analytics
        user_messages = [m for m in st.session_state.messages if m["role"] == "user"]
        ai_messages = [m for m in st.session_state.messages if m["role"] == "assistant"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="score-card"><h3>📝</h3><p>Total Questions</p><h2>{}</h2></div>'.format(len(user_messages)), unsafe_allow_html=True)
        
        with col2:
            avg_q_length = sum(len(m["content"].split()) for m in user_messages) / len(user_messages) if user_messages else 0
            st.markdown('<div class="score-card"><h3>📏</h3><p>Avg Question Length</p><h2>{:.1f}</h2></div>'.format(avg_q_length), unsafe_allow_html=True)
        
        with col3:
            avg_a_length = sum(len(m["content"].split()) for m in ai_messages) / len(ai_messages) if ai_messages else 0
            st.markdown('<div class="score-card"><h3>💬</h3><p>Avg Answer Length</p><h2>{:.1f}</h2></div>'.format(avg_a_length), unsafe_allow_html=True)
        
        with col4:
            # Calculate average relevance score
            all_scores = []
            for m in ai_messages:
                if "relevance_scores" in m and m["relevance_scores"]:
                    all_scores.extend(m["relevance_scores"])
            avg_relevance = sum(all_scores) / len(all_scores) if all_scores else 0
            st.markdown('<div class="score-card"><h3>🎯</h3><p>Avg Relevance</p><h2>{:.3f}</h2></div>'.format(avg_relevance), unsafe_allow_html=True)
        
        # Relevance scores over time
        if any("relevance_scores" in m for m in ai_messages):
            st.markdown("### 📈 Relevance Scores Over Time")
            
            scores_data = []
            for i, message in enumerate(ai_messages):
                if "relevance_scores" in message and message["relevance_scores"]:
                    for j, score in enumerate(message["relevance_scores"]):
                        scores_data.append({
                            "Question": i + 1,
                            "Chunk": j + 1,
                            "Relevance Score": score
                        })
            
            if scores_data:
                df = pd.DataFrame(scores_data)
                fig = px.line(df, x="Question", y="Relevance Score", color="Chunk",
                             title="Document Chunk Relevance Scores")
                st.plotly_chart(fig, use_container_width=True)
        
        # Quality metrics
        st.markdown("### 📊 Response Quality Metrics")
        
        metrics_data = []
        for i, message in enumerate(ai_messages):
            if "metrics" in message and message["metrics"]:
                row = {"Question": i + 1}
                row.update(message["metrics"])
                metrics_data.append(row)
        
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            
            # Display metrics table
            st.dataframe(df_metrics, use_container_width=True)
            
            # Metrics visualization
            if len(df_metrics) > 1:
                numeric_cols = df_metrics.select_dtypes(include=[float, int]).columns
                if len(numeric_cols) > 0:
                    fig = px.bar(df_metrics, x="Question", y=list(numeric_cols),
                               title="Quality Metrics by Question")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Conversation history
        if st.session_state.conversation_history:
            st.markdown("### 💭 Conversation Context")
            for i, exchange in enumerate(st.session_state.conversation_history):
                with st.expander(f"Exchange {i+1}: {exchange['question'][:50]}..."):
                    st.markdown(f"**Question:** {exchange['question']}")
                    st.markdown(f"**Answer:** {exchange['answer']}")
                    st.markdown(f"**Timestamp:** {exchange['timestamp']}")
                    if exchange.get('sources'):
                        st.markdown(f"**Sources:** {', '.join(exchange['sources'])}")
    
    else:
        st.info("🔍 No conversations yet. Start asking questions to see analytics!")

# Footer
st.markdown("---")
st.markdown("**🚀 Enhanced RAG System v2.0** | Vector Embeddings • Conversation Memory • Multi-Format Support • Quality Metrics")

# Debug panel (collapsible)
with st.expander("🔧 Debug Information"):
    if st.button("🔍 Check Enhanced API Status"):
        health_result = make_api_request("/health")
        if health_result["success"]:
            st.success("✅ Enhanced API is running")
            st.json(health_result["data"])
        else:
            st.error(f"❌ API issue: {health_result['error']}")
    
    if st.button("📊 Get Session Statistics"):
        sessions_result = make_api_request("/sessions")
        if sessions_result["success"]:
            st.json(sessions_result["data"])
        else:
            st.error(f"❌ Error: {sessions_result['error']}")