from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
import re
from newspaper import Article
import logging
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Simple TF-IDF based search without external dependencies
from collections import Counter, defaultdict
import math

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Web Scraper API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class URLRequest(BaseModel):
    url: str

class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class URLResponse(BaseModel):
    success: bool
    message: str
    content_preview: Optional[str] = None
    word_count: Optional[int] = None

class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str

# Simple document store and search
class SimpleDocumentStore:
    def __init__(self):
        self.documents = []
        self.tfidf_index = {}
        self.doc_freqs = defaultdict(int)
        self.total_docs = 0

    def add_documents(self, docs):
        """Add documents and build TF-IDF index"""
        self.documents = docs
        self.total_docs = len(docs)
        self._build_tfidf_index()

    def _tokenize(self, text):
        """Simple tokenization"""
        return re.findall(r'\w+', text.lower())

    def _build_tfidf_index(self):
        """Build TF-IDF index for documents"""
        # Calculate document frequencies
        for doc in self.documents:
            words = set(self._tokenize(doc['content']))
            for word in words:
                self.doc_freqs[word] += 1

        # Build TF-IDF vectors for each document
        for i, doc in enumerate(self.documents):
            words = self._tokenize(doc['content'])
            word_counts = Counter(words)
            total_words = len(words)
            
            tfidf_vector = {}
            for word, count in word_counts.items():
                tf = count / total_words
                idf = math.log(self.total_docs / (self.doc_freqs[word] + 1))
                tfidf_vector[word] = tf * idf
            
            self.tfidf_index[i] = tfidf_vector

    def search(self, query, k=3):
        """Search for most relevant documents"""
        query_words = self._tokenize(query)
        query_vector = {}
        
        # Build query TF-IDF vector
        word_counts = Counter(query_words)
        total_words = len(query_words)
        
        for word, count in word_counts.items():
            tf = count / total_words
            idf = math.log(self.total_docs / (self.doc_freqs.get(word, 0) + 1))
            query_vector[word] = tf * idf

        # Calculate cosine similarity with each document
        scores = []
        for doc_id, doc_vector in self.tfidf_index.items():
            score = self._cosine_similarity(query_vector, doc_vector)
            scores.append((doc_id, score))

        # Sort by score and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [self.documents[doc_id] for doc_id, score in scores[:k] if score > 0]

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        intersection = set(vec1.keys()) & set(vec2.keys())
        if not intersection:
            return 0.0

        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        
        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        
        if not denominator:
            return 0.0
        
        return numerator / denominator

# Global document stores for different sessions
document_stores = {}

class WebScraper:
    @staticmethod
    def extract_with_beautifulsoup(url: str) -> str:
        """Extract content using BeautifulSoup"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside", 
                               "iframe", "noscript", "form", "button"]):
                element.decompose()
            
            # Get title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title"
            
            # Find main content
            content_selectors = [
                'article', 'main', '[role="main"]', '.content', '.post-content', 
                '.entry-content', '.article-content', '#content', '.main-content'
            ]
            
            content = None
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text(separator='\n', strip=True)
                    if len(content) > 200:
                        break
            
            if not content or len(content) < 200:
                body = soup.find('body')
                content = body.get_text(separator='\n', strip=True) if body else ""
            
            # Clean up content
            content = re.sub(r'\n\s*\n', '\n\n', content)
            content = re.sub(r' +', ' ', content)
            
            final_content = f"Title: {title_text}\n\nContent:\n{content}"
            return final_content
            
        except Exception as e:
            logger.error(f"BeautifulSoup extraction failed: {e}")
            return None

    @staticmethod
    def extract_with_newspaper(url: str) -> str:
        """Extract content using newspaper3k"""
        try:
            article = Article(url)
            article.config.request_timeout = 15
            article.download()
            article.parse()
            
            content = f"Title: {article.title}\n\n"
            if article.authors:
                content += f"Authors: {', '.join(article.authors)}\n\n"
            if article.publish_date:
                content += f"Published: {article.publish_date}\n\n"
            
            content += f"Content:\n{article.text}"
            return content
        except Exception as e:
            logger.error(f"Newspaper extraction failed: {e}")
            return None

    @classmethod
    def scrape_url(cls, url: str) -> str:
        """Main scraping method"""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        logger.info(f"Scraping URL: {url}")
        
        # Try BeautifulSoup first
        content = cls.extract_with_beautifulsoup(url)
        if content and len(content.strip()) > 100:
            logger.info("BeautifulSoup extraction successful")
            return content
        
        # Try newspaper as fallback
        content = cls.extract_with_newspaper(url)
        if content and len(content.strip()) > 100:
            logger.info("Newspaper extraction successful")
            return content
        
        raise Exception("Could not extract meaningful content from the URL")

def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Simple text splitter"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk = ' '.join(chunk_words)
        chunks.append(chunk)
        
        if i + chunk_size >= len(words):
            break
    
    return chunks

def call_openai_api(question: str, context: str) -> str:
    """Call OpenAI API with proper error handling"""
    try:
        # Check if openai is available
        try:
            import openai
        except ImportError:
            return "OpenAI package is not installed. Please install it with: pip install openai==0.28.1"
        
        # Get API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key or openai_api_key == "your_openai_api_key_here":
            return "OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file"
        
        # Set up OpenAI
        openai.api_key = openai_api_key
        
        # Truncate context if too long
        max_context_length = 3000
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        prompt = f"""Based on the following context, answer the question. If you cannot find the answer in the context, say "I cannot find that information in the provided context."

Context:
{context}

Question: {question}

Answer:"""
        
        # Try main model first
        try:
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=500,
                temperature=0.1,
                stop=None
            )
            return response.choices[0].text.strip()
            
        except openai.error.InvalidRequestError as e:
            # Try fallback model
            if "model" in str(e).lower():
                try:
                    response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=prompt,
                        max_tokens=500,
                        temperature=0.1,
                        stop=None
                    )
                    return response.choices[0].text.strip()
                except:
                    return "Both OpenAI models are currently unavailable. Please try again later."
            return f"OpenAI request error: {str(e)}"
            
        except openai.error.AuthenticationError:
            return "Authentication failed. Please check your OpenAI API key."
            
        except openai.error.RateLimitError:
            return "Rate limit exceeded. Please try again in a moment."
            
        except openai.error.APIConnectionError:
            return "Unable to connect to OpenAI. Please check your internet connection."
            
    except Exception as e:
        return f"Error processing your question: {str(e)}"

@app.post("/scrape", response_model=URLResponse)
async def scrape_website(request: URLRequest):
    """Scrape content from a website URL and create document store"""
    try:
        logger.info(f"Starting to scrape URL: {request.url}")
        
        # Scrape with timeout
        def scrape_with_timeout():
            return WebScraper.scrape_url(request.url)
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(scrape_with_timeout)
                content = future.result(timeout=20)
        except Exception as e:
            if "timeout" in str(e).lower():
                raise HTTPException(
                    status_code=408, 
                    detail="The website is taking too long to respond. Please try a different URL."
                )
            raise e
        
        if not content or len(content.strip()) < 50:
            raise HTTPException(status_code=400, detail="Could not extract meaningful content from URL")
        
        # Split content into chunks
        chunks = split_text(content, chunk_size=1000, overlap=200)
        logger.info(f"Split into {len(chunks)} text chunks")
        
        # Create document objects
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                'id': i,
                'content': chunk,
                'source': request.url
            })
        
        # Create document store for this session
        session_id = "default"
        document_stores[session_id] = SimpleDocumentStore()
        document_stores[session_id].add_documents(documents)
        
        logger.info("Successfully created document store")
        
        word_count = len(content.split())
        preview = content[:500] + "..." if len(content) > 500 else content
        
        return URLResponse(
            success=True,
            message="Content successfully scraped and indexed using simple TF-IDF search",
            content_preview=preview,
            word_count=word_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about the scraped content - simplified version with debug info"""
    try:
        # Log the incoming request
        logger.info(f"Received ask request: {request}")
        logger.info(f"Question: '{request.question}'")
        logger.info(f"Session ID: '{request.session_id}'")
        
        session_id = request.session_id or "default"
        logger.info(f"Using session_id: {session_id}")
        logger.info(f"Available sessions: {list(document_stores.keys())}")
        
        # Check if content exists
        if session_id not in document_stores:
            logger.warning(f"Session {session_id} not found in document_stores")
            logger.info(f"Available sessions: {list(document_stores.keys())}")
            return AnswerResponse(
                answer="No content has been scraped yet. Please scrape a URL first.",
                sources=[],
                session_id=session_id
            )
        
        # Search for relevant documents
        logger.info("Searching for relevant documents...")
        store = document_stores[session_id]
        relevant_docs = store.search(request.question, k=3)
        logger.info(f"Found {len(relevant_docs)} relevant documents")
        
        if not relevant_docs:
            logger.warning("No relevant documents found in search")
            return AnswerResponse(
                answer="I couldn't find relevant information to answer your question. The content might be too short or not relevant to your query.",
                sources=[],
                session_id=session_id
            )
        
        # Combine relevant content
        context = "\n\n".join([doc['content'] for doc in relevant_docs])
        sources = list(set([doc['source'] for doc in relevant_docs]))
        
        logger.info(f"Context length: {len(context)} characters")
        logger.info(f"Sources: {sources}")
        
        # Get answer from OpenAI (this function handles all errors internally)
        logger.info("Calling OpenAI API...")
        answer = call_openai_api(request.question, context)
        logger.info(f"OpenAI response received: {answer[:50]}...")
        
        return AnswerResponse(
            answer=answer,
            sources=sources,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Exception in ask_question: {type(e).__name__}: {str(e)}")
        logger.error(f"Exception details: {e}")
        # Return error as a response instead of raising exception
        return AnswerResponse(
            answer=f"An error occurred while processing your question: {str(e)}",
            sources=[],
            session_id=getattr(request, 'session_id', 'default') or "default"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "search_method": "Simple TF-IDF",
        "version": "1.0.0"
    }

@app.delete("/clear/{session_id}")
async def clear_session(session_id: str = "default"):
    """Clear a specific session's document store"""
    if session_id in document_stores:
        try:
            del document_stores[session_id]
            return {"message": f"Session {session_id} cleared successfully"}
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return {"message": f"Error clearing session {session_id}: {str(e)}"}
    return {"message": f"Session {session_id} not found"}

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "active_sessions": list(document_stores.keys()),
        "total_sessions": len(document_stores)
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Web Scraper API",
        "version": "1.0.0",
        "search_method": "Simple TF-IDF",
        "endpoints": {
            "POST /scrape": "Scrape content from URL",
            "POST /ask": "Ask question about scraped content",
            "GET /health": "Health check",
            "DELETE /clear/{session_id}": "Clear session",
            "GET /sessions": "List active sessions"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)