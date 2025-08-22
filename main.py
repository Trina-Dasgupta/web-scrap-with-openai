from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import requests
from bs4 import BeautifulSoup
import re
from newspaper import Article
import logging
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import io
import json
from datetime import datetime
import math
from collections import Counter, defaultdict
# Add these imports after your existing imports
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import uuid

# Enhanced document processing
import PyPDF2
from docx import Document as DocxDocument
import openpyxl

# Alternative to sentence transformers - using OpenAI embeddings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Advanced text processing
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass
except ImportError:
    NLTK_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced RAG Web Scraper API", version="2.0.0")

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
    use_conversation_history: Optional[bool] = True

class URLResponse(BaseModel):
    success: bool
    message: str
    content_preview: Optional[str] = None
    word_count: Optional[int] = None
    document_type: Optional[str] = None

class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str
    relevance_scores: Optional[List[float]] = None
    conversation_context: Optional[List[Dict]] = None
    evaluation_metrics: Optional[Dict] = None

class ConversationHistory:
    """Manages conversation history for each session"""
    def __init__(self):
        self.history: Dict[str, List[Dict]] = {}
    
    def add_exchange(self, session_id: str, question: str, answer: str, sources: List[str]):
        if session_id not in self.history:
            self.history[session_id] = []
        
        self.history[session_id].append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "sources": sources
        })
        
        # Keep only last 10 exchanges to prevent memory bloat
        if len(self.history[session_id]) > 10:
            self.history[session_id] = self.history[session_id][-10:]
    
    def get_context(self, session_id: str, max_exchanges: int = 3) -> str:
        if session_id not in self.history:
            return ""
        
        recent_history = self.history[session_id][-max_exchanges:]
        context_parts = []
        
        for exchange in recent_history:
            context_parts.append(f"Previous Q: {exchange['question']}")
            context_parts.append(f"Previous A: {exchange['answer']}")
        
        return "\n".join(context_parts)
    
    def clear_session(self, session_id: str):
        if session_id in self.history:
            del self.history[session_id]

# Enhanced document store with TF-IDF + OpenAI embeddings
class ChromaDocumentStore:
    """Enhanced document store using ChromaDB for vector storage"""
    
    def __init__(self, persist_directory: str = "./chromadb_data"):
        """Initialize ChromaDB client with persistence"""
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False
            )
        )
        
        # Setup embedding function
        self.embedding_function = self._setup_embedding_function()
        
        # Store collections by session
        self.collections = {}
        
        logger.info(f"ChromaDB initialized with persistence at {persist_directory}")
    
    def _setup_embedding_function(self):
        """Setup embedding function (sentence transformers or OpenAI)"""
        try:
            # Try to use sentence transformers first
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            logger.info("Using SentenceTransformer embeddings (all-MiniLM-L6-v2)")
            return embedding_function
            
        except ImportError:
            logger.warning("SentenceTransformers not available, falling back to OpenAI")
            
            # Fallback to OpenAI embeddings
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key and openai_api_key != "your_openai_api_key_here":
                embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai_api_key,
                    model_name="text-embedding-ada-002"
                )
                logger.info("Using OpenAI embeddings (text-embedding-ada-002)")
                return embedding_function
            
            # Ultimate fallback to default
            logger.warning("No embedding service configured, using default")
            return embedding_functions.DefaultEmbeddingFunction()
    
    def create_collection(self, session_id: str, reset_if_exists: bool = True):
        """Create or get a collection for a session"""
        collection_name = f"session_{session_id}".replace("-", "_")
        
        try:
            if reset_if_exists:
                # Delete existing collection if it exists
                try:
                    self.client.delete_collection(collection_name)
                    logger.info(f"Deleted existing collection: {collection_name}")
                except Exception:
                    pass  # Collection didn't exist
            
            # Create new collection
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"session_id": session_id, "created_at": datetime.now().isoformat()}
            )
            
            self.collections[session_id] = collection
            logger.info(f"Created ChromaDB collection: {collection_name}")
            return collection
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            # Try to get existing collection
            try:
                collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                self.collections[session_id] = collection
                logger.info(f"Retrieved existing collection: {collection_name}")
                return collection
            except Exception as e2:
                logger.error(f"Error retrieving collection: {e2}")
                raise e
    
    def add_documents(self, documents, session_id="default"):
        """Add documents to ChromaDB collection"""
        try:
            # Get or create collection
            if session_id not in self.collections:
                self.create_collection(session_id)
            
            collection = self.collections[session_id]
            
            # Prepare data for ChromaDB
            doc_texts = []
            doc_metadatas = []
            doc_ids = []
            
            for i, doc in enumerate(documents):
                doc_id = f"{session_id}_{i}_{uuid.uuid4().hex[:8]}"
                
                doc_texts.append(doc['content'])
                doc_metadatas.append({
                    'source': doc.get('source', 'unknown'),
                    'chunk_id': doc.get('id', i),
                    'word_count': len(doc['content'].split()),
                    'char_count': len(doc['content']),
                    'added_at': datetime.now().isoformat()
                })
                doc_ids.append(doc_id)
            
            # Add to ChromaDB (embeddings generated automatically)
            collection.add(
                documents=doc_texts,
                metadatas=doc_metadatas,
                ids=doc_ids
            )
            
            logger.info(f"Added {len(documents)} documents to ChromaDB collection")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            return False
    
    def search(self, query, k=3):
        """Search documents using ChromaDB semantic search"""
        # Keep the same method signature for compatibility
        return self.search_advanced("default", query, k)
    
    def search_advanced(self, session_id, query, k=3, filters=None):
        """Advanced search with session and filters"""
        try:
            if session_id not in self.collections:
                logger.warning(f"No collection found for session {session_id}")
                return []
            
            collection = self.collections[session_id]
            
            # Perform semantic search
            results = collection.query(
                query_texts=[query],
                n_results=k,
                where=filters  # Optional metadata filtering
            )
            
            # Format results to match existing interface
            formatted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'similarity_score': 1.0 - results['distances'][0][i],  # Convert distance to similarity
                        'source': results['metadatas'][0][i].get('source', 'unknown') if results['metadatas'][0] else 'unknown'
                    }
                    formatted_results.append(result)
            
            logger.info(f"ChromaDB search returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []
# Advanced text chunking
class SemanticChunker:
    """Advanced text chunking based on semantic boundaries"""
    
    @staticmethod
    def chunk_by_sentences(text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Chunk text by sentence boundaries"""
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(text)
            except:
                sentences = re.split(r'[.!?]+', text)
        else:
            # Fallback sentence splitting
            sentences = re.split(r'[.!?]+', text)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_size = len(sentence.split())
            
            if current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Create overlap
                words = current_chunk.split()
                if len(words) > overlap:
                    overlap_text = " ".join(words[-overlap:])
                    current_chunk = overlap_text + " " + sentence
                    current_size = len(current_chunk.split())
                else:
                    current_chunk = sentence
                    current_size = sentence_size
            else:
                current_chunk += " " + sentence
                current_size += sentence_size
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

# Document processors
class DocumentProcessor:
    """Process different document types"""
    
    @staticmethod
    def process_pdf(file_content: bytes) -> str:
        """Extract text from PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return ""
    
    @staticmethod
    def process_docx(file_content: bytes) -> str:
        """Extract text from DOCX"""
        try:
            doc = DocxDocument(io.BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error processing DOCX: {e}")
            return ""
    
    @staticmethod
    def process_xlsx(file_content: bytes) -> str:
        """Extract text from XLSX"""
        try:
            workbook = openpyxl.load_workbook(io.BytesIO(file_content))
            text = ""
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                text += f"Sheet: {sheet_name}\n"
                for row in sheet.iter_rows(values_only=True):
                    row_text = " | ".join([str(cell) if cell is not None else "" for cell in row])
                    if row_text.strip():
                        text += row_text + "\n"
                text += "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error processing XLSX: {e}")
            return ""

# Evaluation metrics
class RAGEvaluator:
    """Evaluate RAG responses"""
    
    def __init__(self):
        self.rouge_scorer = None
        try:
            from rouge_score import rouge_scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        except ImportError:
            logger.warning("ROUGE scorer not available")
    
    def evaluate_response(self, question: str, answer: str, retrieved_docs: List[Dict]) -> Dict:
        """Evaluate the quality of a RAG response"""
        metrics = {}
        
        # Calculate relevance scores
        if retrieved_docs and all('similarity_score' in doc for doc in retrieved_docs):
            scores = [doc['similarity_score'] for doc in retrieved_docs if doc.get('similarity_score')]
            if scores:
                metrics['avg_relevance_score'] = float(np.mean(scores))
                metrics['max_relevance_score'] = float(max(scores))
                metrics['min_relevance_score'] = float(min(scores))
        
        # Calculate answer length metrics
        metrics['answer_length'] = len(answer.split())
        metrics['context_coverage'] = self._calculate_context_coverage(answer, retrieved_docs)
        
        return metrics
    
    def _calculate_context_coverage(self, answer: str, retrieved_docs: List[Dict]) -> float:
        """Calculate how much of the retrieved context is used in the answer"""
        if not retrieved_docs:
            return 0.0
        
        context = " ".join([doc['content'] for doc in retrieved_docs])
        
        # Simple word overlap calculation
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        if not context_words:
            return 0.0
        
        overlap = len(answer_words.intersection(context_words))
        return overlap / len(context_words)

# Global instances
document_stores = {}
chroma_store = ChromaDocumentStore()
conversation_history = ConversationHistory()
evaluator = RAGEvaluator()

# Enhanced web scraper
class EnhancedWebScraper:
    @staticmethod
    def extract_with_beautifulsoup(url: str):
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
            
            content_type = response.headers.get('content-type', '').lower()
            doc_type = "webpage"
            
            if 'pdf' in content_type:
                doc_type = "pdf"
                content = DocumentProcessor.process_pdf(response.content)
                return f"Title: PDF Document\n\nContent:\n{content}", doc_type
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside", 
                               "iframe", "noscript", "form", "button"]):
                element.decompose()
            
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title"
            
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
            
            content = re.sub(r'\n\s*\n', '\n\n', content)
            content = re.sub(r' +', ' ', content)
            
            final_content = f"Title: {title_text}\n\nContent:\n{content}"
            return final_content, doc_type
            
        except Exception as e:
            logger.error(f"BeautifulSoup extraction failed: {e}")
            return None, None

    @staticmethod
    def extract_with_newspaper(url: str):
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
            return content, "article"
        except Exception as e:
            logger.error(f"Newspaper extraction failed: {e}")
            return None, None

    @classmethod
    def scrape_url(cls, url: str):
        """Main scraping method"""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        logger.info(f"Scraping URL: {url}")
        
        content, doc_type = cls.extract_with_beautifulsoup(url)
        if content and len(content.strip()) > 100:
            logger.info(f"BeautifulSoup extraction successful - type: {doc_type}")
            return content, doc_type
        
        content, doc_type = cls.extract_with_newspaper(url)
        if content and len(content.strip()) > 100:
            logger.info(f"Newspaper extraction successful - type: {doc_type}")
            return content, doc_type
        
        raise Exception("Could not extract meaningful content from the URL")

def call_openai_api_enhanced(question: str, context: str, conversation_context: str = "") -> str:
    """Enhanced OpenAI API call with conversation context"""
    try:
        import openai
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key or openai_api_key == "your_openai_api_key_here":
            return "OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file"
        
        openai.api_key = openai_api_key
        
        max_context_length = 2500
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        conversation_part = f"Conversation History:\n{conversation_context}\n" if conversation_context else ""
        
        prompt = f"""Based on the following context and conversation history, answer the question. If you cannot find the answer in the context, say "I cannot find that information in the provided context."

{conversation_part}Current Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=600,
                temperature=0.1,
                stop=None
            )
            return response.choices[0].text.strip()
            
        except openai.error.InvalidRequestError as e:
            if "model" in str(e).lower():
                try:
                    response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=prompt,
                        max_tokens=600,
                        temperature=0.1,
                        stop=None
                    )
                    return response.choices[0].text.strip()
                except:
                    return "Both OpenAI models are currently unavailable. Please try again later."
            return f"OpenAI request error: {str(e)}"
            
        except Exception as e:
            return f"OpenAI API error: {str(e)}"
            
    except Exception as e:
        return f"Error processing your question: {str(e)}"

# API Endpoints
@app.post("/scrape", response_model=URLResponse)
async def scrape_website(request: URLRequest):
    """Enhanced scraping with document type detection"""
    try:
        logger.info(f"Starting to scrape URL: {request.url}")
        
        def scrape_with_timeout():
            return EnhancedWebScraper.scrape_url(request.url)
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(scrape_with_timeout)
                content, doc_type = future.result(timeout=20)
        except Exception as e:
            if "timeout" in str(e).lower():
                raise HTTPException(
                    status_code=408, 
                    detail="The website is taking too long to respond. Please try a different URL."
                )
            raise e
        
        if not content or len(content.strip()) < 50:
            raise HTTPException(status_code=400, detail="Could not extract meaningful content from URL")
        
        chunks = SemanticChunker.chunk_by_sentences(content, max_chunk_size=800, overlap=100)
        logger.info(f"Split into {len(chunks)} semantic chunks")
        
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                'id': i,
                'content': chunk,
                'source': request.url
            })
        
        session_id = "default"
        success = chroma_store.add_documents(documents, session_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to index documents in ChromaDB")
        
        conversation_history.clear_session(session_id)
        
        logger.info("Successfully created enhanced document store")
        
        word_count = len(content.split())
        preview = content[:500] + "..." if len(content) > 500 else content
        
        return URLResponse(
            success=True,
            message=f"Content successfully scraped and indexed using hybrid search (TF-IDF + OpenAI embeddings)",
            content_preview=preview,
            word_count=word_count,
            document_type=doc_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=URLResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process different document types"""
    try:
        logger.info(f"Processing uploaded file: {file.filename}")
        
        file_content = await file.read()
        content = ""
        doc_type = "unknown"
        
        filename_lower = file.filename.lower()
        
        if filename_lower.endswith('.pdf'):
            content = DocumentProcessor.process_pdf(file_content)
            doc_type = "pdf"
        elif filename_lower.endswith('.docx'):
            content = DocumentProcessor.process_docx(file_content)
            doc_type = "docx"
        elif filename_lower.endswith('.xlsx'):
            content = DocumentProcessor.process_xlsx(file_content)
            doc_type = "xlsx"
        elif filename_lower.endswith(('.txt', '.md')):
            content = file_content.decode('utf-8')
            doc_type = "text"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Supported: PDF, DOCX, XLSX, TXT, MD")
        
        if not content or len(content.strip()) < 50:
            raise HTTPException(status_code=400, detail="Could not extract meaningful content from the file")
        
        final_content = f"Title: {file.filename}\n\nContent:\n{content}"
        
        chunks = SemanticChunker.chunk_by_sentences(final_content, max_chunk_size=800, overlap=100)
        logger.info(f"Split into {len(chunks)} semantic chunks")
        
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                'id': i,
                'content': chunk,
                'source': file.filename
            })
        
        session_id = "default"
        success = chroma_store.add_documents(documents, session_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to index documents in ChromaDB")
        
        conversation_history.clear_session(session_id)
        
        logger.info("Successfully processed uploaded document")
        
        word_count = len(content.split())
        preview = final_content[:500] + "..." if len(final_content) > 500 else final_content
        
        return URLResponse(
            success=True,
            message=f"File successfully processed and indexed using hybrid search",
            content_preview=preview,
            word_count=word_count,
            document_type=doc_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AnswerResponse)
async def ask_question_enhanced(request: QuestionRequest):
    """Enhanced question answering with conversation memory and evaluation"""
    try:
        session_id = request.session_id or "default"
        logger.info(f"Processing enhanced question: {request.question}")
        # Check if collection exists
        if session_id not in chroma_store.collections:
            return AnswerResponse(
                answer="No content has been scraped yet. Please scrape a URL or upload a document first.",
                sources=[],
                session_id=session_id
            )
        
        relevant_docs = chroma_store.search_advanced(session_id, request.question, k=4)
        logger.info(f"Found {len(relevant_docs)} relevant documents using hybrid search")
        
        if not relevant_docs:
            return AnswerResponse(
                answer="I couldn't find relevant information to answer your question.",
                sources=[],
                session_id=session_id
            )
        
        context = "\n\n".join([doc['content'] for doc in relevant_docs])
        sources = list(set([doc['source'] for doc in relevant_docs]))
        relevance_scores = [doc.get('similarity_score', 0) for doc in relevant_docs]
        
        conversation_context = ""
        if request.use_conversation_history:
            conversation_context = conversation_history.get_context(session_id)
        
        logger.info(f"Context length: {len(context)} characters")
        logger.info(f"Using conversation history: {bool(conversation_context)}")
        
        # Get enhanced answer from OpenAI
        logger.info("Calling enhanced OpenAI API...")
        answer = call_openai_api_enhanced(request.question, context, conversation_context)
        logger.info("Enhanced OpenAI response received")
        
        # Evaluate the response
        metrics = evaluator.evaluate_response(request.question, answer, relevant_docs)
        
        # Add to conversation history
        conversation_history.add_exchange(session_id, request.question, answer, sources)
        
        # Get recent conversation for response
        recent_conversation = conversation_history.history.get(session_id, [])[-3:]
        
        return AnswerResponse(
            answer=answer,
            sources=sources,
            session_id=session_id,
            relevance_scores=relevance_scores,
            conversation_context=recent_conversation,
            evaluation_metrics=metrics
        )
        
    except Exception as e:
        logger.error(f"Error in enhanced ask_question: {type(e).__name__}: {str(e)}")
        return AnswerResponse(
            answer=f"An error occurred while processing your question: {str(e)}",
            sources=[],
            session_id=request.session_id or "default"
        )

@app.get("/conversation/{session_id}")
async def get_conversation_history(session_id: str = "default"):
    """Get conversation history for a session"""
    return {
        "session_id": session_id,
        "conversation": conversation_history.history.get(session_id, [])
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with ChromaDB status"""
    try:
        # Test ChromaDB connection
        collections = chroma_store.client.list_collections()
        chromadb_status = "connected"
        collection_count = len(collections)
    except Exception as e:
        chromadb_status = f"error: {str(e)}"
        collection_count = 0
    
    # Check available features
    features_status = {
        "chromadb": chromadb_status,
        "embedding_function": str(type(chroma_store.embedding_function).__name__),
        "conversation_memory": "enabled",
        "nltk": "available" if NLTK_AVAILABLE else "fallback_regex",
        "document_processing": "available"
    }
    
    return {
        "status": "healthy", 
        "search_method": "ChromaDB Vector Database",
        "features": features_status,
        "collections": collection_count,
        "supported_formats": ["web_pages", "pdf", "docx", "xlsx", "txt", "md"],
        "version": "2.0.0 + ChromaDB"
    }
@app.delete("/clear/{session_id}")
async def clear_session_enhanced(session_id: str = "default"):
    """Clear session with ChromaDB"""
    try:
        # Delete from ChromaDB
        if session_id in chroma_store.collections:
            collection_name = f"session_{session_id}".replace("-", "_")
            chroma_store.client.delete_collection(collection_name)
            del chroma_store.collections[session_id]
        
        # Clear conversation history
        conversation_history.clear_session(session_id)
        
        return {"message": f"Session {session_id} cleared from ChromaDB and conversation history"}
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        return {"message": f"Error clearing session {session_id}: {str(e)}"}
@app.get("/sessions")
async def list_sessions_enhanced():
    """List all active sessions with stats"""
    session_stats = {}
    for session_id, store in document_stores.items():
        conversation_count = len(conversation_history.history.get(session_id, []))
        session_stats[session_id] = {
            "document_count": len(store.documents),
            "conversation_exchanges": conversation_count,
            "search_method": "hybrid_tfidf_openai"
        }
    
    return {
        "active_sessions": list(document_stores.keys()),
        "total_sessions": len(document_stores),
        "session_stats": session_stats
    }

@app.get("/")
async def root():
    """Enhanced root endpoint"""
    return {
        "message": "Enhanced RAG Web Scraper API",
        "version": "2.0.0",
        "features": [
            "Hybrid search (TF-IDF + OpenAI embeddings)",
            "Multiple document format support (PDF, DOCX, XLSX, TXT, MD)",
            "Conversation memory and context",
            "Semantic text chunking",
            "Response quality evaluation",
            "Enhanced web scraping"
        ],
        "endpoints": {
            "POST /scrape": "Scrape content from URL with enhanced processing",
            "POST /upload": "Upload and process documents (PDF, DOCX, XLSX, TXT, MD)",
            "POST /ask": "Ask questions with conversation context and evaluation",
            "GET /conversation/{session_id}": "Get conversation history",
            "GET /health": "Health check with feature status",
            "DELETE /clear/{session_id}": "Clear session and conversation",
            "GET /sessions": "List sessions with statistics"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)