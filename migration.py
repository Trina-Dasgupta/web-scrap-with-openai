#!/usr/bin/env python3
"""
ChromaDB Migration Script
Automatically updates your main.py to use ChromaDB instead of HybridDocumentStore
"""

import os
import shutil
import re
from datetime import datetime

def backup_file(filename):
    """Create a backup of the original file"""
    backup_name = f"{filename}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filename, backup_name)
    print(f"✅ Created backup: {backup_name}")
    return backup_name

def update_requirements():
    """Add ChromaDB dependencies to requirements.txt"""
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print("⚠️  requirements.txt not found, creating new one...")
        with open(requirements_file, 'w') as f:
            f.write("# ChromaDB dependencies\nchromadb==0.4.15\nsentence-transformers==2.2.2\n")
        return
    
    # Read existing requirements
    with open(requirements_file, 'r') as f:
        content = f.read()
    
    # Check if ChromaDB already added
    if 'chromadb' in content.lower():
        print("✅ ChromaDB already in requirements.txt")
        return
    
    # Add ChromaDB dependencies
    new_deps = "\n# ChromaDB dependencies\nchromadb==0.4.15\nsentence-transformers==2.2.2\n"
    
    with open(requirements_file, 'a') as f:
        f.write(new_deps)
    
    print("✅ Added ChromaDB dependencies to requirements.txt")

def add_chromadb_imports(content):
    """Add ChromaDB imports after existing imports"""
    
    # Find where to insert ChromaDB imports (after collections import)
    import_pattern = r'from collections import Counter, defaultdict'
    
    if re.search(import_pattern, content):
        chromadb_imports = """
# ChromaDB imports for vector database
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import uuid"""
        
        content = re.sub(
            import_pattern,
            r'from collections import Counter, defaultdict' + chromadb_imports,
            content
        )
        print("✅ Added ChromaDB imports")
    else:
        print("⚠️  Could not find collections import, please add ChromaDB imports manually")
    
    return content

def add_global_instance(content):
    """Add ChromaDB global instance"""
    
    # Find the global instances section
    pattern = r'(document_stores = \{\}\s*\n)(conversation_history = ConversationHistory\(\))'
    
    replacement = r'\1chroma_store = ChromaDocumentStore()  # ChromaDB instance\n\2'
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        print("✅ Added ChromaDB global instance")
    else:
        print("⚠️  Could not find global instances section")
    
    return content

def update_endpoints(content):
    """Update scrape and upload endpoints"""
    
    # Update scrape endpoint
    scrape_pattern = r'session_id = "default"\s*\n\s*document_stores\[session_id\] = HybridDocumentStore\(\)\s*\n\s*document_stores\[session_id\]\.add_documents\(documents, session_id\)'
    
    scrape_replacement = '''session_id = "default"
        success = chroma_store.add_documents(documents, session_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to index documents in ChromaDB")'''
    
    if re.search(scrape_pattern, content):
        content = re.sub(scrape_pattern, scrape_replacement, content)
        print("✅ Updated scrape endpoint")
    
    # Update upload endpoint  
    upload_pattern = r'session_id = "default"\s*\n\s*document_stores\[session_id\] = HybridDocumentStore\(\)\s*\n\s*document_stores\[session_id\]\.add_documents\(documents, session_id\)'
    
    if re.search(upload_pattern, content):
        content = re.sub(upload_pattern, scrape_replacement, content)
        print("✅ Updated upload endpoint")
    
    return content

def update_ask_endpoint(content):
    """Update ask endpoint to use ChromaDB"""
    
    # Update the search logic
    ask_pattern = r'if session_id not in document_stores:(.*?)store = document_stores\[session_id\]\s*\n\s*relevant_docs = store\.search\(request\.question, k=4\)'
    
    ask_replacement = '''if session_id not in chroma_store.collections:\\1        
        relevant_docs = chroma_store.search_advanced(session_id, request.question, k=4)'''
    
    if re.search(ask_pattern, content, re.DOTALL):
        content = re.sub(ask_pattern, ask_replacement, content, flags=re.DOTALL)
        print("✅ Updated ask endpoint")
    else:
        print("⚠️  Could not update ask endpoint automatically")
    
    return content

def add_chromadb_class(content):
    """Add ChromaDocumentStore class (this is a big addition)"""
    
    # Find where HybridDocumentStore class starts
    class_start = content.find('class HybridDocumentStore:')
    
    if class_start == -1:
        print("⚠️  Could not find HybridDocumentStore class")
        return content
    
    # Find where the class ends (next class or major section)
    class_end = content.find('\nclass ', class_start + 1)
    if class_end == -1:
        class_end = content.find('\n# Global instances', class_start)
    if class_end == -1:
        class_end = content.find('\ndocument_stores = {}', class_start)
    
    if class_end == -1:
        print("⚠️  Could not determine end of HybridDocumentStore class")
        return content
    
    # ChromaDB class implementation
    chromadb_class = '''class ChromaDocumentStore:
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
    
   def get_or_create_collection(self, session_id: str, reset_if_exists: bool = False):
    """Get existing collection or create new one - PRESERVES DATA by default"""
    collection_name = f"session_{session_id}".replace("-", "_")
    
    try:
        # Check if we already have this collection loaded
        if session_id in self.collections and not reset_if_exists:
            document_count = self.collections[session_id].count()
            logger.info(f"✅ Using existing loaded collection: {collection_name} ({document_count} docs)")
            return self.collections[session_id]
        
        # Try to get existing collection first (PRESERVE DATA)
        if not reset_if_exists:
            try:
                collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                self.collections[session_id] = collection
                document_count = collection.count()
                logger.info(f"✅ Retrieved existing collection '{collection_name}' with {document_count} documents")
                return collection
                
            except Exception:
                logger.info(f"Collection '{collection_name}' doesn't exist, creating new one")
        
        # Delete existing if reset requested
        if reset_if_exists:
            try:
                self.client.delete_collection(collection_name)
                logger.info(f"🗑️ Deleted existing collection: {collection_name}")
            except Exception:
                pass  # Collection didn't exist
        
        # Create new collection
        collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={
                "session_id": session_id, 
                "created_at": datetime.now().isoformat(),
                "version": "2.0"
            }
        )
        
        self.collections[session_id] = collection
        logger.info(f"✅ Created new collection: {collection_name}")
        return collection
        
    except Exception as e:
        logger.error(f"❌ Error with collection {collection_name}: {e}")
        raise e



    
    def add_documents(self, documents, session_id="default"):
    """Add documents with persistence verification"""
    try:
        logger.info(f"📝 Adding {len(documents)} documents to session '{session_id}'")
        
        # Get or create collection (preserve existing data)
        collection = self.get_or_create_collection(session_id, reset_if_exists=False)
        
        # Check existing document count
        before_count = collection.count()
        logger.info(f"Collection had {before_count} documents before adding")
        
        # Prepare documents for ChromaDB
        doc_texts = []
        doc_metadatas = []
        doc_ids = []
        
        for i, doc in enumerate(documents):
            # Create unique ID with timestamp to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            doc_id = f"{session_id}_{timestamp}_{i}_{uuid.uuid4().hex[:8]}"
            
            doc_texts.append(doc['content'])
            doc_metadatas.append({
                'source': doc.get('source', 'unknown'),
                'chunk_id': doc.get('id', i),
                'word_count': len(doc['content'].split()),
                'char_count': len(doc['content']),
                'added_at': datetime.now().isoformat(),
                'session_id': session_id
            })
            doc_ids.append(doc_id)
        
        # Add to ChromaDB
        collection.add(
            documents=doc_texts,
            metadatas=doc_metadatas,
            ids=doc_ids
        )
        
        # Verify persistence
        after_count = collection.count()
        added_count = after_count - before_count
        
        logger.info(f"✅ Successfully added {added_count} documents (total: {after_count})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error adding documents: {e}")
        return False
    
    def search(self, query, k=3):
        """Search documents using ChromaDB semantic search"""
        return self.search_advanced("default", query, k)
    
    def search_advanced(self, session_id, query, k=3, filters=None):
        """Advanced search with session and filters"""
        try:
            if session_id not in self.collections:
                logger.warning(f"No collection found for session {session_id}")
                return []
            
            collection = self.collections[session_id]
            
            results = collection.query(
                query_texts=[query],
                n_results=k,
                where=filters
            )
            
            formatted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'similarity_score': 1.0 - results['distances'][0][i],
                        'source': results['metadatas'][0][i].get('source', 'unknown') if results['metadatas'][0] else 'unknown'
                    }
                    formatted_results.append(result)
            
            logger.info(f"ChromaDB search returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []


'''
    
    # Replace the old class with new class
    new_content = content[:class_start] + chromadb_class + content[class_end:]
    print("✅ Replaced HybridDocumentStore with ChromaDocumentStore")
    
    return new_content

def run_migration():
    """Run the complete migration"""
    
    print("🚀 Starting ChromaDB Migration")
    print("=" * 50)
    
    # Check if main.py exists
    if not os.path.exists("main.py"):
        print("❌ main.py not found in current directory")
        return False
    
    # Create backup
    backup_file("main.py")
    
    # Update requirements.txt
    update_requirements()
    
    # Read main.py
    with open("main.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("\n📝 Updating main.py...")
    
    # Apply all transformations
    content = add_chromadb_imports(content)
    content = add_chromadb_class(content)
    content = add_global_instance(content)
    content = update_endpoints(content)
    content = update_ask_endpoint(content)
    
    # Write updated file
    with open("main.py", 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n✅ Migration completed!")
    print("\n📋 Next Steps:")
    print("1. Install new dependencies: pip install chromadb sentence-transformers")
    print("2. Test the server: python main.py")
    print("3. Check health endpoint: http://localhost:8000/health")
    print("4. If issues occur, restore from backup")
    
    return True

def create_test_script():
    """Create a test script to verify the migration"""
    
    test_content = '''#!/usr/bin/env python3
"""
Test script to verify ChromaDB integration
"""

import requests
import time

def test_chromadb_integration():
    """Test the ChromaDB integration"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing ChromaDB Integration")
    print("-" * 40)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health check passed")
            print(f"   Search method: {data.get('search_method', 'Unknown')}")
            print(f"   Collections: {data.get('collections', 'Unknown')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False
    
    # Test 2: Document upload
    print("\\n2. Testing document processing...")
    try:
        test_url = "https://en.wikipedia.org/wiki/Machine_learning"
        response = requests.post(f"{base_url}/scrape", json={"url": test_url})
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Document processing passed")
            print(f"   Message: {data.get('message', 'No message')}")
        else:
            print(f"   ❌ Document processing failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Document processing error: {e}")
        return False
    
    # Test 3: Question answering
    print("\\n3. Testing question answering...")
    try:
        response = requests.post(f"{base_url}/ask", json={
            "question": "What is machine learning?",
            "session_id": "default"
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Question answering passed")
            print(f"   Answer preview: {data.get('answer', '')[:100]}...")
            print(f"   Sources: {len(data.get('sources', []))}")
        else:
            print(f"   ❌ Question answering failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Question answering error: {e}")
        return False
    
    print("\\n🎉 All tests passed! ChromaDB integration is working.")
    return True

if __name__ == "__main__":
    print("Make sure your server is running: python main.py")
    print("Press Enter to start testing...")
    input()
    
    success = test_chromadb_integration()
    
    if success:
        print("\\n✅ Migration successful!")
    else:
        print("\\n❌ Issues detected. Check server logs.")
'''
    
    with open("test_chromadb.py", 'w') as f:
        f.write(test_content)
    
    print("✅ Created test_chromadb.py")

def main():
    """Main migration function"""
    
    print("🔧 ChromaDB Migration Tool")
    print("This will update your RAG system to use ChromaDB")
    print()
    
    # Confirm migration
    response = input("Do you want to proceed with the migration? (y/N): ")
    if response.lower() != 'y':
        print("Migration cancelled.")
        return
    
    # Run migration
    success = run_migration()
    
    if success:
        # Create test script
        create_test_script()
        
        print("\\n" + "=" * 50)
        print("🎉 ChromaDB Migration Complete!")
        print("=" * 50)
        
        print("\\nFiles modified:")
        print("- main.py (backed up)")
        print("- requirements.txt (updated)")
        print("- test_chromadb.py (created)")
        
        print("\\nTo complete the setup:")
        print("1. pip install -r requirements.txt")
        print("2. python main.py")
        print("3. python test_chromadb.py (to verify)")
        
    else:
        print("\\n❌ Migration failed. Please check the errors above.")

if __name__ == "__main__":
    main()