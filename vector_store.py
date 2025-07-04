"""
Simplified but Reliable Vector Store
Works with Python 3.13 and current dependencies
"""

import json
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class SimpleDocument:
    """Simple document class compatible with our system."""
    def __init__(self, page_content: str, metadata: Optional[Dict] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class AdvancedVectorStore:
    """Simplified but reliable vector store using TF-IDF."""
    
    def __init__(self, persist_directory: str = "chroma_advanced_db", model_name: str = "tfidf", device: str = "cpu"):
        """Initialize the vector store."""
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        self.model_name = model_name
        self.device = device
        
        # Initialize embedding model
        self.embedding_model = None
        
        # Storage
        self.documents = []
        self.vectors = None
        self.metadata = []
        self.is_ready = False
        
        # Initialize the embedding model
        self._initialize_embedding_model()
        
        # Try to load existing data
        self._load_existing()
        logger.info(f"✅ Initialized vector store: {model_name}")
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model based on model_name."""
        try:
            print(f"🔧 Initializing embedding model: {self.model_name}")
            
            if self.model_name.startswith("sentence-transformers"):
                self._initialize_sentence_transformers()
            elif self.model_name == "openai":
                self._initialize_openai_embeddings()
            else:
                # Default to sentence-transformers for unknown models
                print(f"⚠️ Unknown model {self.model_name}, using sentence-transformers")
                self.model_name = "sentence-transformers"
                self._initialize_sentence_transformers()
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            logger.info("Falling back to sentence-transformers")
            self.model_name = "sentence-transformers"
            self._initialize_sentence_transformers()
    
    def _initialize_sentence_transformers(self):
        """Initialize Sentence Transformers model with proper PyTorch handling."""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Map model names to actual model IDs
            model_map = {
                "sentence-transformers": "all-MiniLM-L6-v2",  # Fast, good quality
                "sentence-transformers-quality": "all-mpnet-base-v2",  # High quality
                "sentence-transformers-multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
                "sentence-transformers-qa": "multi-qa-MiniLM-L6-cos-v1"
            }
            
            model_id = model_map.get(self.model_name, "all-MiniLM-L6-v2")
            
            print(f"🔄 Loading Sentence Transformer: {model_id}")
            
            # Set torch backend to avoid meta tensor issues
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            
            # Load model with explicit device and trust_remote_code
            self.embedding_model = SentenceTransformer(
                model_id, 
                device=self.device,
                trust_remote_code=True
            )
            
            # Move model to correct device and ensure it's in eval mode
            self.embedding_model.to(self.device)
            self.embedding_model.eval()
            
            print(f"✅ Sentence Transformer loaded successfully on {self.device}")
            
        except ImportError:
            raise Exception("sentence-transformers not installed. Install with: pip install sentence-transformers")
        except Exception as e:
            raise Exception(f"Failed to load Sentence Transformer: {e}")
    
    def _initialize_openai_embeddings(self):
        """Initialize OpenAI embeddings."""
        try:
            import openai
            import os
            
            # Check for API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise Exception("OPENAI_API_KEY not found in environment variables")
            
            self.embedding_model = openai.OpenAI(api_key=api_key)
            print(f"✅ OpenAI embeddings initialized")
            
        except ImportError:
            raise Exception("openai not installed. Install with: pip install openai")
        except Exception as e:
            raise Exception(f"Failed to initialize OpenAI embeddings: {e}")
    
    def _load_existing(self):
        """Load existing vectorstore if it exists."""
        try:
            data_file = self.persist_directory / "data.json"
            if data_file.exists():
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.documents = data.get('documents', [])
                self.metadata = data.get('metadata', [])
                self.is_ready = data.get('is_ready', False)
                
                # Load vectorizer and vectors if they exist
                vectorizer_file = self.persist_directory / "vectorizer.pkl"
                vectors_file = self.persist_directory / "vectors.pkl"
                
                if self.is_ready and vectorizer_file.exists() and vectors_file.exists():
                    with open(vectorizer_file, 'rb') as f:
                        self.vectorizer = pickle.load(f)
                    with open(vectors_file, 'rb') as f:
                        self.vectors = pickle.load(f)
                
                if self.documents:
                    logger.info(f"✅ Loaded existing vectorstore with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"❌ Error loading existing vectorstore: {e}")
    
    def create_vectorstore(self, documents: List, force_recreate: bool = False) -> bool:
        """Create vectorstore from documents."""
        try:
            if force_recreate:
                self.clear()
            
            # Extract text content from documents
            texts = []
            metadatas = []
            
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    texts.append(doc.page_content)
                    metadatas.append(getattr(doc, 'metadata', {}))
                elif isinstance(doc, str):
                    texts.append(doc)
                    metadatas.append({})
                else:
                    texts.append(str(doc))
                    metadatas.append({})
            
            return self.add_documents(texts, metadatas)
            
        except Exception as e:
            logger.error(f"❌ Error creating vectorstore: {e}")
            return False
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> bool:
        """Add documents to the vector store."""
        try:
            if not texts:
                logger.warning("No texts provided")
                return False
            
            # Clean texts
            clean_texts = [text.strip() for text in texts if text and text.strip()]
            if not clean_texts:
                logger.warning("No valid texts after cleaning")
                return False
            
            # Add to documents
            start_idx = len(self.documents)
            self.documents.extend(clean_texts)
            
            # Add metadata
            if metadatas:
                self.metadata.extend(metadatas)
            else:
                for i, text in enumerate(clean_texts):
                    self.metadata.append({
                        'doc_id': start_idx + i,
                        'length': len(text),
                        'source': 'uploaded_document'
                    })
            
            # Create vectors based on model type
            if self.model_name.startswith("sentence-transformers"):
                # Use Sentence Transformers - encode all documents
                print(f"🔄 Encoding {len(self.documents)} documents with Sentence Transformers...")
                self.vectors = self.embedding_model.encode(self.documents)
            elif self.model_name == "openai":
                # Use OpenAI embeddings
                print(f"🔄 Encoding {len(self.documents)} documents with OpenAI...")
                embeddings = []
                for doc in self.documents:
                    response = self.embedding_model.embeddings.create(
                        input=doc,
                        model="text-embedding-ada-002"
                    )
                    embeddings.append(response.data[0].embedding)
                self.vectors = np.array(embeddings)
            
            self.is_ready = True
            
            logger.info(f"✅ Added {len(clean_texts)} documents. Total: {len(self.documents)}")
            
            # Persist the data
            self._persist()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding documents: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 8) -> List:
        """Search for similar documents, filtering out low-similarity results."""
        try:
            if not self.is_ready or not self.documents:
                logger.warning("Vector store not ready or no documents")
                return []
            if not query.strip():
                logger.warning("Empty query")
                return []
            
            # Create query vector based on model type
            if self.model_name.startswith("sentence-transformers"):
                # Use Sentence Transformers
                query_vector = self.embedding_model.encode([query.strip()])
                min_similarity = 0.3  # Higher threshold for neural embeddings
            elif self.model_name == "openai":
                # Use OpenAI embeddings
                response = self.embedding_model.embeddings.create(
                    input=query.strip(),
                    model="text-embedding-ada-002"
                )
                query_vector = np.array([response.data[0].embedding])
                min_similarity = 0.3  # Higher threshold for neural embeddings
            
            similarities = cosine_similarity(query_vector, self.vectors).flatten()
            # Get top k results
            top_indices = np.argsort(similarities)[::-1][:k]
            results = []
            for idx in top_indices:
                if idx < len(self.documents) and similarities[idx] >= min_similarity:
                    doc_text = self.documents[idx]
                    metadata = self.metadata[idx] if idx < len(self.metadata) else {}
                    metadata['similarity_score'] = float(similarities[idx])
                    doc = SimpleDocument(doc_text, metadata)
                    results.append(doc)
            logger.info(f"🔍 Found {len(results)} results for query (filtered by similarity)")
            return results
        except Exception as e:
            logger.error(f"❌ Error in similarity search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 3) -> List[Tuple]:
        """Search with scores."""
        try:
            if not self.is_ready or not self.documents:
                return []
            
            # Create query vector based on model type
            if self.model_name.startswith("sentence-transformers"):
                query_vector = self.embedding_model.encode([query.strip()])  
            elif self.model_name == "openai":
                response = self.embedding_model.embeddings.create(
                    input=query.strip(),
                    model="text-embedding-ada-002"
                )
                query_vector = np.array([response.data[0].embedding])
            
            similarities = cosine_similarity(query_vector, self.vectors).flatten()
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.documents):
                    doc_text = self.documents[idx]
                    score = float(similarities[idx])
                    metadata = self.metadata[idx] if idx < len(self.metadata) else {}
                    doc = SimpleDocument(doc_text, metadata)
                    results.append((doc, score))
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in similarity search with score: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        return {
            'status': 'ready' if self.is_ready else 'not_ready',
            'num_documents': len(self.documents),
            'embedding_dimension': self.vectors.shape[1] if self.vectors is not None else 0,
            'model_name': self.model_name
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            'num_documents': len(self.documents),
            'embedding_dimension': self.vectors.shape[1] if self.vectors is not None else 0,
            'is_ready': self.is_ready,
            'model_name': self.model_name,
            'persist_directory': str(self.persist_directory)
        }
    
    def clear(self):
        """Clear all data from the vector store."""
        try:
            self.documents = []
            self.vectors = None
            self.metadata = []
            self.is_ready = False
            
            # Clear persisted files
            data_file = self.persist_directory / "data.json"
            vectors_file = self.persist_directory / "vectors.pkl"
            
            for file_path in [data_file, vectors_file]:
                if file_path.exists():
                    file_path.unlink()
            
            logger.info("✅ Vector store cleared")
            
        except Exception as e:
            logger.error(f"❌ Error clearing vector store: {e}")
    
    def _persist(self):
        """Persist data to disk."""
        try:
            # Save documents and metadata
            data = {
                'documents': self.documents,
                'metadata': self.metadata,
                'is_ready': self.is_ready
            }
            
            data_file = self.persist_directory / "data.json"
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Save vectors
            if self.is_ready and self.vectors is not None:
                vectors_file = self.persist_directory / "vectors.pkl"
                
                with open(vectors_file, 'wb') as f:
                    pickle.dump(self.vectors, f)
            
            logger.info("✅ Vector store data persisted")
            
        except Exception as e:
            logger.error(f"❌ Error persisting vector store: {e}") 