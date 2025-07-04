"""
Enhanced RAG System with RetrievalQA and Qwen2.5 LLM
Works with Python 3.13 and includes advanced text generation
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from pdf_extractor import PDFExtractor
from document_processor import DocumentProcessor
from vector_store import AdvancedVectorStore

logger = logging.getLogger(__name__)

class AdvancedRAGSystem:
    """Enhanced RAG system with RetrievalQA and Qwen2.5 LLM support."""
    
    def __init__(self,
                 pdf_directory: str = ".",
                 vector_db_path: str = "./enhanced_vector_db",
                 embedding_model: str = "fast",  # Options: fast, quality, multilingual, qa, openai, tfidf
                 llm_provider: str = "qwen2.5",  # Always use Qwen2.5 LLM
                 llm_model: str = "1.5B",  # Use smaller model by default
                 use_streaming: bool = False,
                 quantized: bool = True):
        """Initialize the RAG system."""
        
        self.pdf_directory = pdf_directory
        self.vector_db_path = vector_db_path
        self.embedding_model = embedding_model
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.use_streaming = use_streaming
        self.quantized = quantized
        
        # Initialize components
        self.pdf_extractor = PDFExtractor()
        self.document_processor = DocumentProcessor()
        
        # Map user-friendly names to internal model names
        embedding_map = {
            "fast": "sentence-transformers",
            "quality": "sentence-transformers-quality", 
            "multilingual": "sentence-transformers-multilingual",
            "qa": "sentence-transformers-qa",
            "openai": "openai"
        }
        
        model_name = embedding_map.get(embedding_model, "sentence-transformers")
        
        self.vector_store = AdvancedVectorStore(
            persist_directory=vector_db_path,
            model_name=model_name
        )
        
        # LLM components
        self.llm_manager = None
        self.retrieval_qa = None
        
        # Fallback components (for backward compatibility)
        self.llm_pipeline = None
        self.tokenizer = None
        
        # Initialize LLM
        self._initialize_llm()
        
        # System state
        self.is_ready = False
        self.documents_loaded = False
        self.stats = {}
        
        logger.info("✅ Enhanced RAG system initialized")
    
    def _initialize_llm(self):
        """Initialize the LLM based on provider."""
        try:
            if self.llm_provider == "qwen2.5":
                print("🤖 Initializing Qwen2.5 LLM...")
                self._initialize_qwen2_llm()
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}. Only 'qwen2.5' is supported.")
            self.llm_pipeline = None
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            print(f"❌ LLM initialization failed: {e}")
            raise e
    
    def _initialize_qwen2_llm(self):
        """Initialize Qwen2.5 LLM with RetrievalQA."""
        try:
            from llm_manager import load_qwen2_model
            
            print(f"🔄 Loading Qwen2.5-{self.llm_model} model...")
            
            # Load LLM manager
            self.llm_manager = load_qwen2_model(
                model_size=self.llm_model,
                quantized=self.quantized
            )
            
            # Load the model
            if self.llm_manager.load_model():
                print("✅ Qwen2.5 model loaded successfully!")
                
                # Create pipeline
                if self.llm_manager.create_pipeline():
                    print("✅ LangChain pipeline created!")
                    return True
                else:
                    raise Exception("Failed to create LangChain pipeline")
            else:
                raise Exception("Failed to load Qwen2.5 model")
                
        except Exception as e:
            logger.error(f"Qwen2.5 initialization failed: {e}")
            raise e
    
    def setup(self, force_rebuild: bool = False) -> bool:
        """Set up the RAG system."""
        print("\n" + "="*60)
        print("🚀 SETTING UP UNIVERSAL RAG SYSTEM WITH RETRIEVALQA")
        print("="*60)
        
        try:
            # Check if vector store already exists and has data
            vector_info = self.vector_store.get_collection_info()
            has_existing_data = vector_info.get('num_documents', 0) > 0
            
            if has_existing_data and not force_rebuild:
                print(f"✅ Found existing vector store with {vector_info['num_documents']} documents")
                print("🔄 Skipping PDF processing, using existing data...")
                
                # Create dummy stats for existing data
                self.stats = {
                    'pdf_files': 'existing',
                    'total_chunks': vector_info['num_documents'],
                    'chunk_stats': {'avg_chunk_length': 'N/A'},
                    'vector_info': vector_info,
                    'llm_provider': self.llm_provider,
                    'llm_model': self.llm_model,
                    'has_retrieval_qa': False
                }
                
                # Mark system as ready
                self.is_ready = True
                self.documents_loaded = True
                
            else:
                # Step 1: Extract text from PDFs
                print("\n📄 Step 1: Extracting text from PDFs...")
                documents_text = self.pdf_extractor.extract_from_directory(self.pdf_directory)
                
                if not documents_text:
                    print("❌ No documents extracted. Setup failed.")
                    return False
                
                # Step 2: Process documents into chunks
                print("\n🔧 Step 2: Processing documents into chunks...")
                rag_documents = self.document_processor.create_rag_documents(documents_text)
                
                if not rag_documents:
                    print("❌ No document chunks created. Setup failed.")
                    return False
                
                # Step 3: Create vector store
                print("\n🧠 Step 3: Creating vector store...")
                success = self.vector_store.create_vectorstore(rag_documents, force_recreate=force_rebuild)
                
                if not success:
                    print("❌ Vector store creation failed. Setup failed.")
                    return False
                
                # Store statistics
                self.stats = {
                    'pdf_files': len(documents_text),
                    'total_chunks': len(rag_documents),
                    'chunk_stats': self.document_processor.get_chunk_statistics(rag_documents),
                    'vector_info': self.vector_store.get_collection_info(),
                    'llm_provider': self.llm_provider,
                    'llm_model': self.llm_model,
                    'has_retrieval_qa': False
                }
                
                # Mark system as ready
                self.is_ready = True
                self.documents_loaded = True
            
            # Step 4: Create RetrievalQA chain (if Qwen2.5 is available)
            if self.llm_provider == "qwen2.5" and self.llm_manager:
                print("\n🔗 Step 4: Creating RetrievalQA chain...")
                try:
                    # Create retriever from vector store
                    retriever = self._create_retriever()
                    
                    # Create RetrievalQA chain
                    self.retrieval_qa = self.llm_manager.create_retrieval_qa(retriever)
                    
                    if self.retrieval_qa:
                        print("✅ RetrievalQA chain created successfully!")
                    else:
                        print("⚠️ RetrievalQA creation failed, but continuing with manual Qwen2.5")
                        # Don't change provider - keep using Qwen2.5 manually
                        
                except Exception as e:
                    print(f"⚠️ RetrievalQA setup failed: {e}")
                    print("Continuing with manual Qwen2.5 generation")
                    # Don't change provider - keep using Qwen2.5 manually
            
            # Update system state
            self.is_ready = True
            self.documents_loaded = True
            
            # Update stats with RetrievalQA info
            self.stats['has_retrieval_qa'] = self.retrieval_qa is not None
            
            print("\n" + "="*60)
            print("✅ UNIVERSAL RAG SYSTEM SETUP COMPLETE")
            print("="*60)
            self._print_setup_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ Setup failed with error: {e}")
            logger.error(f"Setup error: {e}")
            return False
    
    def _create_retriever(self):
        """Create a retriever from the vector store for RetrievalQA."""
        try:
            # Create a custom retriever class
            class VectorStoreRetriever:
                def __init__(self, vector_store):
                    self.vector_store = vector_store
                    
                def get_relevant_documents(self, query: str) -> List:
                    return self.vector_store.similarity_search(query, k=5)
                
                def __call__(self, query: str) -> List:
                    return self.get_relevant_documents(query)
            
            return VectorStoreRetriever(self.vector_store)
            
        except Exception as e:
            logger.error(f"Failed to create retriever: {e}")
            return None
    
    def _print_setup_summary(self):
        """Print a summary of the setup process."""
        print(f"📊 SETUP SUMMARY:")
        print(f"   📄 PDF files processed: {self.stats['pdf_files']}")
        print(f"   📋 Document chunks created: {self.stats['total_chunks']}")
        print(f"   🧠 Vector store status: {self.stats['vector_info']['status']}")
        avg_length = self.stats['chunk_stats']['avg_chunk_length']
        if isinstance(avg_length, (int, float)):
            print(f"   📏 Average chunk length: {avg_length:.1f} characters")
        else:
            print(f"   📏 Average chunk length: {avg_length} characters")
        print(f"   🤖 LLM provider: {self.stats['llm_provider']}")
        print(f"   🎯 Embedding model: {self.embedding_model}")
        
        if self.llm_provider == "qwen2.5":
            print(f"   🔗 RetrievalQA: {'✅ Ready' if self.retrieval_qa else '❌ Not available'}")
            if self.llm_manager:
                model_info = self.llm_manager.get_model_info()
                print(f"   💾 Model: {model_info['model_name']}")
                print(f"   🖥️ Device: {model_info['device']}")
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question using the RAG system with LLM."""
        if not self.is_ready:
            return {
                "error": "RAG system not ready. Please process documents first.",
                "answer": "",
                "sources": [],
                "confidence": 0.0
            }
        
        try:
            print(f"\n🔍 Processing question with {self.llm_provider}: '{question}'")
            
            # Use Qwen2.5 LLM
            if self.llm_provider == "qwen2.5":
                if self.llm_manager and hasattr(self.llm_manager, 'is_ready') and self.llm_manager.is_ready:
                    # Try RetrievalQA first, then manual if needed
                    if self.retrieval_qa:
                        return self._ask_with_retrieval_qa(question)
                    else:
                        return self._ask_with_manual_qwen(question)
                else:
                    raise RuntimeError("Qwen2.5 LLM is not ready. Please ensure the model loaded correctly.")
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
                
        except Exception as e:
            logger.error(f"Error in ask_question: {e}")
            return {
                "error": f"Failed to generate answer: {str(e)}",
                "answer": "",
                "sources": [],
                "confidence": 0.0
            }
    
    def _ask_with_retrieval_qa(self, question: str) -> Dict[str, Any]:
        """Ask question using RetrievalQA chain."""
        try:
            print("🔗 Using RetrievalQA with Qwen2.5...")
            
            # Check if we have a proper RetrievalQA chain or direct callable
            if hasattr(self.retrieval_qa, '__call__') and not hasattr(self.retrieval_qa, 'run'):
                # It's our direct callable, use manual retrieval + generation
                return self._ask_with_manual_qwen(question)
            
            # Query the RetrievalQA chain
            result = self.retrieval_qa({"query": question})
            
            answer = result.get("result", "No answer generated")
            source_documents = result.get("source_documents", [])
            
            # Process source documents
            sources = []
            for doc in source_documents[:3]:  # Limit to top 3 sources
                source_info = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source_info)
            
            # Refine answer if possible
            try:
                from response_refiner import ResponseRefiner
                refiner = ResponseRefiner(use_summarizer=False)
                answer = refiner.refine_response(question, answer, target_length="medium")
            except Exception as e:
                print(f"⚠️ Response refinement failed: {e}")
            
            print(f"✅ Generated answer using {len(sources)} sources with RetrievalQA")
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": 0.9,  # Higher confidence with advanced LLM
                "num_sources": len(sources),
                "llm_model": f"Qwen2.5-{self.llm_model}",
                "method": "RetrievalQA"
            }
            
        except Exception as e:
            logger.error(f"RetrievalQA failed: {e}")
            print(f"⚠️ RetrievalQA failed, switching to manual Qwen2.5: {e}")
            return self._ask_with_manual_qwen(question)
    

    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        status = {
            'ready': self.is_ready,
            'documents_loaded': self.documents_loaded,
            'llm_provider': self.llm_provider,
            'llm_model': self.llm_model,
            'embedding_model': self.embedding_model,
            'vector_store_stats': self.vector_store.get_stats(),
            'pdf_directory': self.pdf_directory,
            'vector_db_path': self.vector_db_path,
            'has_retrieval_qa': self.retrieval_qa is not None
        }
        
        # Add LLM manager info if available
        if self.llm_manager:
            status['llm_info'] = self.llm_manager.get_model_info()
        
        return status
    
    def _ask_with_manual_qwen(self, question: str) -> Dict[str, Any]:
        """Ask question using manual retrieval + Qwen2.5 generation."""
        try:
            print("🔍 Using manual retrieval + Qwen2.5 generation...")
            
            # Step 1: Analyze the type of request
            question_lower = question.lower()
            request_type = self._analyze_request_type(question_lower)
            
            # Step 2: Retrieve relevant documents (adjust k based on request type)
            k_docs = 8 if request_type == "summarization" else 5
            relevant_docs = self.vector_store.similarity_search(question, k=k_docs)
            
            if not relevant_docs:
                return {
                    "answer": "I couldn't find relevant information to answer your question.",
                    "sources": [],
                    "confidence": 0.0,
                    "method": "qwen2.5_manual"
                }
            
            # Step 3: Prepare context based on request type
            context_texts = [doc.page_content for doc in relevant_docs]
            if request_type == "summarization":
                context = "\n\n".join(context_texts[:6])  # More context for summaries
            else:
                context = "\n\n".join(context_texts[:3])  # Standard context
            
            # Step 4: Create specialized prompt based on request type
            prompt = self._create_specialized_prompt(question, context, request_type)
            
            # Step 5: Generate response using Qwen2.5
            if self.llm_manager and self.llm_manager.is_ready:
                answer = self.llm_manager.generate_response(prompt, max_tokens=1024)
            else:
                return {
                    "error": "Qwen2.5 model not available for generation",
                    "answer": "",
                    "sources": [],
                    "confidence": 0.0,
                    "method": "qwen2.5_manual_error"
                }
            
            # Step 6: Process sources
            sources = []
            for i, doc in enumerate(relevant_docs[:5]):
                source_info = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata,
                    "similarity": getattr(doc, 'similarity', 0.8)
                }
                sources.append(source_info)
            
            # Step 7: Calculate confidence based on request type
            confidence = 0.9 if request_type == "summarization" else 0.85
            
            print(f"✅ Generated {request_type} answer using {len(sources)} sources with Qwen2.5")
            
            return {
                "answer": answer.strip(),
                "sources": sources,
                "confidence": confidence,
                "num_sources": len(sources),
                "llm_model": f"Qwen2.5-{self.llm_model}",
                "method": "qwen2.5_manual",
                "request_type": request_type
            }
            
        except Exception as e:
            logger.error(f"Error in manual Qwen generation: {e}")
            print(f"⚠️ Qwen2.5 manual generation failed: {e}")
            return {
                "error": f"Qwen2.5 generation failed: {str(e)}",
                "answer": "",
                "sources": [],
                "confidence": 0.0,
                "method": "qwen2.5_manual_error"
            }

    def _analyze_request_type(self, question_lower: str) -> str:
        """Analyze the type of request to customize the response."""
        
        # Summarization keywords
        summarization_keywords = [
            "summarize", "summary", "main points", "key points", "overview", 
            "brief", "outline", "gist", "essence", "recap", "highlights",
            "what does this document say", "what is this about", "main ideas"
        ]
        
        # Explanation keywords  
        explanation_keywords = [
            "explain", "how does", "how do", "what is", "what are", "define",
            "definition", "meaning", "concept", "understand", "clarify"
        ]
        
        # Specific question keywords
        question_keywords = [
            "why", "when", "where", "who", "which", "can you tell me",
            "details about", "information about", "examples of"
        ]
        
        # Check for summarization requests
        if any(keyword in question_lower for keyword in summarization_keywords):
            return "summarization"
        
        # Check for explanation requests
        elif any(keyword in question_lower for keyword in explanation_keywords):
            return "explanation"
        
        # Default to specific question
        else:
            return "question"
    
    def _create_specialized_prompt(self, question: str, context: str, request_type: str) -> str:
        """Create specialized prompts based on request type."""
        
        base_instruction = "You are an expert AI assistant helping users understand documents. "
        
        if request_type == "summarization":
            return f"""{base_instruction}Your task is to provide comprehensive summaries of document content.

Guidelines for summarization:
1. Create a well-structured summary that captures the main ideas
2. Organize information logically with clear sections
3. Include key concepts, important details, and conclusions
4. Make it comprehensive but concise
5. Use bullet points or numbered lists when appropriate

Document content:
{context}

User request: {question}

Please provide a comprehensive summary:"""

        elif request_type == "explanation":
            return f"""{base_instruction}Your task is to provide clear, educational explanations of concepts.

Guidelines for explanations:
1. Start with a clear definition or introduction
2. Break down complex concepts into understandable parts
3. Use examples and analogies when helpful
4. Explain the significance and applications
5. Make it accessible to someone learning the topic

Context from documents:
{context}

Question: {question}

Please provide a clear explanation:"""

        else:  # question type
            return f"""{base_instruction}Your task is to provide accurate, detailed answers to specific questions.

Guidelines for answers:
1. Address the question directly and completely
2. Provide specific details and examples from the documents
3. Include relevant context and background information
4. Cite specific information when possible
5. Be thorough but focused on the question asked

Context from documents:
{context}

Question: {question}

Please provide a detailed answer:"""

    def cleanup(self):
        """Clean up system resources."""
        try:
            if self.llm_manager:
                self.llm_manager.cleanup()
            
            print("🧹 RAG system resources cleaned up")
            
        except Exception as e:
            logger.error(f"⚠️ Cleanup failed: {e}") 