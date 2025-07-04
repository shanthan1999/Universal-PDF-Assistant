"""
Enhanced Universal PDF RAG System - Streamlit App
Advanced question answering system for any PDF documents with improved UI and features.
"""

import streamlit as st
import os
import sys
import logging
import tempfile
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Environment setup
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Enhanced session state initialization
def initialize_session_state():
    """Initialize comprehensive session state."""
    defaults = {
        'rag_system': None,
        'system_initialized': False,
        'documents_processed': False,
        'processed_files': [],
        'chat_history': [],
        'system_config': {
            'embedding_model': 'fast',  # Use sentence-transformers by default
            'llm_provider': 'qwen2.5',  # Use Qwen2.5 LLM for intelligent responses
            'confidence_threshold': 0.3
        },
        'processing_stats': {
            'total_documents': 0,
            'total_chunks': 0,
            'processing_time': 0
        },
        'performance_metrics': {
            'total_queries': 0,
            'avg_response_time': 0.0
        },
        'last_query': '',
        'query_count': 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def safe_import_rag():
    """Safely import RAG system with enhanced error handling."""
    try:
        from rag_system import AdvancedRAGSystem
        return AdvancedRAGSystem, None
    except ImportError as e:
        error_details = f"""
        Import Error: {e}
        
        **Troubleshooting Steps:**
        1. Ensure all required files are present in the directory
        2. Check if dependencies are installed: `pip install -r requirements.txt`
        3. Verify Python environment is correctly set up
        """
        return None, error_details
    except Exception as e:
        return None, f"Unexpected error during import: {e}"

@st.cache_resource
def initialize_rag_system(embedding_model: str = "fast", llm_provider: str = "qwen2.5"):
    """Initialize the RAG system with caching and better configuration."""
    try:
        RAGSystem, error = safe_import_rag()
        if error:
            return None, error
        
        if RAGSystem is None:
            return None, "Failed to import RAG system class"
        
        # Initialize with optimal configuration
        rag_system = RAGSystem(
            pdf_directory=".",
            vector_db_path="./enhanced_vector_db",
            embedding_model=embedding_model,  # Use selected embedding model
            llm_provider=llm_provider,
            llm_model="1.5B",  # Use smaller model for better compatibility
            quantized=True
        )
        
        logger.info(f"✅ RAG system initialized with {embedding_model} embeddings and {llm_provider} LLM")
        return rag_system, None
        
    except Exception as e:
        error_msg = f"Failed to initialize RAG system: {e}"
        logger.error(error_msg)
        return None, error_msg

def get_rag_system():
    """Get or initialize the RAG system."""
    try:
        # Check if we already have a RAG system in session state
        if st.session_state.rag_system is not None:
            return st.session_state.rag_system
        
        # Get system configuration
        config = st.session_state.system_config
        
        # Initialize RAG system
        rag_system, error = initialize_rag_system(
            embedding_model=config['embedding_model'],
            llm_provider=config['llm_provider']
        )
        
        if error:
            st.error(f"❌ Failed to initialize RAG system: {error}")
            return None
        
        if rag_system is None:
            st.error("❌ RAG system not available")
            return None
        
        # Store in session state
        st.session_state.rag_system = rag_system
        
        return rag_system
        
    except Exception as e:
        st.error(f"❌ Error getting RAG system: {e}")
        logger.error(f"RAG system error: {e}")
        return None

def process_uploaded_files(uploaded_files, progress_callback=None) -> Dict[str, Any]:
    """Enhanced file processing with detailed progress tracking."""
    try:
        # Get system configuration
        config = st.session_state.system_config
        rag_system, error = initialize_rag_system(
            embedding_model=config['embedding_model'],
            llm_provider=config['llm_provider']
        )
        
        if error:
            return {"success": False, "error": error, "files_processed": 0}
        
        if rag_system is None:
            return {"success": False, "error": "RAG system not initialized", "files_processed": 0}
        
        # Processing statistics
        stats = {
            "total_files": len(uploaded_files),
            "successful_files": 0,
            "failed_files": 0,
            "total_documents": 0,
            "processing_time": 0,
            "file_details": []
        }
        
        start_time = time.time()
        
        # Process each file with progress tracking
        for i, uploaded_file in enumerate(uploaded_files):
            if progress_callback:
                progress_callback(i / len(uploaded_files), f"Processing {uploaded_file.name}...")
            
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                file_start_time = time.time()
                
                try:
                    # Extract text
                    text = rag_system.pdf_extractor.extract_text_robust(tmp_file_path)
                    if not text or len(text.strip()) < 50:
                        stats["failed_files"] += 1
                        stats["file_details"].append({
                            "name": uploaded_file.name,
                            "status": "failed",
                            "reason": "Could not extract meaningful text",
                            "size": uploaded_file.size
                        })
                        continue
                    
                    # Process into documents
                    temp_doc = {uploaded_file.name: text}
                    rag_documents = rag_system.document_processor.create_rag_documents(temp_doc)
                    
                    if not rag_documents:
                        stats["failed_files"] += 1
                        stats["file_details"].append({
                            "name": uploaded_file.name,
                            "status": "failed",
                            "reason": "Could not process text into chunks",
                            "size": uploaded_file.size
                        })
                        continue
                    
                    # Add to vector store
                    success = rag_system.vector_store.create_vectorstore(rag_documents)
                    if success:
                        file_processing_time = time.time() - file_start_time
                        stats["successful_files"] += 1
                        stats["total_documents"] += len(rag_documents)
                        stats["file_details"].append({
                            "name": uploaded_file.name,
                            "status": "success",
                            "chunks": len(rag_documents),
                            "size": uploaded_file.size,
                            "processing_time": file_processing_time,
                            "text_length": len(text)
                        })
                        
                        # Add to processed files list
                        if uploaded_file.name not in st.session_state.processed_files:
                            st.session_state.processed_files.append(uploaded_file.name)
                    else:
                        stats["failed_files"] += 1
                        stats["file_details"].append({
                            "name": uploaded_file.name,
                            "status": "failed",
                            "reason": "Failed to add to vector store",
                            "size": uploaded_file.size
                        })
                        
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
                        
            except Exception as e:
                stats["failed_files"] += 1
                stats["file_details"].append({
                    "name": uploaded_file.name,
                    "status": "failed",
                    "reason": str(e),
                    "size": uploaded_file.size
                })
                logger.error(f"File processing error for {uploaded_file.name}: {e}")
        
        # Update progress to complete
        if progress_callback:
            progress_callback(1.0, "Processing complete!")
        
        # Calculate total processing time
        stats["processing_time"] = time.time() - start_time
        
        # Update system state
        if stats["successful_files"] > 0:
            rag_system.is_ready = True
            st.session_state.documents_processed = True
            st.session_state.processing_stats = {
                'total_documents': stats["total_documents"],
                'total_chunks': stats["total_documents"],  # Same as documents for simplicity
                'processing_time': stats["processing_time"]
            }
        
        return {
            "success": stats["successful_files"] > 0,
            "stats": stats,
            "files_processed": stats["successful_files"]
        }
        
    except Exception as e:
        logger.error(f"File processing error: {e}")
        return {
            "success": False,
            "error": f"Processing failed: {str(e)}",
            "files_processed": 0
        }

def ask_question(question: str) -> Dict[str, Any]:
    """Enhanced question processing with better error handling and caching."""
    try:
        # Get system configuration
        config = st.session_state.system_config
        rag_system, error = initialize_rag_system(
            embedding_model=config['embedding_model'],
            llm_provider=config['llm_provider']
        )
        
        if error:
            return {"error": error}
        
        if rag_system is None:
            return {"error": "RAG system is not initialized"}
        
        if not rag_system.is_ready or not st.session_state.documents_processed:
            return {"error": "No documents processed yet. Please upload PDF files first."}
        
        # Record query
        st.session_state.query_count += 1
        st.session_state.last_query = question
        
        # Get answer from RAG system
        start_time = time.time()
        result = rag_system.ask_question(question)
        query_time = time.time() - start_time
        
        # Add query time to result
        if result and "error" not in result:
            result["query_time"] = query_time
            result["timestamp"] = datetime.now().isoformat()
            
            # Add to chat history
            st.session_state.chat_history.append({
                "question": question,
                "answer": result.get("answer", ""),
                "timestamp": result["timestamp"],
                "query_time": query_time,
                "confidence": result.get("confidence", 0),
                "sources": len(result.get("sources", []))
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Question processing error: {e}")
        return {"error": f"Failed to process question: {str(e)}"}

def render_system_config():
    """Render system configuration panel."""
    with st.expander("⚙️ System Configuration", expanded=False):
        # Embedding model selection
        embedding_options = {
            "fast": "Fast (all-MiniLM-L6-v2) - Recommended",
            "quality": "High Quality (all-mpnet-base-v2)",
            "multilingual": "Multilingual Support",
            "qa": "Question-Answering Optimized",
            "openai": "OpenAI Embeddings (requires API key)"
        }
        
        current_embedding = st.session_state.system_config.get('embedding_model', 'fast')
        new_embedding = st.selectbox(
            "🧠 Embedding Model",
            options=list(embedding_options.keys()),
            index=list(embedding_options.keys()).index(current_embedding),
            format_func=lambda x: embedding_options[x],
            help="Choose the embedding model for document analysis"
        )
        
        # LLM provider selection
        llm_options = {
            "qwen2.5": "Qwen2.5 LLM (Advanced, high quality responses)"
        }
        
        current_llm = st.session_state.system_config.get('llm_provider', 'qwen2.5')
        new_llm = st.selectbox(
            "🤖 LLM Provider",
            options=list(llm_options.keys()),
            index=list(llm_options.keys()).index(current_llm),
            format_func=lambda x: llm_options[x],
            help="Choose the language model for answer generation"
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "🎯 Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=st.session_state.system_config.get('confidence_threshold', 0.3),
            step=0.1,
            help="Minimum confidence level for answers"
        )
        
        # Update configuration if changed
        if (new_embedding != current_embedding or 
            new_llm != current_llm or 
            confidence_threshold != st.session_state.system_config.get('confidence_threshold', 0.3)):
            
            st.session_state.system_config.update({
                'embedding_model': new_embedding,
                'llm_provider': new_llm,
                'confidence_threshold': confidence_threshold
            })
            
            # Clear cache if embedding model changed
            if new_embedding != current_embedding:
                st.cache_resource.clear()
                st.warning("⚠️ Configuration changed. System will reinitialize on next operation.")

def render_document_management():
    """Render document management interface."""
    st.subheader("📁 Document Management")
    
    # Show processed files
    if st.session_state.processed_files:
        st.write(f"**Processed Files ({len(st.session_state.processed_files)}):**")
        for i, filename in enumerate(st.session_state.processed_files, 1):
            st.write(f"{i}. {filename}")
    else:
        st.info("No documents processed yet.")
    
    # Processing statistics
    if st.session_state.processing_stats['total_documents'] > 0:
        stats = st.session_state.processing_stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", stats['total_documents'])
        with col2:
            st.metric("Total Chunks", stats['total_chunks'])
        with col3:
            st.metric("Processing Time", f"{stats['processing_time']:.1f}s")

def render_chat_interface():
    """Render the chat interface for asking questions."""
    st.subheader("💬 Chat with Your Documents")
    
    # Show example prompts
    with st.expander("💡 Example Questions & Requests", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📝 Summarization Requests:**")
            st.markdown("""
            - "Can you summarize this document?"
            - "What are the main points?"
            - "Give me an overview of the key concepts"
            - "What does this document say about AI?"
            - "Provide a brief summary of the content"
            """)
            
        with col2:
            st.markdown("**❓ Specific Questions:**")
            st.markdown("""
            - "What is deep learning?"
            - "How does machine learning work?"
            - "Explain neural networks"
            - "Why is AI important?"
            - "What are the applications of AI?"
            """)
    
    # Chat input
    if prompt := st.chat_input("Ask a question or request a summary..."):
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Generating response..."):
                try:
                    # Get RAG system
                    rag_system = get_rag_system()
                    if rag_system is None:
                        st.error("❌ RAG system not initialized. Please upload documents first.")
                        return
                    
                    # Generate response
                    start_time = time.time()
                    result = rag_system.ask_question(prompt)
                    end_time = time.time()
                    
                    if result.get('error'):
                        st.error(f"❌ Error: {result['error']}")
                        return
                    
                    # Display response with request type indicator
                    answer = result.get('answer', 'No answer generated')
                    request_type = result.get('request_type', 'question')
                    
                    # Add request type indicator
                    type_emoji = {
                        'summarization': '📋',
                        'explanation': '🎓', 
                        'question': '❓'
                    }
                    type_label = {
                        'summarization': 'Summary',
                        'explanation': 'Explanation',
                        'question': 'Answer'
                    }
                    
                    st.markdown(f"**{type_emoji.get(request_type, '💬')} {type_label.get(request_type, 'Response')}:**")
                    st.markdown(answer)
                    
                    # Show metadata
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        confidence = result.get('confidence', 0.0)
                        color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
                        st.metric("Confidence", f"{confidence:.1%}", delta=None)
                        st.markdown(f"{color}")
                    
                    with col2:
                        st.metric("Sources", len(result.get('sources', [])))
                    
                    with col3:
                        st.metric("Response Time", f"{end_time - start_time:.1f}s")
                    
                    with col4:
                        method = result.get('method', 'unknown')
                        st.metric("Method", method.replace('_', ' ').title())
                    
                    # Show sources if available
                    sources = result.get('sources', [])
                    if sources:
                        with st.expander(f"📚 View {len(sources)} Sources", expanded=False):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Source {i}:**")
                                st.markdown(f"```\n{source['content']}\n```")
                                if source.get('metadata'):
                                    st.caption(f"Metadata: {source['metadata']}")
                                st.divider()
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": {
                            "confidence": confidence,
                            "sources": len(sources),
                            "response_time": end_time - start_time,
                            "method": method,
                            "request_type": request_type
                        },
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # Update performance metrics
                    st.session_state.performance_metrics['total_queries'] += 1
                    st.session_state.performance_metrics['avg_response_time'] = (
                        (st.session_state.performance_metrics['avg_response_time'] * 
                         (st.session_state.performance_metrics['total_queries'] - 1) + 
                         (end_time - start_time)) / 
                        st.session_state.performance_metrics['total_queries']
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
                    logger.error(f"Chat error: {e}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.divider()
        st.subheader("📜 Chat History")
        
        for i, message in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10 messages
            with st.container():
                col1, col2 = st.columns([1, 6])
                with col1:
                    if message["role"] == "user":
                        st.markdown("**👤 You:**")
                    else:
                        st.markdown("**🤖 AI:**")
                    st.caption(message["timestamp"])
                
                with col2:
                    if message["role"] == "user":
                        st.markdown(f"*{message['content']}*")
                    else:
                        # Show request type for AI responses
                        if 'metadata' in message and 'request_type' in message['metadata']:
                            request_type = message['metadata']['request_type']
                            type_emoji = {'summarization': '📋', 'explanation': '🎓', 'question': '❓'}
                            st.markdown(f"{type_emoji.get(request_type, '💬')} {message['content'][:200]}...")
                        else:
                            st.markdown(f"{message['content'][:200]}...")
                        
                        # Show performance metrics
                        if 'metadata' in message:
                            meta = message['metadata']
                            st.caption(f"⚡ {meta.get('response_time', 0):.1f}s | "
                                     f"🎯 {meta.get('confidence', 0):.1%} | "
                                     f"📚 {meta.get('sources', 0)} sources")
                
                st.divider()

def main():
    """Enhanced main Streamlit application."""
    
    # Initialize session state
    initialize_session_state()
    
    # Page configuration
    st.set_page_config(
        page_title="Universal PDF RAG System",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced CSS
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    
    .stColumns > div {
        min-height: 300px;
    }
    
    .stTextArea textarea {
        min-height: 100px !important;
    }
    
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    
    .stats-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .chat-bubble {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with enhanced info
    st.title("📄 Universal PDF RAG System")
    st.markdown("**Advanced AI-Powered Document Analysis & Question Answering**")
    
    # Enhanced system info
    st.info("""
    🚀 **Enhanced Features:**
    - ✅ **Multiple PDF Processing** with progress tracking
    - ✅ **Advanced Embeddings** (Sentence Transformers + OpenAI support)
    - ✅ **Document Retrieval System** with confidence scoring
    - ✅ **Real-time Configuration** for optimal performance
    - ✅ **Chat History** and document management
    - ✅ **Performance Monitoring** and detailed statistics
    """)
    
    # Enhanced sidebar
    with st.sidebar:
        st.header("📊 System Dashboard")
        
        # System status
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.documents_processed:
                st.success("✅ Ready")
            else:
                st.warning("⚠️ No Docs")
        
        with col2:
            st.metric("Queries", st.session_state.query_count)
        
        # Quick stats
        if st.session_state.processing_stats['total_documents'] > 0:
            st.metric("Documents", st.session_state.processing_stats['total_documents'])
            st.metric("Files", len(st.session_state.processed_files))
        
        # System configuration
        render_system_config()
        
        # Controls
        st.header("🔧 Controls")
        
        if st.button("🔄 Reinitialize System"):
            st.cache_resource.clear()
            st.success("✅ System cache cleared!")
            st.rerun()
        
        if st.button("🗑️ Clear All Data"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            initialize_session_state()
            st.success("✅ All data cleared!")
            st.rerun()
        
        if st.button("📥 Export Chat History"):
            if st.session_state.chat_history:
                chat_export = json.dumps(st.session_state.chat_history, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=chat_export,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No chat history to export")
    
    # Main content with tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "❓ Ask Questions", "📊 Analytics"])
    
    with tab1:
        st.header("📄 Document Upload & Processing")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            help="Upload multiple PDF documents to analyze with the RAG system",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # File information
            total_size = sum(f.size for f in uploaded_files)
            st.write(f"**Selected Files:** {len(uploaded_files)} | **Total Size:** {total_size:,} bytes")
            
            # Show file list
            with st.expander("📋 View Selected Files", expanded=True):
                for i, file in enumerate(uploaded_files, 1):
                    st.write(f"**{i}.** {file.name} ({file.size:,} bytes)")
            
            # Processing button
            if st.button("🚀 Process Documents", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                
                with st.spinner("Processing documents..."):
                    result = process_uploaded_files(uploaded_files, progress_callback)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Show results
                if result["success"]:
                    st.success(f"✅ Successfully processed {result['files_processed']} files!")
                    
                    # Show detailed statistics
                    if "stats" in result:
                        stats = result["stats"]
                        st.markdown("### 📊 Processing Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Files", stats["total_files"])
                        with col2:
                            st.metric("Successful", stats["successful_files"])
                        with col3:
                            st.metric("Failed", stats["failed_files"])
                        with col4:
                            st.metric("Documents", stats["total_documents"])
                        
                        # Detailed file results
                        with st.expander("📝 Detailed Results", expanded=False):
                            for file_detail in stats["file_details"]:
                                if file_detail["status"] == "success":
                                    st.success(f"✅ {file_detail['name']}: {file_detail['chunks']} chunks")
                                else:
                                    st.error(f"❌ {file_detail['name']}: {file_detail['reason']}")
                    
                    st.rerun()
                else:
                    st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
        
        # Document management
        render_document_management()
    
    with tab2:
        st.header("❓ Ask Questions")
        
        if not st.session_state.documents_processed:
            st.warning("⚠️ Please upload and process documents first!")
            return
        
        # Chat interface
        render_chat_interface()
    
    with tab3:
        st.header("📊 System Analytics")
        
        # System metrics
        if st.session_state.query_count > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Queries", st.session_state.query_count)
            with col2:
                avg_confidence = sum(chat.get('confidence', 0) for chat in st.session_state.chat_history) / len(st.session_state.chat_history) if st.session_state.chat_history else 0
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            with col3:
                avg_time = sum(chat.get('query_time', 0) for chat in st.session_state.chat_history) / len(st.session_state.chat_history) if st.session_state.chat_history else 0
                st.metric("Avg Query Time", f"{avg_time:.2f}s")
        
        # Document statistics
        if st.session_state.processing_stats['total_documents'] > 0:
            st.markdown("### 📄 Document Statistics")
            stats = st.session_state.processing_stats
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Documents", stats['total_documents'])
                st.metric("Processed Files", len(st.session_state.processed_files))
            with col2:
                st.metric("Total Chunks", stats['total_chunks'])
                st.metric("Processing Time", f"{stats['processing_time']:.1f}s")
        
        # System configuration display
        st.markdown("### ⚙️ Current Configuration")
        config = st.session_state.system_config
        st.json(config)
        
        # System status
        st.markdown("### 🔧 System Status")
        status_info = {
            "System Initialized": st.session_state.system_initialized,
            "Documents Processed": st.session_state.documents_processed,
            "Embedding Model": config.get('embedding_model', 'Not set'),
            "LLM Provider": config.get('llm_provider', 'Not set'),
            "Confidence Threshold": config.get('confidence_threshold', 'Not set')
        }
        st.json(status_info)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Universal PDF RAG System v2.0** - Enhanced with Advanced Features
    
    🔗 **Workflow:**
    1. 📤 **Upload**: Add your PDF documents
    2. ⚙️ **Configure**: Choose optimal settings for your use case
    3. 🚀 **Process**: System creates embeddings and prepares for queries
    4. ❓ **Query**: Ask questions and get answers from your documents
    5. 📊 **Analyze**: Review performance and chat history
    
    ---
    **Ready for any PDF document type** 🚀 | **Powered by Sentence Transformers & Advanced AI**
    """)

if __name__ == "__main__":
    main()