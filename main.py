#!/usr/bin/env python3
"""
Universal PDF RAG System - Command Line Interface
Question answering system for any PDF documents.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add current directory to Python path for reliable imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def safe_import():
    """Safely import RAG system components with detailed error reporting."""
    try:
        from rag_system import AdvancedRAGSystem
        from pdf_extractor import PDFExtractor
        from document_processor import DocumentProcessor
        from response_refiner import ResponseRefiner
        return True, None
    except ImportError as e:
        error_msg = f"""
❌ IMPORT ERROR: {e}

TROUBLESHOOTING:
1. Ensure all required files are present:
   - rag_system.py
   - pdf_extractor.py
   - document_processor.py
   - response_refiner.py

2. Install dependencies:
   pip install -r requirements.txt

3. Check Python path and current directory
        """
        return False, error_msg
    except Exception as e:
        return False, f"Unexpected import error: {e}"

def setup_environment():
    """Set up environment variables for stable operation."""
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('TRANSFORMERS_CACHE', './cache')
    
    # Create cache directory if it doesn't exist
    cache_dir = Path('./cache')
    cache_dir.mkdir(exist_ok=True)

def setup_system(pdf_directory: str = ".", force_rebuild: bool = False) -> bool:
    """Set up the RAG system with enhanced error handling."""
    imports_ok, error_msg = safe_import()
    if not imports_ok:
        print(error_msg)
        return False
    
    # Import after successful check
    from rag_system import AdvancedRAGSystem
    
    try:
        print("\n" + "="*60)
        print("🚀 UNIVERSAL PDF RAG SYSTEM SETUP")
        print("="*60)
        
        # Validate PDF directory
        pdf_path = Path(pdf_directory)
        if not pdf_path.exists():
            print(f"❌ Directory not found: {pdf_directory}")
            return False
        
        pdf_files = list(pdf_path.glob("*.pdf"))
        if not pdf_files:
            print(f"❌ No PDF files found in {pdf_directory}")
            print("Please add PDF files to the directory and try again.")
            return False
        
        print(f"📄 Found {len(pdf_files)} PDF files:")
        for pdf_file in pdf_files:
            print(f"   - {pdf_file.name}")
        
        # Initialize and setup RAG system
        vector_db_path = pdf_path / "vector_db"
        rag_system = AdvancedRAGSystem(
            pdf_directory=str(pdf_path),
            vector_db_path=str(vector_db_path),
            embedding_model="tfidf",
            llm_provider="qwen2.5",  # Use Qwen2.5 by default
            llm_model="7B",  # Use 7B model by default
            quantized=True  # Use quantization for efficiency
        )
        
        print(f"\n🔧 Processing documents (force_rebuild={force_rebuild})...")
        success = rag_system.setup(force_rebuild=force_rebuild)
        
        if success:
            print("\n✅ Setup completed successfully!")
            print("\nYou can now:")
            print("  1. Use CLI: python main.py --query 'Your question here'")
            print("  2. Start web UI: streamlit run streamlit_app.py")
            print("  3. Run interactive mode: python main.py --interactive")
            return True
        else:
            print("\n❌ Setup failed. Check error messages above.")
            return False
            
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        print("\nTROUBLESHOOTING tips:")
        print("- Ensure sufficient memory (at least 2GB available)")
        print("- Check internet connection for model downloads")
        print("- Verify PDF files are readable and not corrupted")
        return False

def query_system(question: str, pdf_directory: str = ".", num_results: int = 3) -> bool:
    """Query the RAG system with error handling."""
    imports_ok, error_msg = safe_import()
    if not imports_ok:
        print(error_msg)
        return False
    
    from rag_system import AdvancedRAGSystem
    
    try:
        # Check if vector store exists
        vector_db_path = Path(pdf_directory) / "vector_db"
        if not vector_db_path.exists():
            print("❌ Vector store not found. Please run setup first:")
            print("   python main.py --setup")
            return False
        
        # Initialize RAG system
        rag_system = AdvancedRAGSystem(
            pdf_directory=pdf_directory,
            vector_db_path=str(vector_db_path),
            embedding_model="tfidf",
            llm_provider="qwen2.5",  # Use Qwen2.5 by default
            llm_model="7B",  # Use 7B model by default
            quantized=True  # Use quantization for efficiency
        )
        
        # Load existing vector store
        print("🔍 Loading vector store...")
        if not rag_system.is_ready:
            success = rag_system.setup(force_rebuild=False)
            if not success:
                print("❌ Failed to load vector store")
                return False
        
        print(f"\n💭 Question: {question}")
        print("-" * 60)
        
        # Get answer from RAG system
        result = rag_system.ask_question(question)
        
        if result and "answer" in result:
            print(f"📝 Answer:\n{result['answer']}")
            
            if result.get('sources'):
                print(f"\n📚 Sources:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"   {i}. {source.get('content', 'No content')[:200]}...")
        else:
            print("❌ No answer could be generated")
            print("Try rephrasing your question or using different keywords")
        
        return True
        
    except Exception as e:
        print(f"❌ Query error: {e}")
        return False

def interactive_mode(pdf_directory: str = "."):
    """Run interactive query mode with error handling."""
    imports_ok, error_msg = safe_import()
    if not imports_ok:
        print(error_msg)
        return
    
    from rag_system import AdvancedRAGSystem
    
    try:
        # Initialize system
        rag_system = AdvancedRAGSystem(
            pdf_directory=pdf_directory,
            vector_db_path=os.path.join(pdf_directory, "vector_db"),
            embedding_model="tfidf",
            llm_provider="qwen2.5",  # Use Qwen2.5 by default
            llm_model="7B",  # Use 7B model by default
            quantized=True  # Use quantization for efficiency
        )
        
        print("\n" + "="*60)
        print("🤖 UNIVERSAL PDF RAG SYSTEM - INTERACTIVE MODE")
        print("="*60)
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'help' for available commands")
        print("-" * 60)
        
        while True:
            try:
                question = input("\n💭 Enter your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if question.lower() == 'help':
                    print("\n📖 Available commands:")
                    print("  - Type any question to get an answer")
                    print("  - 'quit' or 'exit' to end session")
                    print("  - 'help' to show this message")
                    continue
                
                if not question:
                    print("❌ Please enter a question")
                    continue
                
                print("-" * 60)
                result = rag_system.ask_question(question)
                
                if result and "answer" in result:
                    print(f"📝 Answer:\n{result['answer']}")
                    
                    if result.get('sources'):
                        print(f"\n📚 Sources:")
                        for i, source in enumerate(result['sources'], 1):
                            print(f"   {i}. {source.get('content', 'No content')[:200]}...")
                else:
                    print("❌ No answer could be generated")
                    print("Try rephrasing your question")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
    except Exception as e:
        print(f"❌ Interactive mode error: {e}")

def get_system_info(pdf_directory: str = "."):
    """Get system information and status."""
    try:
        vector_db_path = Path(pdf_directory) / "vector_db"
        
        print("\n" + "="*60)
        print("📊 SYSTEM INFORMATION")
        print("="*60)
        
        # Check PDF files
        pdf_path = Path(pdf_directory)
        if pdf_path.exists():
            pdf_files = list(pdf_path.glob("*.pdf"))
            print(f"📄 PDF Files: {len(pdf_files)} found")
            for pdf_file in pdf_files:
                print(f"   - {pdf_file.name}")
        else:
            print("❌ PDF directory not found")
        
        # Check vector store
        if vector_db_path.exists():
            print(f"✅ Vector Store: {vector_db_path}")
            # Count files in vector store
            vector_files = list(vector_db_path.rglob("*"))
            print(f"   - {len(vector_files)} files in vector store")
        else:
            print("❌ Vector Store: Not found")
        
        print("\n🔧 System Status:")
        print("   - Ready for queries: Use 'python main.py --query'")
        print("   - Web UI: Use 'streamlit run streamlit_app.py'")
        print("   - Interactive: Use 'python main.py --interactive'")
        
    except Exception as e:
        print(f"❌ Error getting system info: {e}")

def main():
    """Main CLI interface."""
    
    # Handle special commands first
    if '--list-embeddings' in sys.argv:
        from embedding_selector import EmbeddingSelector
        EmbeddingSelector.list_models()
        return
    
    parser = argparse.ArgumentParser(
        description="PDF RAG System - Document Analysis and Question Answering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --setup                    # Setup the system
  python main.py --list-embeddings         # Show available embedding models
  python main.py --embedding-model quality # Use high-quality embeddings
  python main.py --query "What is AI?"      # Ask a question
  python main.py --interactive              # Start interactive mode
  python main.py --info                     # Show system information
        """
    )
    
    parser.add_argument(
        '--setup', 
        action='store_true',
        help='Setup the RAG system with PDF files'
    )
    
    parser.add_argument(
        '--query', 
        type=str,
        help='Ask a specific question'
    )
    
    parser.add_argument(
        '--interactive', 
        action='store_true',
        help='Start interactive query mode'
    )
    
    parser.add_argument(
        '--info', 
        action='store_true',
        help='Show system information'
    )
    
    parser.add_argument(
        '--pdf-dir', 
        type=str, 
        default='.',
        help='PDF directory path (default: current directory)'
    )
    
    parser.add_argument(
        '--force-rebuild', 
        action='store_true',
        help='Force rebuild vector store during setup'
    )
    
    # Embedding Model Options
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='fast',
        choices=['fast', 'quality', 'multilingual', 'qa', 'openai'],
        help='Embedding model to use (default: fast)'
    )
    
    parser.add_argument(
        '--list-embeddings',
        action='store_true',
        help='List all available embedding models'
    )
    
    # LLM Configuration Options  
    parser.add_argument(
        '--llm-provider',
        type=str,
        default='enhanced_simple',
        choices=['qwen2.5', 'enhanced_simple'],
        help='LLM provider to use (default: enhanced_simple)'
    )
    
    parser.add_argument(
        '--llm-model',
        type=str,
        default='7B',
        choices=['0.5B', '1.5B', '3B', '7B', '14B', '32B'],
        help='Qwen2.5 model size (default: 7B)'
    )
    
    parser.add_argument(
        '--no-quantization',
        action='store_true',
        help='Disable quantization (requires more GPU memory)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for model inference (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Handle commands
    if args.setup:
        setup_system(args.pdf_dir, args.force_rebuild)
    elif args.query:
        query_system(args.query, args.pdf_dir)
    elif args.interactive:
        interactive_mode(args.pdf_dir)
    elif args.info:
        get_system_info(args.pdf_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
