#!/usr/bin/env python3
"""
Startup script for Streamlit deployment
Handles environment setup and launches the Streamlit app
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Set up environment variables for deployment."""
    # Disable tokenizer parallelism to avoid warnings
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    
    # Set up cache directory
    cache_dir = Path('./cache')
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault('TRANSFORMERS_CACHE', str(cache_dir))
    
    # Set up vector database directory
    vector_db_dir = Path('./enhanced_vector_db')
    vector_db_dir.mkdir(exist_ok=True)
    
    print("✅ Environment setup complete")

def main():
    """Main startup function."""
    print("🚀 Starting Universal PDF RAG System...")
    
    # Setup environment
    setup_environment()
    
    # Get port from environment (required for Heroku)
    port = os.environ.get('PORT', '8501')
    
    # Launch Streamlit app
    print(f"🌐 Launching Streamlit app on port {port}")
    
    # Build the streamlit command
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
        '--server.port', port,
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false'
    ]
    
    # Execute the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main() 