"""
Simple Embedding Model Selector
Easy switching between different embedding models for the RAG system
"""

import os
import sys
from pathlib import Path

class EmbeddingSelector:
    """Simple utility to select and configure embedding models."""
    
    AVAILABLE_MODELS = {
        "fast": {
            "name": "Sentence Transformers (Fast)",
            "description": "Modern neural embeddings, good balance of speed and quality",
            "requirements": "sentence-transformers package",
            "memory": "~500MB",
            "speed": "⚡ Fast",
            "quality": "⭐⭐⭐⭐ Very Good",
            "recommended_for": "Most general use cases, default choice"
        },
        
        "quality": {
            "name": "Sentence Transformers (High Quality)",
            "description": "Best quality neural embeddings, slower processing",
            "requirements": "sentence-transformers package",
            "memory": "~1GB",
            "speed": "🔄 Medium",
            "quality": "⭐⭐⭐⭐⭐ Excellent",
            "recommended_for": "When accuracy is most important"
        },
        
        "multilingual": {
            "name": "Multilingual Embeddings",
            "description": "Supports multiple languages",
            "requirements": "sentence-transformers package",
            "memory": "~600MB",
            "speed": "🔄 Medium",
            "quality": "⭐⭐⭐⭐ Very Good",
            "recommended_for": "Multi-language documents"
        },
        
        "qa": {
            "name": "Question-Answer Optimized",
            "description": "Optimized for question-answering tasks",
            "requirements": "sentence-transformers package",
            "memory": "~500MB",
            "speed": "⚡ Fast",
            "quality": "⭐⭐⭐⭐ Very Good",
            "recommended_for": "Q&A systems, RAG applications"
        },
        
        "openai": {
            "name": "OpenAI Embeddings",
            "description": "OpenAI's state-of-the-art embeddings (requires API key)",
            "requirements": "openai package + API key",
            "memory": "No local memory",
            "speed": "🌐 API",
            "quality": "⭐⭐⭐⭐⭐ Excellent",
            "recommended_for": "Highest quality, requires internet & API key"
        }
    }
    
    @classmethod
    def list_models(cls):
        """List all available embedding models."""
        print("\n🎯 Available Embedding Models:")
        print("=" * 80)
        
        for key, info in cls.AVAILABLE_MODELS.items():
            print(f"\n📊 {key.upper()}")
            print(f"   Name: {info['name']}")
            print(f"   Speed: {info['speed']} | Quality: {info['quality']} | Memory: {info['memory']}")
            print(f"   📝 {info['description']}")
            print(f"   🎯 Best for: {info['recommended_for']}")
            print(f"   📋 Requirements: {info['requirements']}")
    
    @classmethod
    def check_requirements(cls, model_key: str) -> bool:
        """Check if requirements are met for the specified model."""
        if model_key not in cls.AVAILABLE_MODELS:
            print(f"❌ Unknown model: {model_key}")
            return False
        
        info = cls.AVAILABLE_MODELS[model_key]
        
        if model_key in ["fast", "quality", "multilingual", "qa"]:
            try:
                import sentence_transformers
                print("✅ sentence-transformers available")
                return True
            except ImportError:
                print("❌ sentence-transformers not installed")
                print("   Install with: pip install sentence-transformers")
                return False
        
        elif model_key == "openai":
            try:
                import openai
                import os
                if os.getenv("OPENAI_API_KEY"):
                    print("✅ OpenAI package and API key available")
                    return True
                else:
                    print("❌ OPENAI_API_KEY not found in environment variables")
                    print("   Set with: export OPENAI_API_KEY='your-api-key'")
                    return False
            except ImportError:
                print("❌ openai package not installed")
                print("   Install with: pip install openai")
                return False
        
        return True
    
    @classmethod
    def recommend_model(cls, use_case: str = "general") -> str:
        """Recommend a model based on use case."""
        recommendations = {
            "general": "fast",
            "speed": "fast",
            "quality": "quality",
            "lightweight": "fast",
            "testing": "fast",
            "production": "quality",
            "multilingual": "multilingual",
            "qa": "qa"
        }
        
        recommended = recommendations.get(use_case.lower(), "fast")
        
        print(f"\n💡 Recommended for '{use_case}': {recommended}")
        info = cls.AVAILABLE_MODELS[recommended]
        print(f"   {info['name']} - {info['description']}")
        
        return recommended
    
    @classmethod
    def test_model(cls, model_key: str) -> bool:
        """Test if a model can be loaded successfully."""
        print(f"\n🧪 Testing {model_key}...")
        
        if not cls.check_requirements(model_key):
            return False
        
        try:
            # Test the model by creating a simple vector store
            from vector_store import AdvancedVectorStore
            
            # Map user-friendly names to internal model names
            embedding_map = {
                "fast": "sentence-transformers",
                "quality": "sentence-transformers-quality", 
                "multilingual": "sentence-transformers-multilingual",
                "qa": "sentence-transformers-qa",
                "openai": "openai"
            }
            
            internal_model_name = embedding_map.get(model_key, "sentence-transformers")
            
            test_store = AdvancedVectorStore(
                persist_directory="./test_embedding",
                model_name=internal_model_name
            )
            
            # Test with sample text (longer to avoid TF-IDF issues)
            test_texts = [
                "This is a comprehensive test document that contains enough text to work with TF-IDF vectorization systems.",
                "Another detailed test document with sufficient content for proper embedding generation and similarity testing."
            ]
            success = test_store.add_documents(test_texts)
            
            if success:
                # Test search
                results = test_store.similarity_search("test", k=1)
                if results:
                    print(f"✅ {model_key} test successful!")
                    
                    # Clean up test files
                    test_store.clear()
                    return True
            
            print(f"❌ {model_key} test failed")
            return False
            
        except Exception as e:
            print(f"❌ {model_key} test error: {e}")
            return False
    
    @classmethod
    def interactive_selector(cls):
        """Interactive model selection."""
        print("\n🎯 Embedding Model Selector")
        print("=" * 50)
        
        # Show available models
        cls.list_models()
        
        # Get user preference
        print(f"\nAvailable models: {list(cls.AVAILABLE_MODELS.keys())}")
        
        while True:
            try:
                choice = input("\nEnter model name (or 'quit' to exit): ").strip().lower()
                
                if choice in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    return None
                
                if choice in cls.AVAILABLE_MODELS:
                    print(f"\n🔍 Selected: {choice}")
                    
                    # Check requirements
                    if cls.check_requirements(choice):
                        # Test the model
                        if cls.test_model(choice):
                            print(f"\n🎉 {choice} is ready to use!")
                            print(f"💡 Update your system with: --embedding-model {choice}")
                            return choice
                        else:
                            print(f"❌ {choice} failed testing")
                    else:
                        print(f"❌ {choice} requirements not met")
                
                else:
                    print(f"❌ Unknown model: {choice}")
                    print(f"Available: {list(cls.AVAILABLE_MODELS.keys())}")
            
            except KeyboardInterrupt:
                print("\n👋 Selection cancelled")
                return None
    
    @classmethod
    def get_model_config(cls, model_key: str) -> dict:
        """Get configuration for a model."""
        return cls.AVAILABLE_MODELS.get(model_key, cls.AVAILABLE_MODELS["fast"])


def main():
    """Main function for the embedding selector."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "list":
            EmbeddingSelector.list_models()
        
        elif command == "test":
            if len(sys.argv) > 2:
                model = sys.argv[2]
                EmbeddingSelector.test_model(model)
            else:
                print("Usage: python embedding_selector.py test <model_name>")
        
        elif command == "recommend":
            use_case = sys.argv[2] if len(sys.argv) > 2 else "general"
            EmbeddingSelector.recommend_model(use_case)
        
        elif command == "interactive":
            EmbeddingSelector.interactive_selector()
        
        else:
            print("Unknown command. Available: list, test, recommend, interactive")
    
    else:
        # Default to interactive mode
        EmbeddingSelector.interactive_selector()


if __name__ == "__main__":
    main() 