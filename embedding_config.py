"""
Embedding Configuration Utility
Easy switching between different embedding models
"""

import os
from typing import Dict, Any

class EmbeddingConfig:
    """Configuration manager for embedding models."""
    
    # Available embedding configurations
    EMBEDDING_OPTIONS = {
        "fast": {
            "name": "Sentence Transformers (Fast)",
            "model_name": "sentence-transformers",
            "model_id": "all-MiniLM-L6-v2",
            "dimension": 384,
            "speed": "⚡ Fast",
            "quality": "⭐⭐⭐⭐",
            "memory": "~500MB",
            "description": "Fast and efficient, good for most use cases",
            "use_case": "General purpose, quick setup"
        }
    }
    
    @classmethod
    def get_config(cls, embedding_type: str) -> Dict[str, Any]:
        """Get configuration for specified embedding type."""
        return cls.EMBEDDING_OPTIONS.get(embedding_type, cls.EMBEDDING_OPTIONS["fast"])
    
    @classmethod
    def list_available(cls) -> None:
        """Print all available embedding options."""
        print("\n🎯 Available Embedding Models:")
        print("=" * 80)
        
        for key, config in cls.EMBEDDING_OPTIONS.items():
            print(f"\n📊 {key.upper()}: {config['name']}")
            print(f"   Speed: {config['speed']} | Quality: {config['quality']} | Memory: {config['memory']}")
            print(f"   📝 {config['description']}")
            print(f"   🎯 Best for: {config['use_case']}")
    
    @classmethod
    def recommend_embedding(cls, use_case: str = "general") -> str:
        """Recommend embedding based on use case."""
        
        recommendations = {
            "general": "fast",
            "high_quality": "quality", 
            "multilingual": "multilingual",
            "qa": "qa",
            "research": "openai",
            "lightweight": "tfidf",
            "offline": "tfidf",
            "production": "quality"
        }
        
        recommended = recommendations.get(use_case.lower(), "fast")
        config = cls.get_config(recommended)
        
        print(f"\n💡 Recommended for '{use_case}': {recommended}")
        print(f"   {config['name']} - {config['description']}")
        
        return recommended
    
    @classmethod
    def validate_requirements(cls, embedding_type: str) -> bool:
        """Check if requirements are met for the embedding type."""
        config = cls.get_config(embedding_type)
        
        if config["model_name"] == "openai":
            # Check for OpenAI API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("❌ OpenAI embeddings require OPENAI_API_KEY environment variable")
                print("   Set with: export OPENAI_API_KEY='your-api-key'")
                return False
            print("✅ OpenAI API key found")
        
        elif config["model_name"] == "sentence-transformers":
            # Check if sentence-transformers is available
            try:
                import sentence_transformers
                print(f"✅ Sentence Transformers available")
            except ImportError:
                print("❌ sentence-transformers not installed")
                print("   Install with: pip install sentence-transformers")
                return False
        
        elif config["model_name"] == "tfidf":
            # TF-IDF has no special requirements
            print("✅ TF-IDF ready (no additional requirements)")
        
        return True
    
    @classmethod
    def setup_embedding(cls, embedding_type: str, force: bool = False) -> bool:
        """Set up and validate an embedding model."""
        
        print(f"\n🔧 Setting up '{embedding_type}' embeddings...")
        
        # Get configuration
        config = cls.get_config(embedding_type)
        print(f"📊 Model: {config['name']}")
        print(f"📏 Dimension: {config['dimension']}")
        print(f"💾 Memory: {config['memory']}")
        
        # Validate requirements
        if not cls.validate_requirements(embedding_type):
            if not force:
                print("\n⚠️ Requirements not met. Use --force to ignore")
                return False
            else:
                print("\n⚠️ Continuing despite missing requirements (--force enabled)")
        
        print(f"\n✅ '{embedding_type}' embeddings ready!")
        return True


def interactive_embedding_selector():
    """Interactive tool to help users select the best embedding model."""
    
    print("\n🎯 Embedding Model Selector")
    print("=" * 50)
    
    # Ask about use case
    print("\nWhat's your primary use case?")
    print("1. General document analysis")
    print("2. High-quality research") 
    print("3. Multi-language documents")
    print("4. Question-answering system")
    print("5. Lightweight/offline system")
    print("6. Production deployment")
    
    try:
        choice = input("\nEnter choice (1-6): ").strip()
        
        use_case_map = {
            "1": "general",
            "2": "high_quality",
            "3": "multilingual", 
            "4": "qa",
            "5": "lightweight",
            "6": "production"
        }
        
        use_case = use_case_map.get(choice, "general")
        recommended = EmbeddingConfig.recommend_embedding(use_case)
        
        # Ask about constraints
        print(f"\nConstraints check for '{recommended}':")
        config = EmbeddingConfig.get_config(recommended)
        
        print(f"📊 This will use: {config['name']}")
        print(f"💾 Memory required: {config['memory']}")
        print(f"⚡ Speed: {config['speed']}")
        
        if config["model_name"] == "openai":
            print("🔑 Requires OpenAI API key")
        
        confirm = input(f"\nUse '{recommended}' embeddings? (y/n): ").strip().lower()
        
        if confirm == 'y':
            if EmbeddingConfig.setup_embedding(recommended):
                print(f"\n🎉 Ready to use '{recommended}' embeddings!")
                print(f"💡 Start your system with: --embedding-model {recommended}")
                return recommended
        else:
            print("\n📋 All available options:")
            EmbeddingConfig.list_available()
            
    except KeyboardInterrupt:
        print("\n👋 Selection cancelled")
        return None
    
    return None


if __name__ == "__main__":
    # Test the embedding configuration
    print("🧪 Testing Embedding Configuration...")
    
    # List all options
    EmbeddingConfig.list_available()
    
    # Test recommendations
    for use_case in ["general", "high_quality", "multilingual", "qa", "lightweight"]:
        EmbeddingConfig.recommend_embedding(use_case)
    
    # Interactive selector
    print("\n" + "=" * 60)
    interactive_embedding_selector() 