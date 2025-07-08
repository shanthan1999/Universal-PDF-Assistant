"""
LLM Manager for Qwen2.5 Integration
Handles model loading, inference, and optimization
"""

import logging
import torch
from typing import Optional, Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
try:
    from langchain_huggingface import HuggingFacePipeline
except ImportError:
    from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

class Qwen2LLMManager:
    """Manage Qwen2.5 LLM for RAG applications."""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-1]0.5B-Instruct",  # Use smaller model by default
                 device: str = "auto",
                 load_in_4bit: bool = True,
                 max_memory: Optional[Dict] = None):
        """Initialize the Qwen2.5 LLM manager."""
        
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_memory = max_memory
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.langchain_llm = None
        
        # Configuration
        self.generation_config = None
        self.is_ready = False
        
        logger.info(f"🤖 Initializing Qwen2.5 LLM Manager with model: {model_name}")
    
    def load_model(self) -> bool:
        """Load the Qwen2.5 model with optimizations."""
        try:
            print("🔄 Loading Qwen2.5 model...")
            
            # Configure device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            print(f"📱 Using device: {self.device}")
            
            # Load tokenizer
            print("📝 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Configure model loading parameters
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            # Add quantization if requested and CUDA available
            if self.load_in_4bit and torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    print("⚡ Using 4-bit quantization for efficiency")
                except ImportError:
                    print("⚠️ BitsAndBytes not available, using FP16")
                    self.load_in_4bit = False
            
            # Add memory management
            if self.max_memory:
                model_kwargs["max_memory"] = self.max_memory
            
            # Load model
            print("🧠 Loading model weights...")
            if self.device == "cpu":
                # Simplified loading for CPU
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=self.device,
                    **model_kwargs
                )
            
            # Configure generation
            self.generation_config = GenerationConfig(
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                max_new_tokens=512,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            print("✅ Qwen2.5 model loaded successfully!")
            self.is_ready = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Qwen2.5 model: {e}")
            print(f"❌ Model loading failed: {e}")
            return False
    
    def create_pipeline(self) -> bool:
        """Create HuggingFace pipeline for LangChain integration."""
        try:
            if not self.is_ready:
                print("❌ Model not ready. Please load model first.")
                return False
            
            print("🔗 Creating LangChain pipeline...")
            
            # Create HuggingFace pipeline - remove generation_config to avoid conflict
            from transformers import pipeline
            
            hf_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                # Remove generation_config to avoid duplicate parameter error
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Wrap for LangChain
            self.langchain_llm = HuggingFacePipeline(
                pipeline=hf_pipeline,
                model_kwargs={
                    "max_new_tokens": 512,
                    "temperature": 0.7,
                    "do_sample": True,
                    "top_p": 0.9
                }
            )
            
            print("✅ Pipeline created successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create pipeline: {e}")
            print(f"❌ Pipeline creation failed: {e}")
            
            # Try alternative approach without pipeline wrapper
            try:
                print("🔄 Trying direct model approach...")
                
                # Create a simple callable that works with LangChain
                def qwen_generate(prompt: str) -> str:
                    try:
                        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
                        if self.device == "cuda":
                            inputs = inputs.cuda()
                        
                        with torch.no_grad():
                            outputs = self.model.generate(
                                inputs,
                                max_new_tokens=512,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.9,
                                repetition_penalty=1.1,
                                pad_token_id=self.tokenizer.pad_token_id,
                                eos_token_id=self.tokenizer.eos_token_id
                            )
                        
                        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        # Remove input prompt from response
                        response = response[len(prompt):].strip()
                        return response
                        
                    except Exception as e:
                        return f"Generation error: {str(e)}"
                
                # Store the callable
                self.langchain_llm = qwen_generate
                print("✅ Direct model approach successful!")
                return True
                
            except Exception as e2:
                logger.error(f"❌ Alternative approach also failed: {e2}")
                return False
    
    def create_retrieval_qa(self, retriever) -> Optional[RetrievalQA]:
        """Create RetrievalQA chain with custom prompt."""
        try:
            if not self.langchain_llm:
                print("❌ LangChain LLM not ready. Please create pipeline first.")
                return None
            
            print("🔗 Creating RetrievalQA chain...")
            
            # Custom prompt template for better responses
            prompt_template = """You are a helpful AI assistant that answers questions based on the provided context. 
            Use the following pieces of context to answer the question at the end. 
            If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

            Context:
            {context}

            Question: {question}

            Answer: """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create RetrievalQA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.langchain_llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            print("✅ RetrievalQA chain created successfully!")
            return qa_chain
            
        except Exception as e:
            logger.error(f"❌ Failed to create RetrievalQA chain: {e}")
            print(f"❌ RetrievalQA creation failed: {e}")
            return None
    
    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response directly from model."""
        try:
            if not self.is_ready:
                return "Model not ready"
            
            print(f"🤖 Generating response with Qwen2.5-{self.model_name.split('-')[-2]}...")
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = inputs.cuda()
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode full response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input prompt from response (more robust method)
            if full_response.startswith(prompt):
                response = full_response[len(prompt):].strip()
            else:
                # Fallback: find where the actual response starts
                prompt_tokens = self.tokenizer.encode(prompt)
                response_tokens = outputs[0][len(prompt_tokens):]
                response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            # Clean up response
            if response:
                # Remove any remaining prompt artifacts
                response = response.strip()
                if response.startswith(('Answer:', 'Response:', 'A:')):
                    response = response.split(':', 1)[1].strip()
                    
                print(f"✅ Generated {len(response)} character response")
                return response
            else:
                return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return f"Generation failed: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and status."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "quantization": self.load_in_4bit,
            "is_ready": self.is_ready,
            "has_pipeline": self.langchain_llm is not None,
            "memory_usage": self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        memory_info = {}
        
        if torch.cuda.is_available():
            memory_info["gpu_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_info["gpu_cached"] = torch.cuda.memory_reserved() / 1024**3  # GB
        
        import psutil
        memory_info["cpu_percent"] = psutil.virtual_memory().percent
        
        return memory_info
    
    def cleanup(self):
        """Clean up model resources."""
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            if self.pipeline:
                del self.pipeline
            if self.langchain_llm:
                del self.langchain_llm
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_ready = False
            print("🧹 Model resources cleaned up")
            
        except Exception as e:
            logger.error(f"⚠️ Cleanup failed: {e}")


# Utility functions for easy model management
def load_qwen2_model(model_size: str = "0.5B", quantized: bool = True) -> Qwen2LLMManager:
    """Load Qwen2.5 model with recommended settings."""
    
    model_map = {
        "0.5B": "Qwen/Qwen2.5-0.5B-Instruct"
    }
    
    model_name = model_map.get(model_size, model_map["0.5B"])
    
    # Configure memory limits based on model size
    max_memory = None
    if model_size in ["14B", "32B"]:
        max_memory = {0: "10GB", "cpu": "30GB"}
    
    llm_manager = Qwen2LLMManager(
        model_name=model_name,
        load_in_4bit=quantized,
        max_memory=max_memory
    )
    
    return llm_manager


if __name__ == "__main__":
    # Test the LLM manager
    print("🧪 Testing Qwen2.5 LLM Manager...")
    
    # Load smaller model for testing
    manager = load_qwen2_model("0.5B", quantized=True)
    
    if manager.load_model():
        print("✅ Model loaded successfully!")
        
        # Test generation
        response = manager.generate_response("What is artificial intelligence?", max_tokens=100)
        print(f"🤖 Response: {response}")
        
        # Cleanup
        manager.cleanup()
    else:
        print("❌ Model loading failed") 