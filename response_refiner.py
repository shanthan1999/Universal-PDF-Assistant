import re
import torch
from typing import Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


class ResponseRefiner:
    """Refine and improve RAG responses using transformer models."""
    
    def __init__(self, use_summarizer: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.summarizer = None
        self.use_summarizer = use_summarizer
        
        if self.use_summarizer:
            self._load_summarizer()
    
    def _load_summarizer(self):
        """Load a lightweight summarization model."""
        try:
            # Using a small, efficient model for summarization
            model_name = "facebook/bart-large-cnn"
            print(f"Loading summarization model: {model_name}")
            
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            print("✅ Summarizer loaded successfully")
        except Exception as e:
            print(f"⚠️ Summarizer loading failed: {e}")
            print("Falling back to rule-based refinement only")
            self.summarizer = None
    
    def extract_key_sentences(self, text: str, query: str, max_sentences: int = 3) -> str:
        """Extract sentences most relevant to the query."""
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return text
        
        # Calculate relevance scores based on query word overlap
        query_words = set(query.lower().split())
        scored_sentences = []
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            if overlap > 0:
                scored_sentences.append((sentence, overlap))
        
        # Sort by relevance score
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if scored_sentences:
            top_sentences = [s[0] for s in scored_sentences[:max_sentences]]
            return '. '.join(top_sentences) + '.'
        else:
            # If no relevant sentences found, return first few sentences
            return '. '.join(sentences[:max_sentences]) + '.'
    
    def remove_redundancy(self, text: str) -> str:
        """Remove repetitive and redundant content."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        unique_sentences = []
        seen_content = set()
        
        for sentence in sentences:
            # Create a simplified version for comparison
            simplified = re.sub(r'[^\w\s]', '', sentence.lower())
            words = set(simplified.split())
            hashable_words = tuple(sorted(list(words)))
            
            # Check for significant overlap with existing sentences
            is_duplicate = False
            for seen_words_tuple in seen_content:
                overlap = len(words.intersection(set(seen_words_tuple)))
                if overlap > len(words) * 0.7:  # 70% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate and len(words) > 2:
                unique_sentences.append(sentence)
                seen_content.add(hashable_words)
        
        return '. '.join(unique_sentences) + '.' if unique_sentences else text
    
    def clean_response_text(self, text: str) -> str:
        """Clean common verbose patterns and filler words."""
        # Define verbose patterns to remove
        verbose_patterns = [
            r'Based on the (?:context|document|information) (?:provided|given|above)[,.]?\s*',
            r'According to the (?:document|text|information)[,.]?\s*',
            r'The document (?:states|mentions|says) that\s*',
            r'From the (?:information|context) (?:provided|given)[,.]?\s*',
            r'As (?:mentioned|stated|indicated) in the (?:document|text)[,.]?\s*',
            r'It is (?:mentioned|stated|indicated) that\s*',
            r'The (?:above|following) information (?:shows|indicates|suggests)\s*',
            r'In (?:summary|conclusion)[,.]?\s*',
            r'To (?:summarize|conclude)[,.]?\s*'
        ]
        
        cleaned = text
        for pattern in verbose_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Clean up extra whitespace and punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\s*[.]{2,}\s*', '. ', cleaned)
        cleaned = cleaned.strip()
        
        # Ensure proper ending punctuation
        if cleaned and not cleaned.endswith(('.', '!', '?')):
            cleaned += '.'
        
        return cleaned
    
    def refine_response(self, query: str, response: str, target_length: str = "short") -> str:
        """Main method to refine lengthy responses based on query."""
        if not response or not query:
            return response
        
        # Step 1: Clean verbose patterns
        cleaned = self.clean_response_text(response)
        
        # Step 2: Remove redundancy
        deduplicated = self.remove_redundancy(cleaned)
        
        # Step 3: Extract relevant content
        length_map = {"short": 2, "medium": 3, "long": 4}
        max_sentences = length_map.get(target_length, 2)
        relevant = self.extract_key_sentences(deduplicated, query, max_sentences)
        
        # Step 4: Use AI summarization if available and text is long enough
        if self.summarizer and len(relevant.split()) > 100:
            try:
                max_length = {"short": 50, "medium": 80, "long": 120}[target_length]
                min_length = max_length // 2
                
                summarized = self.summarizer(
                    relevant,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )[0]['summary_text']
                
                return summarized
            except Exception as e:
                print(f"⚠️ Summarization failed: {e}")
                return relevant
        
        return relevant
    
    def quick_refine(self, query: str, response: str, max_words: int = 50) -> str:
        """Quick refinement without AI models."""
        cleaned = self.clean_response_text(response)
        relevant = self.extract_key_sentences(cleaned, query, max_sentences=2)
        
        words = relevant.split()
        if len(words) > max_words:
            relevant = ' '.join(words[:max_words]) + '...'
        
        return relevant
    
    def batch_refine(self, queries_responses: list, target_length: str = "short") -> list:
        """Refine multiple query-response pairs."""
        refined_responses = []
        
        for query, response in queries_responses:
            refined = self.refine_response(query, response, target_length)
            refined_responses.append((query, response, refined))
        
        return refined_responses
    
    def get_refinement_stats(self) -> dict:
        """Return statistics about the response refiner configuration."""
        return {
            'use_summarizer': self.use_summarizer,
            'summarizer_available': self.summarizer is not None,
            'model_type': 'AI-powered' if self.summarizer else 'Rule-based',
            'status': 'ready'
        }


if __name__ == "__main__":
    # Example usage
    refiner = ResponseRefiner(use_summarizer=False)  # Set to True to use AI summarization
    
    print("Response Refiner Ready!")
    print("="*50)
    
    # Example refinement
            sample_query = "What are the main points discussed in this document?"
        sample_response = """
        Based on the document provided, the main points discussed include several key aspects that 
    organizations must consider. According to the information given, AI systems should be designed with 
    fairness in mind, ensuring that they do not discriminate against any group. The document states that 
    transparency is crucial, meaning AI decisions should be explainable. According to the text, accountability 
    is also essential, with clear responsibility chains. The document mentions that privacy must be protected 
    throughout the AI lifecycle. It is stated that AI should be beneficial and not cause harm. The above 
    information shows that these principles guide responsible AI development. To summarize, the main principles 
    are fairness, transparency, accountability, privacy, and beneficence.
    """
    
    print(f"Original Query: {sample_query}")
    print(f"Original Response Length: {len(sample_response.split())} words")
    print(f"Original Response:\n{sample_response}")
    
    print("\n" + "="*50)
    print("REFINED RESPONSES")
    print("="*50)
    
    # Test different refinement approaches
    refined_short = refiner.refine_response(sample_query, sample_response, "short")
    print(f"\nShort Refinement ({len(refined_short.split())} words):")
    print(refined_short)
    
    refined_medium = refiner.refine_response(sample_query, sample_response, "medium")
    print(f"\nMedium Refinement ({len(refined_medium.split())} words):")
    print(refined_medium)
    
    quick_refined = refiner.quick_refine(sample_query, sample_response, max_words=25)
    print(f"\nQuick Refinement (max 25 words):")
    print(quick_refined) 