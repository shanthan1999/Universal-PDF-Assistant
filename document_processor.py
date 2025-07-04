import re
from typing import List, Dict, Optional

class SimpleDocument:
    """Simple document class compatible with our system."""
    def __init__(self, page_content: str, metadata: Optional[Dict] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class DocumentProcessor:
    """Process and chunk documents for RAG system."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove multiple newlines and spaces
        cleaned_text = re.sub(r'\n+', '\n', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()
        return cleaned_text
    
    def split_text(self, text: str) -> List[str]:
        """Split text into smaller, sentence-based chunks for better retrieval."""
        if not text:
            return []
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Overlap: take last chunk_overlap chars from previous chunk
                overlap = current_chunk[-self.chunk_overlap:] if self.chunk_overlap < len(current_chunk) else current_chunk
                current_chunk = overlap + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks
    
    def create_rag_documents(self, documents_text: Dict[str, str]) -> List[SimpleDocument]:
        """Create RAG document chunks from extracted text."""
        print("\n" + "="*50)
        print("🔧 CREATING RAG DOCUMENT CHUNKS")
        print("="*50)
        
        all_rag_documents = []
        
        for source, text in documents_text.items():
            # Clean the text
            cleaned_text = self.clean_text(text)
            
            # Split into chunks
            chunks = self.split_text(cleaned_text)
            valid_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
            
            print(f"📄 {source}: {len(valid_chunks)} chunks created")
            
            # Create Document objects
            for i, chunk in enumerate(valid_chunks):
                doc = SimpleDocument(
                    page_content=chunk.strip(),
                    metadata={
                        "source": source.replace('.pdf', ''),
                        "chunk_id": i,
                        "total_chunks": len(valid_chunks)
                    }
                )
                all_rag_documents.append(doc)
        
        print(f"\n✅ Total RAG documents created: {len(all_rag_documents)}")
        
        if len(all_rag_documents) == 0:
            print("❌ No valid document chunks created!")
            return []
        
        return all_rag_documents
    
    def get_chunk_statistics(self, documents: List[SimpleDocument]) -> Dict:
        """Get statistics about document chunks."""
        if not documents:
            return {}
        
        chunk_lengths = [len(doc.page_content) for doc in documents]
        sources = [doc.metadata.get('source', 'unknown') for doc in documents]
        
        stats = {
            'total_chunks': len(documents),
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0,
            'min_chunk_length': min(chunk_lengths) if chunk_lengths else 0,
            'max_chunk_length': max(chunk_lengths) if chunk_lengths else 0,
            'sources': list(set(sources)),
            'chunks_per_source': {source: sources.count(source) for source in set(sources)}
        }
        
        return stats
    
    def preview_chunks(self, documents: List[SimpleDocument], num_samples: int = 3) -> None:
        """Preview a few document chunks."""
        print(f"\n📋 PREVIEW OF {min(num_samples, len(documents))} CHUNKS:")
        print("="*50)
        
        for i, doc in enumerate(documents[:num_samples]):
            source = doc.metadata.get('source', 'unknown')
            chunk_id = doc.metadata.get('chunk_id', 'unknown')
            preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            
            print(f"\n🔍 Chunk {i+1} from {source} (ID: {chunk_id}):")
            print(f"   {preview}")


if __name__ == "__main__":
    # Example usage
    from pdf_extractor import PDFExtractor
    
    print("Starting document processing...")
    
    # Extract text from PDFs
    extractor = PDFExtractor()
    documents_text = extractor.extract_from_directory()
    
    if documents_text:
        # Process documents
        processor = DocumentProcessor()
        rag_documents = processor.create_rag_documents(documents_text)
        
        if rag_documents:
            # Show statistics
            stats = processor.get_chunk_statistics(rag_documents)
            print("\n" + "="*50)
            print("📊 DOCUMENT PROCESSING STATISTICS")
            print("="*50)
            print(f"Total chunks: {stats['total_chunks']}")
            print(f"Average chunk length: {stats['avg_chunk_length']:.1f} characters")
            print(f"Min/Max chunk length: {stats['min_chunk_length']}/{stats['max_chunk_length']}")
            print(f"Sources: {', '.join(stats['sources'])}")
            
            for source, count in stats['chunks_per_source'].items():
                print(f"  {source}: {count} chunks")
            
            # Preview some chunks
            processor.preview_chunks(rag_documents)
    else:
        print("No documents to process.") 