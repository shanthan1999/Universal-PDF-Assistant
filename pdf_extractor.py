import os
import PyPDF2
from typing import Dict, List


class PDFExtractor:
    """Robust PDF text extraction with multiple fallback methods."""
    
    def __init__(self):
        self.extraction_stats = {}
    
    def extract_text_robust(self, pdf_path: str) -> str:
        """Try multiple methods to extract text from PDF."""
        text = ""
        
        # Method 1: PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                print(f"  📖 {pdf_path}: {len(reader.pages)} pages (PyPDF2)")
                
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text and len(page_text.strip()) > 10:
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                
                if len(text.strip()) > 100:
                    print(f"  ✅ PyPDF2 success: {len(text)} characters")
                    return text
                else:
                    print(f"  ⚠️ PyPDF2 got little text")
                    
        except Exception as e:
            print(f"  ❌ PyPDF2 failed: {e}")
        
        return text if text.strip() else None
    
    def extract_from_directory(self, directory: str = ".") -> Dict[str, str]:
        """Extract text from all PDF files in a directory."""
        pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
        print(f"Found PDFs: {pdf_files}")
        
        all_documents_text = {}
        
        for pdf_file in pdf_files:
            file_path = os.path.join(directory, pdf_file)
            size = os.path.getsize(file_path)
            print(f"📄 {pdf_file}: {size:,} bytes")
            
            print(f"\nExtracting from {pdf_file}:")
            extracted_text = self.extract_text_robust(file_path)
            
            if extracted_text:
                all_documents_text[pdf_file] = extracted_text
                preview = extracted_text[:200].replace('\n', ' ')
                print(f"  📝 Preview: {preview}...")
                self.extraction_stats[pdf_file] = {
                    'status': 'success',
                    'length': len(extracted_text),
                    'size_bytes': size
                }
            else:
                print(f"  ❌ Failed to extract meaningful text from {pdf_file}")
                self.extraction_stats[pdf_file] = {
                    'status': 'failed',
                    'length': 0,
                    'size_bytes': size
                }
        
        if not all_documents_text:
            print("\n❌ No text extracted from any PDF!")
            print("Your PDFs might be:")
            print("- Image-based (scanned documents)")
            print("- Password protected")
            print("- Corrupted files")
            return {}
        
        print(f"\n✅ Successfully extracted text from {len(all_documents_text)} documents")
        return all_documents_text
    
    def get_extraction_stats(self) -> Dict:
        """Return extraction statistics."""
        return self.extraction_stats


if __name__ == "__main__":
    extractor = PDFExtractor()
    documents = extractor.extract_from_directory()
    
    if documents:
        print("\n" + "="*50)
        print("📊 EXTRACTION SUMMARY")
        print("="*50)
        
        for filename, text in documents.items():
            print(f"📄 {filename}: {len(text):,} characters")
        
        stats = extractor.get_extraction_stats()
        print(f"\nTotal files processed: {len(stats)}")
        successful = len([s for s in stats.values() if s['status'] == 'success'])
        print(f"Successful extractions: {successful}/{len(stats)}") 