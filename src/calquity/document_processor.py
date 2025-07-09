"""
Calquity - Professional Document Ingestion System
===============================================

Advanced document processing and ingestion pipeline for financial documents.
Handles extraction, transformation, and loading of real financial data including
annual reports, 10-K filings, and earnings call transcripts into vector database.

Key Features:
- Intelligent metadata extraction from filenames and content
- Robust text processing and chunking algorithms
- High-performance batch embedding generation
- Comprehensive error handling and logging

Author: Divyanshu Chaudhary
Version: 1.0.0
Created: 2025
License: Proprietary
"""

import os
import sys
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from .config_manager import DATABASE_CONFIG, EMBEDDING_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Module metadata
__author__ = "Divyanshu Chaudhary"
__version__ = "1.0.0"
__status__ = "Production"
__copyright__ = "Copyright 2025, Divyanshu Chaudhary"


class DocumentProcessor:
    """
    Professional document processing and ingestion system.
    
    Handles the complete pipeline from raw financial documents to searchable
    vector embeddings, including intelligent metadata extraction, content
    processing, and database storage optimization.
    """
    
    def __init__(self):
        """Initialize the document processor with embedding model and database connection."""
        try:
            self.model = SentenceTransformer(EMBEDDING_CONFIG['dense_model'])
            logger.info(f"Initialized embedding model: {EMBEDDING_CONFIG['dense_model']}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def extract_metadata(self, filename: str) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from financial document filename.
        
        Intelligently parses company tickers, years, quarters, and document types
        from standardized financial document naming conventions.
        
        Args:
            filename: Document filename to parse
            
        Returns:
            Dictionary containing extracted metadata fields
        """
        # Professional pattern matching for financial document naming conventions
        patterns = {
            'company': r'^([A-Z]+)[-_]',           # Company ticker at start with delimiter
            'year': r'(\d{4})',                   # 4-digit year identification
            'quarter': r'Q(\d)',                  # Quarter notation (Q1, Q2, etc.)
            'type': r'(10K|earnings|annual)',     # Document type classification
            'report_type': r'(Annual[-_]?Report|10[-_]?K|Earnings[-_]?Call)'  # Extended type matching
        }
        
        metadata = {
            'filename': filename,
            'processed_date': datetime.now().isoformat()
        }
        filename_upper = filename.upper()
        
        # Extract and validate company ticker
        company_match = re.search(patterns['company'], filename_upper)
        if company_match:
            ticker = company_match.group(1)
            # Handle common filename variations and corrections
            ticker_corrections = {
                'APPL': 'AAPL',  # Common typo correction
                'TSLA': 'TSLA',  # Tesla verification
                'MSFT': 'MSFT',  # Microsoft verification
                'META': 'META',  # Meta verification
                'NVDA': 'NVDA',  # NVIDIA verification
                'AVGO': 'AVGO',  # Broadcom verification
                'QCOM': 'QCOM'   # Qualcomm verification
            }
            metadata['ticker'] = ticker_corrections.get(ticker, ticker)
            metadata['company'] = metadata['ticker']
        else:
            logger.warning(f"Could not extract company ticker from filename: {filename}")
            metadata['ticker'] = 'UNKNOWN'
            metadata['company'] = 'UNKNOWN'
        
        # Extract and validate year
        year_match = re.search(patterns['year'], filename_upper)
        if year_match:
            year = int(year_match.group(1))
            # Validate reasonable year range for financial documents
            if 2020 <= year <= 2025:
                metadata['year'] = year
            else:
                logger.warning(f"Year {year} outside expected range (2020-2025) in {filename}")
                metadata['year'] = year
        else:
            logger.warning(f"Could not extract year from filename: {filename}")
            metadata['year'] = None
        
        # Extract quarter information
        quarter_match = re.search(patterns['quarter'], filename_upper)
        if quarter_match:
            metadata['quarter'] = f"Q{quarter_match.group(1)}"
        else:
            metadata['quarter'] = None
        
        # Determine document type with enhanced classification
        doc_type = 'Unknown'
        if re.search(r'10K', filename_upper):
            doc_type = '10-K Filing'
        elif re.search(r'ANNUAL[-_]?REPORT', filename_upper):
            doc_type = 'Annual Report'
        elif re.search(r'EARNINGS[-_]?CALL', filename_upper):
            doc_type = 'Earnings Call'
        elif re.search(r'EARNINGS', filename_upper):
            doc_type = 'Earnings Call'
        
        metadata['document_type'] = doc_type
        
        # Additional file-based metadata
        file_path = Path(filename)
        metadata['file_extension'] = file_path.suffix.lower()
        metadata['file_size_category'] = self._categorize_file_size(filename)
        
        logger.debug(f"Extracted metadata from {filename}: {metadata}")
        return metadata
    
    def _categorize_file_size(self, filename: str) -> str:
        """Categorize file size for processing optimization."""
        try:
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                if size < 100_000:  # < 100KB
                    return 'small'
                elif size < 1_000_000:  # < 1MB
                    return 'medium'
                else:  # >= 1MB
                    return 'large'
        except Exception:
            pass
        return 'unknown'
    
    def read_file_content(self, filepath: str) -> str:
        """
        Read and process file content with proper encoding handling.
        
        Args:
            filepath: Path to the file to read
            
        Returns:
            Processed file content as string
        """
        try:
            # Try common encodings for financial documents
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        content = f.read()
                    logger.debug(f"Successfully read {filepath} with {encoding} encoding")
                    return content
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, read as binary and decode with errors='replace'
            with open(filepath, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')
            logger.warning(f"Had to use error-tolerant decoding for {filepath}")
            return content
            
        except Exception as e:
            logger.error(f"Failed to read file {filepath}: {e}")
            raise
    
    def chunk_text(self, text: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Intelligently chunk text into overlapping segments for optimal embedding.
        
        Args:
            text: Input text to chunk
            max_chunk_size: Maximum characters per chunk
            overlap: Character overlap between chunks
            
        Returns:
            List of text chunks with optimal overlap
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            # If we're not at the end, try to find a good break point
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                last_space = text.rfind(' ', start, end)
                
                # Choose the best break point
                break_point = max(last_period, last_newline, last_space)
                if break_point > start + max_chunk_size - 200:  # Within reasonable distance
                    end = break_point + 1
            
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap
            
            # Prevent infinite loops
            if start >= len(text):
                break
        
        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks
        
        # Extract document type
        if '10K' in filename_upper:
            metadata['document_type'] = '10-K'
        elif 'EARNINGS' in filename_upper:
            metadata['document_type'] = 'Earnings Call'
        elif 'ANNUAL' in filename_upper:
            metadata['document_type'] = 'Annual Report'
        else:
            metadata['document_type'] = 'Financial Document'
            
        return metadata
    
    def read_file_content(self, file_path):
        """Read content from different file types"""
        import re  # Import at function level
        try:
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_path.endswith('.html'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Basic HTML cleaning - remove tags
                    content = re.sub(r'<[^>]+>', ' ', content)
                    content = re.sub(r'\s+', ' ', content)
                    return content.strip()
            elif file_path.endswith('.pdf'):
                # PDF processing using PyPDF2
                try:
                    from PyPDF2 import PdfReader
                    reader = PdfReader(file_path)
                    content = ""
                    for page in reader.pages:
                        content += page.extract_text() + "\n"
                    
                    # Clean up the text
                    content = re.sub(r'\s+', ' ', content)
                    return content.strip()
                except Exception as pdf_error:
                    print(f"❌ Error reading PDF {file_path}: {pdf_error}")
                    return None
            else:
                print(f"⚠️ Unsupported file type: {file_path}")
                return None
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            return None
    
    def chunk_content(self, content, chunk_size=1000, overlap=200):
        """Split content into overlapping chunks"""
        if not content:
            return []
            
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
                
        return chunks
    
    def generate_embedding(self, text):
        """Generate dense embedding using sentence transformer"""
        try:
            embedding = self.model.encode(text)
            return embedding.astype(np.float32)
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return None
    
    def generate_sparse_embedding(self, text):
        """Generate simple sparse embedding (keyword frequencies)"""
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = {}
        for word in words:
            if len(word) > 2:  # Skip very short words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Keep only top 50 most frequent words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:50]
        return dict(sorted_words)

def ingest_documents():
    """Main ingestion function"""
    print("🚀 DOCUMENT INGESTION - calcuityTask")
    print("=" * 50)
    
    processor = DocumentProcessor()
    docs_folder = "real_documents"
    
    if not os.path.exists(docs_folder):
        print(f"❌ Documents folder not found: {docs_folder}")
        return
    
    # Connect to database
    try:
        conn = psycopg2.connect(**DATABASE_CONFIG)
        register_vector(conn)
        cursor = conn.cursor()
        print(f"✅ Connected to database: {DATABASE_CONFIG['database']}")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return
    
    # Process each file
    files = [f for f in os.listdir(docs_folder) if f.endswith(('.txt', '.html', '.pdf'))]
    print(f"📁 Found {len(files)} documents to process")
    
    total_chunks = 0
    
    for filename in files:
        print(f"\n📄 Processing: {filename}")
        file_path = os.path.join(docs_folder, filename)
        
        # Extract metadata
        metadata = processor.extract_metadata(filename)
        print(f"   Company: {metadata['company']} | Year: {metadata['year']} | Type: {metadata['document_type']}")
        
        # Read content
        content = processor.read_file_content(file_path)
        if not content:
            continue
            
        # Chunk content
        chunks = processor.chunk_content(content)
        print(f"   Created {len(chunks)} chunks")
        
        # Process each chunk
        for chunk_idx, chunk in enumerate(chunks):
            try:
                # Generate embeddings
                dense_embedding = processor.generate_embedding(chunk)
                sparse_embedding = processor.generate_sparse_embedding(chunk)
                
                if dense_embedding is None:
                    continue
                
                # Insert into database
                insert_sql = """
                INSERT INTO documents (
                    title, content, file_path, file_type, chunk_index,
                    dense_embedding, sparse_embedding, company, ticker,
                    year, quarter, document_type
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                cursor.execute(insert_sql, (
                    filename,  # title
                    chunk,     # content
                    file_path, # file_path
                    filename.split('.')[-1],  # file_type
                    chunk_idx, # chunk_index
                    dense_embedding,  # dense_embedding
                    psycopg2.extras.Json(sparse_embedding), # sparse_embedding (JSON)
                    metadata['company'],     # company
                    metadata['ticker'],      # ticker
                    metadata['year'],        # year
                    metadata['quarter'],     # quarter
                    metadata['document_type'] # document_type
                ))
                
                total_chunks += 1
                
            except Exception as e:
                print(f"   ❌ Error processing chunk {chunk_idx}: {e}")
                continue
    
    # Commit and close
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"\n🎉 INGESTION COMPLETE!")
    print("=" * 50)
    print(f"✅ Total documents processed: {len(files)}")
    print(f"✅ Total chunks inserted: {total_chunks}")
    print(f"✅ Database: {DATABASE_CONFIG['database']}")
    print("✅ Ready for vector search!")

if __name__ == "__main__":
    # Install required package if not present
    try:
        import sentence_transformers
    except ImportError:
        print("Installing sentence-transformers...")
        os.system("pip install sentence-transformers")
        import sentence_transformers
    
    ingest_documents()
