# Calquity API Documentation

**Author:** Divyanshu Chaudhary  
**Version:** 1.0.0  

## Core Components

### HybridSearchEngine

The main search interface providing dense, sparse, and hybrid search capabilities.

#### Methods

- `search(query, search_type='hybrid', limit=10, company_filter=None)`
- `dense_search(query, limit=10, company_filter=None)`
- `sparse_search(query, limit=10, company_filter=None)`
- `hybrid_search(query, limit=10, company_filter=None)`
- `health_check()`
- `get_search_stats()`

### ConfigManager

Central configuration management system.

#### Properties

- `database`: Database configuration
- `embedding`: Embedding model configuration  
- `search`: Search algorithm configuration
- `system`: System-wide configuration

### DocumentProcessor

Advanced document processing and ingestion pipeline.

#### Methods

- `extract_metadata(filename)`
- `read_file_content(filepath)`
- `chunk_text(text, max_chunk_size=1000, overlap=200)`
