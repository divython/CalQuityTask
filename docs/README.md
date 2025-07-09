# Calquity - Professional Hybrid Search System for Financial Documents

**Author:** Divyanshu Chaudhary  
**Version:** 1.0.0  
**Status:** Production Ready  
**Created:** 2025  

## Executive Summary

Calquity is a production-grade hybrid search system specifically engineered for financial document analysis and retrieval. The system combines state-of-the-art dense vector search (semantic similarity) with traditional sparse full-text search to deliver comprehensive search capabilities across real-world financial documents including annual reports, 10-K filings, and earnings call transcripts.

## Technical Architecture

### Core Technologies
- **Database**: PostgreSQL 14+ with PGVector extension for high-performance vector operations
- **Vector Embeddings**: Sentence-BERT (all-MiniLM-L6-v2) for semantic understanding
- **Search Engine**: Hybrid architecture combining dense and sparse retrieval methods
- **Web Interface**: Streamlit-based professional dashboard with real-time analytics
- **Language**: Python 3.8+ with enterprise-grade libraries

### System Components

#### 1. Hybrid Search Engine (`hybrid_search.py`)
Professional search orchestration system providing:
- **Dense Search**: Semantic similarity using 384-dimensional embeddings
- **Sparse Search**: PostgreSQL full-text search with advanced ranking
- **Hybrid Search**: Optimized weighted combination (70% dense, 30% sparse)
- **Performance Monitoring**: Comprehensive timing and quality metrics
- **Health Checks**: System status validation and diagnostics

#### 2. Configuration Management (`config.py`)
Enterprise-grade configuration system featuring:
- **Environment Integration**: Automatic environment variable loading
- **Validation Framework**: Comprehensive configuration validation
- **Connection Management**: Database connection pooling and optimization
- **Type Safety**: Full type hints and dataclass-based configuration

#### 3. Document Ingestion Pipeline (`ingest_documents.py`)
Advanced document processing system including:
- **Intelligent Metadata Extraction**: Automatic company, year, and document type identification
- **Robust Content Processing**: Multi-encoding support and error handling
- **Optimized Chunking**: Smart text segmentation with semantic boundaries
- **Batch Processing**: High-performance embedding generation

#### 4. Performance Analysis Framework (`experiment_analysis.py`)
Comprehensive evaluation system providing:
- **Comparative Analysis**: Performance metrics across all search methods
- **Statistical Reporting**: Detailed timing and quality analysis
- **Overlap Assessment**: Method complementarity evaluation
- **Production Monitoring**: Continuous performance tracking

#### 5. Interactive Web Interface (`streamlit_app.py`)
Professional user interface featuring:
- **Real-time Search**: Live search across all methods with performance metrics
- **Visual Analytics**: Interactive charts and performance comparisons
- **Result Visualization**: Professional formatting with metadata display
- **System Monitoring**: Health status and configuration display

## Data Coverage

### Financial Documents Repository
The system operates on a comprehensive dataset of **1,930 document chunks** from **7 major companies** spanning **2022-2025**:

| Company | Ticker | Document Types | Coverage |
|---------|--------|----------------|----------|
| **Apple** | AAPL | 10-K Reports, Earnings Calls | 2022-2025 |
| **Broadcom** | AVGO | Annual Reports, Earnings Calls | 2024-2025 |
| **Coca-Cola** | CC | Earnings Calls | 2025 |
| **Meta** | META | Annual Reports, Earnings Calls | 2025 |
| **Microsoft** | MSFT | 10-K Reports | 2022 |
| **NVIDIA** | NVDA | Annual Reports | 2025 |
| **Qualcomm** | QCOM | Annual Reports | 2024 |
| **Tesla** | TSLA | 10-K Reports, Earnings Calls | 2022-2025 |

### Document Types
- **10-K Filings**: Comprehensive annual reports filed with SEC
- **Annual Reports**: Company-published yearly performance summaries  
- **Earnings Calls**: Quarterly earnings conference call transcripts

## Performance Analysis & Results

### Comprehensive Performance Benchmarking

Our extensive testing across 25 professional financial queries demonstrates the system's production-ready performance characteristics:

| Search Method | Avg Time (ms) | Min Time (ms) | Max Time (ms) | Std Dev (ms) | Consistency |
|---------------|---------------|---------------|---------------|--------------|-------------|
| **Sparse (Keyword)** | 34.0 | 0.4 | 115.1 | 28.8 | Variable |
| **Dense (Semantic)** | 37.7 | 23.1 | 55.0 | 7.0 | Highly Consistent |
| **Hybrid (Combined)** | 65.4 | 30.8 | 134.8 | 28.9 | Balanced |

### Method Complementarity Analysis

| Method Pair | Avg Overlap (%) | Median (%) | Interpretation |
|-------------|----------------|------------|----------------|
| Dense ∩ Sparse | 7.0 | 0.0 | Highly Complementary |
| Dense ∩ Hybrid | 74.0 | 70.0 | Strong Semantic Influence |
| Sparse ∩ Hybrid | 37.5 | 30.0 | Balanced Integration |

### Key Performance Insights

1. **Optimal Speed-Quality Balance**:
   - Sparse search delivers fastest individual query response (34ms avg)
   - Dense search provides most consistent performance (7ms std dev)
   - Hybrid search optimizes for comprehensive result quality

2. **Method Complementarity**:
   - Dense and sparse methods show minimal overlap (7%), indicating strong complementarity
   - Hybrid search successfully captures benefits from both approaches
   - Different query types favor different search methods

3. **Production Scalability**:
   - All methods demonstrate sub-100ms average response times
   - System handles 1,930+ document chunks efficiently
   - Consistent performance across diverse financial query types

## Installation & Setup

### Prerequisites
```bash
# System Requirements
- Python 3.8+
- PostgreSQL 14+ with PGVector extension
- 4GB+ RAM recommended
- Modern CPU with vector instruction support
```

### Database Configuration
```sql
-- Create database and enable vector extension
CREATE DATABASE calquitytask;
\c calquitytask;
CREATE EXTENSION vector;

-- Create documents table with vector indexing
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    content_vector vector(384),
    company VARCHAR(10),
    document_type VARCHAR(50),
    year INTEGER,
    quarter VARCHAR(5),
    filename VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create optimized indexes
CREATE INDEX idx_documents_vector ON documents USING ivfflat (content_vector vector_cosine_ops);
CREATE INDEX idx_documents_company ON documents(company);
CREATE INDEX idx_documents_type ON documents(document_type);
CREATE INDEX idx_documents_year ON documents(year);
```

### Python Environment Setup
```bash
# Clone repository
git clone [repository-url]
cd calquity

# Install dependencies
pip install -r requirements.txt

# Configure environment
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=calquitytask
export DB_USER=postgres
export DB_PASSWORD=your_password
```

### Data Ingestion
```bash
# Process and ingest financial documents
python ingest_documents.py

# Verify data loading
python -c "from hybrid_search import HybridSearchEngine; engine = HybridSearchEngine(); print(engine.get_stats())"
```

## Usage Examples

### Command Line Interface
```python
from hybrid_search import HybridSearchEngine

# Initialize search engine
engine = HybridSearchEngine()

# Perform hybrid search
results = engine.hybrid_search("revenue growth and profitability", limit=5)

# Display results
for i, result in enumerate(results, 1):
    print(f"{i}. {result['company']} - {result['document_type']}")
    print(f"   Score: {result['combined_score']:.4f}")
    print(f"   Content: {result['content'][:200]}...")
```

### Web Interface
```bash
# Launch professional web interface
streamlit run streamlit_app.py

# Access at http://localhost:8501
# Features:
# - Real-time search across all methods
# - Performance comparison analytics
# - Interactive result visualization
# - System health monitoring
```

### Performance Analysis
```python
from experiment_analysis import SearchExperiment

# Run comprehensive performance analysis
experiment = SearchExperiment()
results = experiment.run_full_experiment()

# Generate detailed performance report
experiment.generate_comprehensive_report()
```

## API Reference

### Core Search Methods

#### `hybrid_search(query, limit=10, weights=None)`
**Primary search method combining dense and sparse approaches**
- `query`: Search query string
- `limit`: Maximum results to return
- `weights`: Optional custom weighting (default: 70% dense, 30% sparse)
- **Returns**: List of ranked results with combined scores

#### `dense_search(query, limit=10)`
**Semantic similarity search using embeddings**
- Leverages sentence-transformer embeddings for conceptual matching
- Optimal for finding contextually similar content

#### `sparse_search(query, limit=10)`
**Traditional keyword-based full-text search**
- PostgreSQL full-text search with advanced ranking
- Optimal for exact term matching and specific keyword queries

### System Monitoring

#### `get_stats()`
**Comprehensive system statistics**
- Document count and distribution
- Database performance metrics
- Search method usage statistics

#### `health_check()`
**System health validation**
- Database connectivity verification
- Model loading status
- Performance benchmark validation

## Project Structure

```
calquity/
├── hybrid_search.py          # Core search engine implementation
├── config.py                 # Professional configuration management
├── ingest_documents.py       # Document processing pipeline
├── experiment_analysis.py    # Performance analysis framework
├── streamlit_app.py          # Professional web interface
├── README.md                 # Comprehensive documentation
├── real_documents/           # Financial document repository
│   ├── AAPL_*.{html,txt}    # Apple financial documents
│   ├── AVGO_*.{pdf,txt}     # Broadcom documents
│   ├── META_*.{pdf,txt}     # Meta documents
│   ├── MSFT_*.html          # Microsoft documents
│   ├── NVDA_*.pdf           # NVIDIA documents
│   ├── QCOM_*.pdf           # Qualcomm documents
│   └── TSLA_*.{html,txt}    # Tesla documents
└── search_experiment_*.json  # Performance analysis results
```

## Professional Attributes

- **Production Ready**: Comprehensive error handling, logging, and monitoring
- **Scalable Architecture**: Designed for enterprise-scale document collections
- **Performance Optimized**: Sub-100ms average query response times
- **Comprehensive Testing**: Extensive validation across 25+ professional queries
- **Professional Documentation**: Complete API reference and usage examples
- **Maintainable Codebase**: Type hints, docstrings, and modular architecture

## Technical Specifications

- **Database**: PostgreSQL 14+ with PGVector extension
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Vector Index**: IVFFlat with cosine similarity
- **Search Weights**: 70% dense, 30% sparse (optimized)
- **Performance**: <100ms average response time
- **Scalability**: Supports 10,000+ document collections

---

**Created by:** Divyanshu Chaudhary  
**License:** Proprietary  
**Version:** 1.0.0 Production  
**Copyright:** 2025 Divyanshu Chaudhary  
**Status:** Enterprise Ready
   - Sparse search is fastest (34ms avg) but highly variable
   - Dense search is consistent (37.7ms ± 7ms)
   - Hybrid search takes ~2x longer but provides comprehensive results

2. **Result Diversity**:
   - Dense and sparse searches show minimal overlap (7%), indicating they capture different aspects
   - Hybrid search is heavily influenced by dense results (74% overlap)
   - Each method contributes unique value to the overall search experience

3. **Search Quality**:
   - Dense search excels at semantic understanding and concept matching
   - Sparse search captures exact keyword matches and specific terminology
   - Hybrid search provides the most comprehensive coverage

## Application Features

### Streamlit Web Interface
- **Real-time Search**: Compare all three search methods simultaneously
- **Performance Metrics**: Live timing and overlap analysis
- **Visual Analytics**: Charts showing search performance and result distribution
- **Sample Queries**: Pre-loaded financial domain queries
- **Search History**: Track and compare previous searches

### Key Capabilities
- Semantic search for conceptual queries
- Keyword-based search for specific terms
- Company and document type filtering
- Performance comparison and analytics
- Export and analysis of search results

## Technical Implementation

### Core Files
- `config.py`: Database and model configuration
- `hybrid_search.py`: Main search engine implementation
- `streamlit_app.py`: Web interface and user experience
- `experiment_analysis.py`: Comprehensive performance testing
- `ingest_documents.py`: Document processing and ingestion

### Search Engine Details

#### Dense Search
- **Model**: all-MiniLM-L6-v2 (384-dimensional embeddings)
- **Method**: PGVector cosine similarity
- **Index**: IVFFlat index for fast approximate search
- **Strength**: Semantic understanding, concept matching

#### Sparse Search  
- **Method**: PostgreSQL full-text search with ts_rank
- **Features**: Stemming, stop word removal, phrase matching
- **Index**: GIN index on tsvector columns
- **Strength**: Exact keyword matching, terminology precision

#### Hybrid Search
- **Combination**: Weighted merge of dense and sparse results
- **Scoring**: Normalized and combined similarity scores
- **Deduplication**: Intelligent merging of overlapping results
- **Strength**: Comprehensive coverage, balanced relevance

## Use Case Recommendations

### When to Use Dense Search
- ✅ Conceptual queries ("risk management strategies")
- ✅ Semantic similarity ("financial performance metrics")
- ✅ Cross-domain concept matching
- ✅ Exploratory research queries

### When to Use Sparse Search
- ✅ Specific term searches ("EBITDA", "dividend yield")
- ✅ Exact phrase matching ("supply chain disruption")
- ✅ Regulatory or technical terminology
- ✅ Fast keyword-based retrieval

### When to Use Hybrid Search
- ✅ Comprehensive document discovery
- ✅ Unknown terminology mixed with concepts
- ✅ Research requiring both precision and recall
- ✅ General-purpose financial document search

## Performance Optimization

### Database Optimizations
- Vector indexing with IVFFlat for dense search
- GIN indexing for full-text search
- Connection pooling for concurrent access
- Proper query parameterization

### Search Optimizations
- Cached embedding model loading
- Optimized SQL queries with proper joins
- Result deduplication and merging
- Configurable result limits and filtering

## Justification for HTML Data Storage

### Why HTML Documents Are Included
1. **Industry Standard**: Many financial documents are published in HTML format
2. **Rich Structure**: HTML preserves document structure and formatting
3. **Search Compatibility**: HTML content works well with both dense and sparse search
4. **Real-world Relevance**: Represents actual data sources used in financial analysis

### HTML Processing Approach
- Content extraction while preserving meaningful structure
- Entity decoding for proper text representation
- Removal of navigation and boilerplate content
- Retention of financial data tables and structured information

## Future Enhancements

### Technical Improvements
- Advanced sparse search using JSONB storage for custom scoring
- Query expansion and synonym handling
- Multi-modal search incorporating document structure
- Real-time index updates for new documents

### Feature Additions
- Saved search and alert capabilities
- Advanced filtering by date ranges and document sections
- Export capabilities for search results
- Integration with external financial data sources

## Deployment Considerations

### Production Requirements
- PostgreSQL cluster with replication
- Load balancing for the web application
- Monitoring and alerting for search performance
- Backup and disaster recovery procedures

### Scaling Strategies
- Horizontal scaling with database sharding
- Caching layer for frequent queries
- Asynchronous processing for large document ingestion
- CDN integration for static assets

## Conclusion

The Calquity hybrid search system successfully demonstrates the complementary nature of dense and sparse search methods for financial document analysis. While each method has its strengths, the hybrid approach provides the most comprehensive search experience, justifying the additional computational cost for applications requiring thorough document discovery and analysis.

The system is production-ready with proper error handling, comprehensive testing, and a user-friendly interface. The experimental results provide clear guidance on when to use each search method, enabling users to choose the optimal approach for their specific use cases.
