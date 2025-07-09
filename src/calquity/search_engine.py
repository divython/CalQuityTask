"""
Hybrid Search Engine for Document Analysis

Author: Divyanshu Chaudhary
Version: 1.0.0
"""

import logging
import time
from typing import List, Dict, Optional, Union
import re

import psycopg2
import psycopg2.extras
import numpy as np
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

from .config_manager import DATABASE_CONFIG, EMBEDDING_CONFIG, SEARCH_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridSearchEngine:
    """
    Hybrid search engine combining dense vector search with sparse keyword search.
    
    Author: Divyanshu Chaudhary
    """
    
    def __init__(self):
        """Initialize the search engine with database connection and ML model."""
        logger.info("Initializing HybridSearchEngine")
        
        self.model = None
        self.conn = None
        self.cursor = None
        
        self._load_model()
        self._connect()
        
        logger.info("HybridSearchEngine initialized")
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            model_name = EMBEDDING_CONFIG['dense_model']
            logger.info(f"Loading model: {model_name}")
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def _connect(self) -> None:
        """Connect to PostgreSQL database with PGVector."""
        try:
            self.conn = psycopg2.connect(**DATABASE_CONFIG)
            register_vector(self.conn)
            self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def _reconnect(self) -> None:
        """Attempt to reconnect to the database in case of connection loss."""
        logger.warning("Attempting database reconnection")
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
        except Exception:
            pass
        
        self._connect()
    
    def generate_dense_embedding(self, query: str) -> np.ndarray:
        """Generate dense vector embedding for semantic search."""
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        try:
            embedding = self.model.encode(query)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Dense embedding generation failed: {e}")
            raise Exception(f"Error generating dense embedding: {e}")
    
    def generate_sparse_embedding(self, query: str) -> Dict[str, float]:
        """Generate sparse keyword-based embedding for exact term matching."""
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        try:
            # Extract meaningful words (length > 2)
            words = re.findall(r'\b\w+\b', query.lower())
            word_counts = {}
            
            # Count word frequencies
            for word in words:
                if len(word) > 2:
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            # Convert to relative weights
            total_words = sum(word_counts.values())
            if total_words > 0:
                sparse_embedding = {
                    word: count / total_words 
                    for word, count in word_counts.items()
                }
            else:
                sparse_embedding = {}
            
            return sparse_embedding
            
        except Exception as e:
            logger.error(f"Sparse embedding generation failed: {e}")
            raise Exception(f"Error generating sparse embedding: {e}")
    
    def dense_search(self, query: str, limit: int = 10, company_filter: Optional[str] = None) -> List[Dict]:
        """
        Perform semantic similarity search using dense vector embeddings.
        
        Uses cosine similarity with PGVector for efficient nearest neighbor search.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            company_filter: Optional company code filter (e.g., 'AAPL')
            
        Returns:
            List of search results with metadata and similarity scores
            
        Raises:
            Exception: If search execution fails
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        if limit <= 0 or limit > 100:
            raise ValueError("Limit must be between 1 and 100")
        
        try:
            logger.info(f"Executing dense search: '{query[:50]}...' (limit={limit})")
            start_time = time.time()
            
            # Generate query embedding
            query_embedding = self.generate_dense_embedding(query)
            
            # Construct SQL query with optional company filter
            sql_base = """
                SELECT 
                    id, title, company, document_type, year, quarter,
                    LEFT(content, 300) as content_preview,
                    dense_embedding <-> %s as similarity_score
                FROM documents 
                WHERE dense_embedding IS NOT NULL
            """
            
            params = [query_embedding]
            
            if company_filter:
                sql_base += " AND company = %s"
                params.append(company_filter.upper())
            
            sql_final = sql_base + " ORDER BY dense_embedding <-> %s LIMIT %s"
            params.extend([query_embedding, limit])
            
            # Execute query
            self.cursor.execute(sql_final, params)
            raw_results = self.cursor.fetchall()
            
            # Format results with metadata
            search_time = time.time() - start_time
            formatted_results = []
            
            for row in raw_results:
                result = dict(row)
                result['search_time'] = search_time
                result['search_type'] = 'dense'
                formatted_results.append(result)
            
            logger.info(f"Dense search completed: {len(formatted_results)} results in {search_time:.3f}s")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            raise Exception(f"Dense search error: {e}")
    
    def sparse_search(self, query: str, limit: int = 10, company_filter: Optional[str] = None) -> List[Dict]:
        """Perform keyword-based search using PostgreSQL full-text search."""
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        if limit <= 0 or limit > 100:
            raise ValueError("Limit must be between 1 and 100")
        
        try:
            logger.info(f"Executing sparse search: '{query[:50]}...' (limit={limit})")
            start_time = time.time()
            
            # Generate sparse embedding to extract keywords
            query_sparse = self.generate_sparse_embedding(query)
            if not query_sparse:
                logger.warning("No valid keywords extracted from query")
                return []
            
            # Create search terms for full-text search
            keywords = list(query_sparse.keys())
            search_terms = " | ".join(keywords)
            
            # Construct SQL query with full-text search
            sql_base = """
                SELECT 
                    id, title, company, document_type, year, quarter,
                    LEFT(content, 300) as content_preview,
                    ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) as similarity_score
                FROM documents 
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
            """
            
            params = [search_terms, search_terms]
            
            if company_filter:
                sql_base += " AND company = %s"
                params.append(company_filter.upper())
            
            sql_final = sql_base + " ORDER BY similarity_score DESC LIMIT %s"
            params.append(limit)
            
            # Execute query
            self.cursor.execute(sql_final, params)
            raw_results = self.cursor.fetchall()
            search_time = time.time() - start_time
            formatted_results = []
            
            for row in raw_results:
                result = dict(row)
                result['search_time'] = search_time
                result['search_type'] = 'sparse'
                formatted_results.append(result)
            
            logger.info(f"Sparse search completed: {len(formatted_results)} results in {search_time:.3f}s")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            raise Exception(f"Sparse search error: {e}")
    
    def hybrid_search(self, query: str, limit: int = 10, company_filter: Optional[str] = None) -> List[Dict]:
        """Perform intelligent hybrid search combining dense and sparse methods."""
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        if limit <= 0 or limit > 100:
            raise ValueError("Limit must be between 1 and 100")
        
        try:
            logger.info(f"Executing hybrid search: '{query[:50]}...' (limit={limit})")
            start_time = time.time()
            
            # Get results from both search methods
            search_limit = min(limit * 2, 50)
            
            dense_results = self.dense_search(query, search_limit, company_filter)
            sparse_results = self.sparse_search(query, search_limit, company_filter)
            
            # Combine and deduplicate results
            combined_results = {}
            
            # Process dense search results
            for rank, result in enumerate(dense_results):
                doc_id = result['id']
                
                # Convert distance to similarity score
                dense_similarity = 1.0 - result['similarity_score']
                dense_rank_score = 1.0 / (rank + 1)
                
                combined_results[doc_id] = {
                    **result,
                    'dense_score': dense_similarity,
                    'dense_rank': dense_rank_score,
                    'sparse_score': 0.0,
                    'sparse_rank': 0.0
                }
            
            # Process sparse search results
            for rank, result in enumerate(sparse_results):
                doc_id = result['id']
                
                sparse_score = result['similarity_score'] or 0.0
                sparse_rank_score = 1.0 / (rank + 1)
                
                if doc_id in combined_results:
                    combined_results[doc_id]['sparse_score'] = sparse_score
                    combined_results[doc_id]['sparse_rank'] = sparse_rank_score
                else:
                    combined_results[doc_id] = {
                        **result,
                        'dense_score': 0.0,
                        'dense_rank': 0.0,
                        'sparse_score': sparse_score,
                        'sparse_rank': sparse_rank_score
                    }
            
            # Calculate weighted hybrid scores
            dense_weight = SEARCH_CONFIG['hybrid_weights']['dense']
            sparse_weight = SEARCH_CONFIG['hybrid_weights']['sparse']
            
            for doc_id, result in combined_results.items():
                hybrid_score = (
                    dense_weight * result['dense_rank'] + 
                    sparse_weight * result['sparse_rank']
                )
                
                result['hybrid_score'] = hybrid_score
                result['search_time'] = time.time() - start_time
                result['search_type'] = 'hybrid'
            
            # Sort by hybrid score and return top results
            sorted_results = sorted(
                combined_results.values(), 
                key=lambda x: x['hybrid_score'], 
                reverse=True
            )
            
            final_results = sorted_results[:limit]
            
            logger.info(f"Hybrid search completed: {len(final_results)} results in {time.time() - start_time:.3f}s")
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise Exception(f"Hybrid search error: {e}")
    
    def search(self, query: str, search_type: str = 'hybrid', limit: int = 10, 
               company_filter: Optional[str] = None) -> List[Dict]:
        """
        Main search interface supporting all search methods.
        
        Provides unified access to dense, sparse, and hybrid search with
        automatic connection recovery and comprehensive error handling.
        
        Args:
            query: Search query string
            search_type: Type of search ('dense', 'sparse', 'hybrid')
            limit: Maximum number of results to return
            company_filter: Optional company code filter
            
        Returns:
            List of search results with metadata
            
        Raises:
            ValueError: If search_type is invalid or parameters are invalid
        """
        # Validate inputs
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        if search_type not in ['dense', 'sparse', 'hybrid']:
            raise ValueError("search_type must be 'dense', 'sparse', or 'hybrid'")
        
        if limit <= 0 or limit > 100:
            raise ValueError("Limit must be between 1 and 100")
        
        try:
            # Route to appropriate search method
            if search_type == 'dense':
                return self.dense_search(query, limit, company_filter)
            elif search_type == 'sparse':
                return self.sparse_search(query, limit, company_filter)
            elif search_type == 'hybrid':
                return self.hybrid_search(query, limit, company_filter)
                
        except Exception as e:
            logger.warning(f"Search failed, attempting reconnection: {e}")
            
            # Attempt reconnection and retry
            try:
                self._reconnect()
                
                if search_type == 'dense':
                    return self.dense_search(query, limit, company_filter)
                elif search_type == 'sparse':
                    return self.sparse_search(query, limit, company_filter)
                elif search_type == 'hybrid':
                    return self.hybrid_search(query, limit, company_filter)
                    
            except Exception as retry_error:
                logger.error(f"Search retry failed: {retry_error}")
                return []
    
    def get_companies(self) -> List[str]:
        """Retrieve list of available companies in the database."""
        try:
            self.cursor.execute("SELECT DISTINCT company FROM documents ORDER BY company")
            companies = [row[0] for row in self.cursor.fetchall()]
            return companies
        except Exception as e:
            logger.error(f"Failed to get companies: {e}")
            return []
    
    def get_search_stats(self) -> Dict[str, Union[int, str]]:
        """Retrieve database statistics for monitoring and reporting."""
        try:
            query = """
                SELECT 
                    COUNT(*) as total_documents,
                    COUNT(DISTINCT company) as companies,
                    COUNT(DISTINCT document_type) as document_types,
                    MIN(year) as earliest_year,
                    MAX(year) as latest_year
                FROM documents
            """
            
            self.cursor.execute(query)
            stats = dict(self.cursor.fetchone())
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database statistics: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Union[bool, str, int]]:
        """Perform comprehensive health check of the search system."""
        health_status = {
            'database_connected': False,
            'model_loaded': False,
            'total_documents': 0,
            'status': 'unhealthy'
        }
        
        try:
            # Check model
            if self.model is not None:
                health_status['model_loaded'] = True
            
            # Check database connection
            if self.conn and not self.conn.closed:
                health_status['database_connected'] = True
                
                # Get document count
                stats = self.get_search_stats()
                health_status['total_documents'] = stats.get('total_documents', 0)
            
            # Overall status
            if health_status['database_connected'] and health_status['model_loaded']:
                health_status['status'] = 'healthy'
            
            logger.info(f"Health check completed: {health_status['status']}")
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status['error'] = str(e)
            return health_status
    
    def close(self) -> None:
        """Properly close database connections and clean up resources."""
        try:
            logger.info("Closing HybridSearchEngine connections")
            
            if self.cursor:
                self.cursor.close()
                self.cursor = None
                
            if self.conn:
                self.conn.close()
                self.conn = None
                
            logger.info("HybridSearchEngine closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.close()

__version__ = "1.0.0"
__author__ = "Divyanshu Chaudhary"
