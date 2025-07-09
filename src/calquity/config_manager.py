"""
Calquity - Professional Configuration Management System
=====================================================

Enterprise-grade configuration management for the Calquity hybrid search system.
Handles database connections, embedding models, search parameters, and system settings.

Author: Divyanshu Chaudhary
Version: 1.0.0
Created: 2025
License: Proprietary
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = "localhost"
    port: int = 5432
    database: str = "calquitytask"
    user: str = "postgres"
    password: str = "root"
    pool_size: int = 20
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    def get_connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    def get_connection_dict(self) -> Dict[str, Any]:
        """Get database connection parameters as dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password
        }

@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    dense_model: str = "all-MiniLM-L6-v2"
    vector_dimension: int = 384
    batch_size: int = 32
    max_seq_length: int = 512
    normalize_embeddings: bool = True
    device: str = "cpu"
    
    def __post_init__(self):
        """Validate embedding configuration."""
        if self.vector_dimension <= 0:
            raise ValueError("Vector dimension must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.max_seq_length <= 0:
            raise ValueError("Max sequence length must be positive")

@dataclass
class SearchConfig:
    """Search algorithm configuration."""
    hybrid_weights: Dict[str, float] = field(default_factory=lambda: {
        "dense": 0.7,
        "sparse": 0.3
    })
    max_results: int = 10
    similarity_threshold: float = 0.1
    sparse_rank_threshold: float = 0.01
    rerank_top_k: int = 100
    
    def __post_init__(self):
        """Validate search configuration."""
        if not abs(sum(self.hybrid_weights.values()) - 1.0) < 1e-6:
            raise ValueError("Hybrid weights must sum to 1.0")
        if self.max_results <= 0:
            raise ValueError("Max results must be positive")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")

@dataclass
class SystemConfig:
    """System-wide configuration."""
    log_level: str = "INFO"
    debug_mode: bool = False
    performance_monitoring: bool = True
    cache_enabled: bool = True
    cache_ttl: int = 3600
    max_query_length: int = 1000
    request_timeout: int = 30
    
    def configure_logging(self):
        """Configure system logging."""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

class ConfigManager:
    """
    Central configuration management system for Calquity.
    
    Provides centralized access to all configuration settings with validation,
    environment variable support, and runtime configuration updates.
    """
    
    def __init__(self):
        """Initialize configuration manager."""
        self.database = DatabaseConfig()
        self.embedding = EmbeddingConfig()
        self.search = SearchConfig()
        self.system = SystemConfig()
        
        # Load environment overrides
        self._load_from_environment()
        
        # Configure logging
        self.system.configure_logging()
        
        logger.info("Configuration manager initialized")
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # Database configuration
        self.database.host = os.getenv("DB_HOST", self.database.host)
        self.database.port = int(os.getenv("DB_PORT", str(self.database.port)))
        self.database.database = os.getenv("DB_NAME", self.database.database)
        self.database.user = os.getenv("DB_USER", self.database.user)
        self.database.password = os.getenv("DB_PASSWORD", self.database.password)
        
        # Embedding configuration
        self.embedding.dense_model = os.getenv("EMBEDDING_MODEL", self.embedding.dense_model)
        self.embedding.device = os.getenv("EMBEDDING_DEVICE", self.embedding.device)
        
        # System configuration
        self.system.log_level = os.getenv("LOG_LEVEL", self.system.log_level)
        self.system.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    def validate_all(self) -> bool:
        """Validate all configuration settings."""
        try:
            # Validate database connection
            import psycopg2
            conn = psycopg2.connect(**self.database.get_connection_dict())
            conn.close()
            
            # Validate embedding model
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.embedding.dense_model)
            
            logger.info("All configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration."""
        return {
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "user": self.database.user
            },
            "embedding": {
                "model": self.embedding.dense_model,
                "dimension": self.embedding.vector_dimension,
                "device": self.embedding.device
            },
            "search": {
                "weights": self.search.hybrid_weights,
                "max_results": self.search.max_results
            },
            "system": {
                "log_level": self.system.log_level,
                "debug_mode": self.system.debug_mode,
                "performance_monitoring": self.system.performance_monitoring
            }
        }

# Global configuration instance
config = ConfigManager()

# Backward compatibility - Legacy configuration dictionaries
DATABASE_CONFIG = config.database.get_connection_dict()
EMBEDDING_CONFIG = {
    "dense_model": config.embedding.dense_model,
    "vector_dimension": config.embedding.vector_dimension
}
SEARCH_CONFIG = {
    "hybrid_weights": config.search.hybrid_weights
}

# Module metadata
__author__ = "Divyanshu Chaudhary"
__version__ = "1.0.0"
__email__ = "divyanshu.chaudhary@example.com"
__status__ = "Production"
__copyright__ = "Copyright 2025, Divyanshu Chaudhary"
