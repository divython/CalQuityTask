"""
Calquity - Professional Financial Document Search System
======================================================

A production-ready hybrid search system for financial document analysis.
Combines dense vector search with sparse keyword search for comprehensive
document retrieval across financial reports, earnings calls, and SEC filings.

Author: Divyanshu Chaudhary
Version: 1.0.0
License: Proprietary
"""

__version__ = "1.0.0"
__author__ = "Divyanshu Chaudhary"
__email__ = "divyanshu.chaudhary@company.com"
__status__ = "Production"
__copyright__ = "Copyright 2025, Divyanshu Chaudhary"

# Import main components for easy access
from .search_engine import HybridSearchEngine
from .config_manager import ConfigManager
from .document_processor import DocumentProcessor

# Export main classes
__all__ = [
    'HybridSearchEngine',
    'ConfigManager', 
    'DocumentProcessor'
]
