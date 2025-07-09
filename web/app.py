"""
Calquity - Professional Streamlit Web Application
===============================================

Interactive web interface for the Calquity hybrid search system.
Provides real-time search capabilities across financial documents with
comprehensive analytics and visualization.

Author: Divyanshu Chaudhary
Version: 1.0.0
Created: 2025
License: Proprietary
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from calquity import HybridSearchEngine
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module metadata
__author__ = "Divyanshu Chaudhary"
__version__ = "1.0.0"
__status__ = "Production"

# Configure page
st.set_page_config(
    page_title="Calquity - Financial Document Hybrid Search",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .search-result {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 10px 0;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .company-tag {
        background: #e3f2fd;
        color: #1565c0;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .doc-type-tag {
        background: #f3e5f5;
        color: #7b1fa2;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class CalquityApp:
    """
    Main Calquity Streamlit application class.
    
    Provides a professional web interface for financial document search
    with comprehensive analytics and visualization capabilities.
    """
    
    def __init__(self):
        """Initialize the Calquity application."""
        self.search_engine = None
        self._initialize_search_engine()
    
    def _initialize_search_engine(self) -> None:
        """Initialize the hybrid search engine with error handling."""
        try:
            self.search_engine = HybridSearchEngine()
            logger.info("Search engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize search engine: {e}")
            st.error(f"❌ Failed to initialize search engine: {e}")
            st.stop()
    
    def format_result(self, result: Dict[str, Any], rank: int) -> str:
        """
        Format a search result for display in the UI.
        
        Args:
            result: Search result dictionary
            rank: Result ranking position
            
        Returns:
            Formatted HTML string for display
        """
        company = result.get('company', 'Unknown')
        doc_type = result.get('document_type', 'Unknown')
        year = result.get('year', 'Unknown')
        content = result.get('content_preview', result.get('content', ''))
        
        # Truncate content if too long
        if len(content) > 500:
            content = content[:500] + "..."
        
        # Get score with fallback options
        score = result.get('similarity_score', 
                          result.get('combined_score', 
                          result.get('relevance_score', 
                          result.get('score', 0))))
        
        return f"""
        <div class="search-result">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div>
                    <span class="company-tag">{company}</span>
                    <span class="doc-type-tag">{doc_type}</span>
                    <span style="color: #666; font-size: 0.9em; margin-left: 10px;">Year: {year}</span>
                </div>
                <div style="text-align: right;">
                    <div style="font-weight: bold; color: #007bff;">Rank #{rank}</div>
                    <div style="color: #666; font-size: 0.9em;">Score: {score:.4f}</div>
                </div>
            </div>
            <div style="color: #333; line-height: 1.6;">
                {content}
            </div>
        </div>
        """
    
    def perform_comprehensive_search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Perform search using all available methods.
        
        Args:
            query: Search query string
            limit: Maximum number of results per method
            
        Returns:
            Dictionary containing results from all search methods
        """
        if not self.search_engine:
            st.error("Search engine not initialized")
            return {}
        
        results = {}
        
        try:
            # Dense search
            start_time = time.time()
            dense_results = self.search_engine.dense_search(query, limit=limit)
            dense_time = time.time() - start_time
            results['dense'] = {
                'results': dense_results,
                'time': dense_time,
                'name': 'Dense (Semantic)'
            }
            
            # Sparse search
            start_time = time.time()
            sparse_results = self.search_engine.sparse_search(query, limit=limit)
            sparse_time = time.time() - start_time
            results['sparse'] = {
                'results': sparse_results,
                'time': sparse_time,
                'name': 'Sparse (Keyword)'
            }
            
            # Hybrid search
            start_time = time.time()
            hybrid_results = self.search_engine.hybrid_search(query, limit=limit)
            hybrid_time = time.time() - start_time
            results['hybrid'] = {
                'results': hybrid_results,
                'time': hybrid_time,
                'name': 'Hybrid (Dense + Sparse)'
            }
            
            logger.info(f"Search completed for query: '{query[:50]}...'")
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            st.error(f"❌ Search failed: {e}")
            return {}
        
        return results


@st.cache_resource
def get_search_engine():
    """Initialize and cache the search engine (legacy function for compatibility)."""
    try:
        engine = HybridSearchEngine()
        return engine
    except Exception as e:
        st.error(f"Failed to initialize search engine: {e}")
        return None

def format_result(result, rank):
    """Format a search result for display"""
    company = result.get('company', 'Unknown')
    doc_type = result.get('document_type', 'Unknown')
    year = result.get('year', 'Unknown')
    content = result.get('content_preview', result.get('content', ''))[:500] + "..." if len(result.get('content_preview', result.get('content', ''))) > 500 else result.get('content_preview', result.get('content', ''))
    score = result.get('similarity_score', result.get('combined_score', result.get('relevance_score', result.get('score', 0))))
    
    return f"""
    <div class="search-result">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <div>
                <span class="company-tag">{company}</span>
                <span class="doc-type-tag">{doc_type}</span>
                <span style="color: #666; font-size: 0.9em; margin-left: 10px;">Year: {year}</span>
            </div>
            <div style="text-align: right;">
                <div style="font-weight: bold; color: #007bff;">Rank #{rank}</div>
                <div style="color: #666; font-size: 0.9em;">Score: {score:.4f}</div>
            </div>
        </div>
        <div style="color: #333; line-height: 1.6;">
            {content}
        </div>
    </div>
    """

def run_search_comparison(query, limit=10):
    """Run all three search methods and return results with timing"""
    search_engine = get_search_engine()
    if not search_engine:
        return {}
    
    results = {}
    
    try:
        # Dense search
        start_time = time.time()
        dense_results = search_engine.dense_search(query, limit=limit)
        dense_time = time.time() - start_time
        results['dense'] = {
            'results': dense_results,
            'time': dense_time,
            'name': 'Dense (Vector)'
        }
        
        # Sparse search
        start_time = time.time()
        sparse_results = search_engine.sparse_search(query, limit=limit)
        sparse_time = time.time() - start_time
        results['sparse'] = {
            'results': sparse_results,
            'time': sparse_time,
            'name': 'Sparse (Full-text)'
        }
        
        # Hybrid search
        start_time = time.time()
        hybrid_results = search_engine.hybrid_search(query, limit=limit)
        hybrid_time = time.time() - start_time
        results['hybrid'] = {
            'results': hybrid_results,
            'time': hybrid_time,
            'name': 'Hybrid (Dense + Sparse)'
        }
    except Exception as e:
        st.error(f"Search failed: {e}")
        return {}
    
    return results

def calculate_overlap_metrics(results_dict):
    """Calculate overlap metrics between different search methods"""
    dense_ids = [r['id'] for r in results_dict['dense']['results']]
    sparse_ids = [r['id'] for r in results_dict['sparse']['results']]
    hybrid_ids = [r['id'] for r in results_dict['hybrid']['results']]
    
    metrics = {}
    
    # Calculate overlap percentages (with zero-division protection)
    dense_sparse_overlap = (len(set(dense_ids) & set(sparse_ids)) / len(dense_ids) * 100) if len(dense_ids) > 0 else 0
    dense_hybrid_overlap = (len(set(dense_ids) & set(hybrid_ids)) / len(dense_ids) * 100) if len(dense_ids) > 0 else 0
    sparse_hybrid_overlap = (len(set(sparse_ids) & set(hybrid_ids)) / len(sparse_ids) * 100) if len(sparse_ids) > 0 else 0
    
    metrics['overlaps'] = {
        'Dense ∩ Sparse': dense_sparse_overlap,
        'Dense ∩ Hybrid': dense_hybrid_overlap,
        'Sparse ∩ Hybrid': sparse_hybrid_overlap
    }
    
    # Calculate unique results
    all_unique = len(set(dense_ids) | set(sparse_ids) | set(hybrid_ids))
    metrics['unique_total'] = all_unique
    
    return metrics

def create_performance_chart(results_dict):
    """Create a performance comparison chart"""
    methods = [results_dict[k]['name'] for k in results_dict.keys()]
    times = [results_dict[k]['time'] * 1000 for k in results_dict.keys()]  # Convert to ms
    result_counts = [len(results_dict[k]['results']) for k in results_dict.keys()]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Search Time (ms)', 'Results Returned'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Time chart
    fig.add_trace(
        go.Bar(x=methods, y=times, name='Time (ms)', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Results count chart
    fig.add_trace(
        go.Bar(x=methods, y=result_counts, name='Results', marker_color='lightgreen'),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Search Performance Comparison",
        showlegend=False,
        height=400
    )
    
    return fig

def create_overlap_chart(overlap_metrics):
    """Create an overlap visualization chart"""
    overlaps = overlap_metrics['overlaps']
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(overlaps.keys()),
            y=list(overlaps.values()),
            marker_color=['#ff9999', '#66b3ff', '#99ff99'],
            text=[f'{v:.1f}%' for v in overlaps.values()],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Result Overlap Between Search Methods',
        xaxis_title='Method Pairs',
        yaxis_title='Overlap Percentage (%)',
        yaxis=dict(range=[0, 100]),
        height=400
    )
    
    return fig

def main():
    # Header
    st.title("📊 Calquity - Financial Document Hybrid Search")
    st.markdown("Compare dense vector search, sparse full-text search, and hybrid search on financial documents")
    
    # Database status check
    search_engine = get_search_engine()
    if search_engine:
        stats = search_engine.get_search_stats()
        if stats:
            st.success(f"✅ Connected to database - {stats.get('total_documents', 0)} documents from {stats.get('companies', 0)} companies ({stats.get('earliest_year', 'N/A')}-{stats.get('latest_year', 'N/A')})")
        else:
            st.warning("⚠️ Connected but unable to retrieve database statistics")
    else:
        st.error("❌ Unable to connect to database")
        return
    
    # Sidebar configuration
    st.sidebar.header("🔧 Search Configuration")
    
    # Sample queries
    st.sidebar.subheader("📝 Sample Queries")
    sample_queries = [
        "revenue growth and financial performance",
        "artificial intelligence and machine learning investments",
        "supply chain challenges and inflation impact",
        "electric vehicle market and competition",
        "cloud computing and digital transformation",
        "regulatory compliance and legal risks",
        "dividend policy and shareholder returns",
        "market share and competitive positioning"
    ]
    
    selected_sample = st.sidebar.selectbox(
        "Choose a sample query:",
        [""] + sample_queries
    )
    
    # Search parameters
    limit = st.sidebar.slider("Number of results per method", 5, 20, 10)
    
    # Main search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "🔍 Enter your search query:",
            value=selected_sample,
            placeholder="e.g., revenue growth and financial performance"
        )
    
    with col2:
        search_button = st.button("🚀 Search", type="primary", use_container_width=True)
    
    # Initialize session state
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    # Perform search
    if search_button and query.strip():
        with st.spinner("🔄 Running hybrid search comparison..."):
            try:
                results = run_search_comparison(query, limit)
                overlap_metrics = calculate_overlap_metrics(results)
                
                st.session_state.search_results = {
                    'query': query,
                    'results': results,
                    'overlap_metrics': overlap_metrics,
                    'timestamp': datetime.now()
                }
                
                # Add to search history
                st.session_state.search_history.append({
                    'query': query,
                    'timestamp': datetime.now(),
                    'total_time': sum(r['time'] for r in results.values())
                })
                
                st.success("✅ Search completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Search failed: {str(e)}")
    
    # Display results
    if st.session_state.search_results:
        results_data = st.session_state.search_results
        query = results_data['query']
        results = results_data['results']
        overlap_metrics = results_data['overlap_metrics']
        
        st.markdown("---")
        st.subheader(f"🎯 Results for: \"{query}\"")
        
        # Performance metrics
        st.subheader("📊 Performance Metrics")
        
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            total_time = sum(r['time'] for r in results.values()) * 1000
            st.metric("Total Search Time", f"{total_time:.1f} ms")
        
        with metric_cols[1]:
            fastest = min(results.values(), key=lambda x: x['time'])
            st.metric("Fastest Method", fastest['name'], f"{fastest['time']*1000:.1f} ms")
        
        with metric_cols[2]:
            unique_results = overlap_metrics['unique_total']
            st.metric("Unique Results Found", unique_results)
        
        with metric_cols[3]:
            avg_overlap = np.mean(list(overlap_metrics['overlaps'].values()))
            st.metric("Avg Method Overlap", f"{avg_overlap:.1f}%")
        
        # Charts
        chart_cols = st.columns(2)
        
        with chart_cols[0]:
            perf_chart = create_performance_chart(results)
            st.plotly_chart(perf_chart, use_container_width=True)
        
        with chart_cols[1]:
            overlap_chart = create_overlap_chart(overlap_metrics)
            st.plotly_chart(overlap_chart, use_container_width=True)
        
        # Method comparison tabs
        st.subheader("🔍 Search Results Comparison")
        
        tabs = st.tabs([f"🎯 {results[k]['name']}" for k in ['dense', 'sparse', 'hybrid']])
        
        for i, method in enumerate(['dense', 'sparse', 'hybrid']):
            with tabs[i]:
                method_results = results[method]['results']
                method_time = results[method]['time'] * 1000
                
                if method_results:
                    st.info(f"⏱️ Search completed in {method_time:.1f} ms | Found {len(method_results)} results")
                    
                    for rank, result in enumerate(method_results, 1):
                        st.markdown(format_result(result, rank), unsafe_allow_html=True)
                else:
                    st.warning("No results found for this search method.")
    
    # Search history sidebar
    if st.session_state.search_history:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📚 Recent Searches")
        
        for i, search in enumerate(reversed(st.session_state.search_history[-5:])):  # Show last 5
            with st.sidebar.expander(f"🕒 {search['timestamp'].strftime('%H:%M:%S')}"):
                st.write(f"**Query:** {search['query']}")
                st.write(f"**Time:** {search['total_time']*1000:.1f} ms")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        Built with Streamlit • PostgreSQL + PGVector • Sentence Transformers
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
