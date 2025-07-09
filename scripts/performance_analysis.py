"""
Calquity - Comprehensive Search Performance Analysis System
========================================================

Professional experimental analysis framework for comparing hybrid search performance.
Conducts systematic evaluation of dense vector search, sparse full-text search, 
and hybrid search combinations across real financial documents.

Key Features:
- Automated performance benchmarking
- Statistical analysis and reporting
- Result quality assessment
- Comparative methodology evaluation

Author: Divyanshu Chaudhary
Version: 1.0.0
Created: 2025
License: Proprietary
"""

import time
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from calquity import HybridSearchEngine
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module metadata
__author__ = "Divyanshu Chaudhary"
__version__ = "1.0.0"
__status__ = "Production"
__copyright__ = "Copyright 2025, Divyanshu Chaudhary"


class SearchExperiment:
    """
    Professional search performance analysis framework.
    
    Conducts comprehensive experiments to evaluate and compare:
    1. Dense vector search (semantic similarity)
    2. Sparse full-text search (keyword matching) 
    3. Hybrid search (optimized combination)
    
    Metrics evaluated:
    - Query response time and performance characteristics
    - Result relevance and quality assessment
    - Method overlap and complementarity analysis
    - Search method strengths and limitation identification
    """
    
    def __init__(self):
        """Initialize the search experiment framework."""
        try:
            self.search_engine = HybridSearchEngine()
            logger.info("Search experiment framework initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize search engine: {e}")
            raise
        
        # Professional test query set covering key financial analysis areas
        self.test_queries = [
            # Financial performance and metrics
            "revenue growth and quarterly earnings performance",
            "profit margins and cost reduction strategies",
            "dividend policy and shareholder returns analysis",
            "cash flow management and liquidity position",
            "debt levels and capital structure optimization",
            
            # Technology and innovation investment
            "artificial intelligence and machine learning investments",
            "cloud computing and digital transformation initiatives",
            "research and development spending allocation",
            "technology infrastructure modernization projects",
            "cybersecurity and data protection measures",
            
            # Market dynamics and competition
            "market share and competitive positioning",
            "customer acquisition and retention strategies",
            "pricing strategy and market dynamics",
            "supply chain optimization and efficiency",
            "geographic expansion and market penetration",
            
            # Risk management and compliance
            "regulatory compliance and legal risk factors",
            "environmental sustainability and ESG initiatives",
            "operational risk management procedures",
            "financial risk and credit exposure",
            "business continuity and crisis management",
            
            # Strategic and operational focus
            "merger and acquisition activities",
            "strategic partnerships and alliances",
            "workforce development and talent acquisition",
            "operational efficiency and productivity improvements",
            "future outlook and growth projections"
        ]
        
        # Results storage
        self.experiment_results = defaultdict(list)
        self.performance_metrics = {}
        self.overlap_analysis = {}
        
    def run_single_query_experiment(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Execute comprehensive search experiment for a single query.
        
        Args:
            query: Search query string
            limit: Maximum number of results per method
            
        Returns:
            Dictionary containing timing, results, and analysis for all methods
        """
        experiment_result = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'methods': {}
        }
        
        logger.info(f"Running experiment for query: '{query[:50]}...'")
        
        # Test each search method with comprehensive timing
        search_methods = [
            ('dense', self.search_engine.dense_search),
            ('sparse', self.search_engine.sparse_search),
            ('hybrid', self.search_engine.hybrid_search)
        ]
        
        for method_name, method_func in search_methods:
            
            # Measure time
            start_time = time.time()
            results = method_func(query, limit=limit)
            end_time = time.time()
            
            search_time = end_time - start_time
            
            # Extract result IDs and scores
            result_ids = [r['id'] for r in results]
            result_scores = [r.get('similarity_score', r.get('score', 0)) for r in results]
            
            # Company and document type distribution
            companies = [r.get('company', 'Unknown') for r in results]
            doc_types = [r.get('document_type', 'Unknown') for r in results]
            
            company_dist = pd.Series(companies).value_counts().to_dict()
            doc_type_dist = pd.Series(doc_types).value_counts().to_dict()
            
            experiment_result['methods'][method_name] = {
                'time_ms': search_time * 1000,
                'num_results': len(results),
                'result_ids': result_ids,
                'scores': result_scores,
                'avg_score': np.mean(result_scores) if result_scores else 0,
                'score_std': np.std(result_scores) if result_scores else 0,
                'company_distribution': company_dist,
                'doc_type_distribution': doc_type_dist,
                'top_3_results': results[:3]  # Store top 3 for manual inspection
            }
        
        # Calculate overlap metrics
        dense_ids = set(experiment_result['methods']['dense']['result_ids'])
        sparse_ids = set(experiment_result['methods']['sparse']['result_ids'])
        hybrid_ids = set(experiment_result['methods']['hybrid']['result_ids'])
        
        experiment_result['overlap_metrics'] = {
            'dense_sparse_overlap': len(dense_ids & sparse_ids),
            'dense_hybrid_overlap': len(dense_ids & hybrid_ids),
            'sparse_hybrid_overlap': len(sparse_ids & hybrid_ids),
            'total_unique_results': len(dense_ids | sparse_ids | hybrid_ids),
            'overlap_percentages': {
                'dense_sparse': len(dense_ids & sparse_ids) / len(dense_ids) * 100 if dense_ids else 0,
                'dense_hybrid': len(dense_ids & hybrid_ids) / len(dense_ids) * 100 if dense_ids else 0,
                'sparse_hybrid': len(sparse_ids & hybrid_ids) / len(sparse_ids) * 100 if sparse_ids else 0
            }
        }
        
        return experiment_result
    
    def run_all_experiments(self, limit=10):
        """Run experiments for all test queries"""
        for i, query in enumerate(self.test_queries, 1):
            try:
                result = self.run_single_experiment(query, limit)
                self.results.append(result)
                self.detailed_results[query] = result
            except Exception as e:
                continue
    
    def analyze_performance(self):
        """Analyze performance metrics across all experiments"""
        # Aggregate timing data
        timing_data = {
            'dense': [],
            'sparse': [],
            'hybrid': []
        }
        
        # Aggregate overlap data
        overlap_data = {
            'dense_sparse': [],
            'dense_hybrid': [],
            'sparse_hybrid': []
        }
        
        # Aggregate result count and score data
        result_data = {
            'dense': {'counts': [], 'avg_scores': [], 'score_stds': []},
            'sparse': {'counts': [], 'avg_scores': [], 'score_stds': []},
            'hybrid': {'counts': [], 'avg_scores': [], 'score_stds': []}
        }
        
        for result in self.results:
            # Timing data
            for method in ['dense', 'sparse', 'hybrid']:
                timing_data[method].append(result['methods'][method]['time_ms'])
                result_data[method]['counts'].append(result['methods'][method]['num_results'])
                result_data[method]['avg_scores'].append(result['methods'][method]['avg_score'])
                result_data[method]['score_stds'].append(result['methods'][method]['score_std'])
            
            # Overlap data
            for overlap_type in ['dense_sparse', 'dense_hybrid', 'sparse_hybrid']:
                overlap_data[overlap_type].append(result['overlap_metrics']['overlap_percentages'][overlap_type])
        
        # Calculate summary statistics
        performance_summary = {
            'timing_stats': {},
            'overlap_stats': {},
            'result_stats': {}
        }
        
        # Timing statistics
        for method in ['dense', 'sparse', 'hybrid']:
            times = timing_data[method]
            if times:  # Only calculate if we have data
                performance_summary['timing_stats'][method] = {
                    'mean_ms': np.mean(times),
                    'median_ms': np.median(times),
                    'std_ms': np.std(times),
                    'min_ms': np.min(times),
                    'max_ms': np.max(times)
                }
            else:
                performance_summary['timing_stats'][method] = {
                    'mean_ms': 0,
                    'median_ms': 0,
                    'std_ms': 0,
                    'min_ms': 0,
                    'max_ms': 0
                }
        
        # Overlap statistics
        for overlap_type in ['dense_sparse', 'dense_hybrid', 'sparse_hybrid']:
            overlaps = overlap_data[overlap_type]
            if overlaps:  # Only calculate if we have data
                performance_summary['overlap_stats'][overlap_type] = {
                    'mean_percent': np.mean(overlaps),
                    'median_percent': np.median(overlaps),
                    'std_percent': np.std(overlaps),
                    'min_percent': np.min(overlaps),
                    'max_percent': np.max(overlaps)
                }
            else:
                performance_summary['overlap_stats'][overlap_type] = {
                    'mean_percent': 0,
                    'median_percent': 0,
                    'std_percent': 0,
                    'min_percent': 0,
                    'max_percent': 0
                }
        
        # Result statistics
        for method in ['dense', 'sparse', 'hybrid']:
            counts = result_data[method]['counts']
            avg_scores = result_data[method]['avg_scores']
            performance_summary['result_stats'][method] = {
                'avg_result_count': np.mean(counts),
                'avg_score_mean': np.mean(avg_scores),
                'avg_score_std': np.mean(result_data[method]['score_stds'])
            }
        
        return performance_summary
    
    def generate_insights(self, performance_summary):
        """Generate insights and recommendations"""
        insights = {
            'performance_insights': [],
            'overlap_insights': [],
            'recommendation': '',
            'use_cases': {}
        }
        
        # Performance insights
        timing_stats = performance_summary['timing_stats']
        if any(timing_stats[method]['mean_ms'] > 0 for method in timing_stats):
            fastest_method = min(timing_stats.keys(), key=lambda x: timing_stats[x]['mean_ms'] if timing_stats[x]['mean_ms'] > 0 else float('inf'))
            slowest_method = max(timing_stats.keys(), key=lambda x: timing_stats[x]['mean_ms'])
            
            insights['performance_insights'].append(
                f"Fastest method: {fastest_method.title()} ({timing_stats[fastest_method]['mean_ms']:.1f}ms avg)"
            )
            insights['performance_insights'].append(
                f"Slowest method: {slowest_method.title()} ({timing_stats[slowest_method]['mean_ms']:.1f}ms avg)"
            )
            
            # Speed difference analysis
            if timing_stats[fastest_method]['mean_ms'] > 0:
                speed_diff = timing_stats[slowest_method]['mean_ms'] - timing_stats[fastest_method]['mean_ms']
                speed_percent = speed_diff / timing_stats[fastest_method]['mean_ms'] * 100
                insights['performance_insights'].append(
                    f"Speed difference: {speed_diff:.1f}ms ({speed_percent:.1f}% slower)"
                )
        else:
            insights['performance_insights'].append("No timing data available")
        
        # Overlap insights
        overlap_stats = performance_summary['overlap_stats']
        highest_overlap = max(overlap_stats.keys(), key=lambda x: overlap_stats[x]['mean_percent'])
        lowest_overlap = min(overlap_stats.keys(), key=lambda x: overlap_stats[x]['mean_percent'])
        
        insights['overlap_insights'].append(
            f"Highest overlap: {highest_overlap.replace('_', ' & ').title()} ({overlap_stats[highest_overlap]['mean_percent']:.1f}%)"
        )
        insights['overlap_insights'].append(
            f"Lowest overlap: {lowest_overlap.replace('_', ' & ').title()} ({overlap_stats[lowest_overlap]['mean_percent']:.1f}%)"
        )
        
        # Generate recommendations
        if timing_stats['hybrid']['mean_ms'] < timing_stats['dense']['mean_ms'] + timing_stats['sparse']['mean_ms']:
            insights['recommendation'] = "Hybrid search is efficient and combines benefits of both methods"
        elif abs(timing_stats['dense']['mean_ms'] - timing_stats['sparse']['mean_ms']) < 10:
            insights['recommendation'] = "Dense and sparse methods have similar performance; hybrid provides best coverage"
        else:
            insights['recommendation'] = f"Consider {fastest_method} for speed-critical applications, hybrid for comprehensive results"
        
        # Use case recommendations
        insights['use_cases'] = {
            'dense': 'Best for semantic similarity and concept-based queries',
            'sparse': 'Best for exact keyword matching and specific term searches',
            'hybrid': 'Best for comprehensive search combining semantic and keyword relevance'
        }
        
        return insights
    
    def save_results(self, filename_prefix="search_experiment"):
        """Save experiment results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = f"{filename_prefix}_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.detailed_results, f, indent=2, default=str)
        
        # Save summary data
        summary_file = f"{filename_prefix}_summary_{timestamp}.json"
        performance_summary = self.analyze_performance()
        insights = self.generate_insights(performance_summary)
        
        summary_data = {
            'experiment_info': {
                'total_queries': len(self.test_queries),
                'total_experiments': len(self.results),
                'timestamp': timestamp,
                'limit': 10
            },
            'performance_summary': performance_summary,
            'insights': insights,
            'test_queries': self.test_queries
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        return results_file, summary_file
    
    def print_summary(self):
        """Print a comprehensive summary of results"""
        if not self.results:
            print("❌ No experiment results available")
            return
        
        performance_summary = self.analyze_performance()
        insights = self.generate_insights(performance_summary)
        
        print("\n" + "="*60)
        print("📊 HYBRID SEARCH EXPERIMENT SUMMARY")
        print("="*60)
        
        print(f"\n📈 Experiment Overview:")
        print(f"  • Total queries tested: {len(self.test_queries)}")
        print(f"  • Successful experiments: {len(self.results)}")
        print(f"  • Results per query: 10")
        
        print(f"\n⏱️  Performance Statistics:")
        for method in ['dense', 'sparse', 'hybrid']:
            stats = performance_summary['timing_stats'][method]
            print(f"  • {method.title()} search: {stats['mean_ms']:.1f}ms avg (±{stats['std_ms']:.1f}ms)")
        
        print(f"\n🔄 Result Overlap Analysis:")
        for overlap_type in ['dense_sparse', 'dense_hybrid', 'sparse_hybrid']:
            stats = performance_summary['overlap_stats'][overlap_type]
            method_pair = overlap_type.replace('_', ' & ').title()
            print(f"  • {method_pair}: {stats['mean_percent']:.1f}% avg overlap")
        
        print(f"\n💡 Key Insights:")
        for insight in insights['performance_insights']:
            print(f"  • {insight}")
        
        print(f"\n🎯 Recommendations:")
        print(f"  • {insights['recommendation']}")
        
        print(f"\n📋 Use Case Guidelines:")
        for method, use_case in insights['use_cases'].items():
            print(f"  • {method.title()}: {use_case}")
        
        print("\n" + "="*60)

def main():
    """Main execution function"""
    # Initialize experiment
    experiment = SearchExperiment()
    
    # Run experiments
    experiment.run_all_experiments(limit=10)
    
    # Print summary
    experiment.print_summary()
    
    # Save results
    experiment.save_results()

if __name__ == "__main__":
    main()
