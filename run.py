#!/usr/bin/env python3
"""
Calquity System Launcher
=======================


Provides command-line interface for testing, web interface, and system validation.

Author: Divyanshu Chaudhary
Version: 1.0.0
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main entry point for Calquity system."""
    parser = argparse.ArgumentParser(description='Calquity Financial Document Search System')
    parser.add_argument('--test', action='store_true', help='Run system tests')
    parser.add_argument('--web', action='store_true', help='Launch web interface')
    parser.add_argument('--analyze', action='store_true', help='Run performance analysis')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CALQUITY - FINANCIAL DOCUMENT SEARCH SYSTEM")
    print("Author: Divyanshu Chaudhary")
    print("Version: 1.0.0")
    print("=" * 60)
    
    if args.test:
        print("Running system tests...")
        os.system("python tests/test_system.py")
    
    elif args.web:
        print("Launching web interface...")
        os.chdir("web")
        os.system("streamlit run app.py")
    
    elif args.analyze:
        print("Running performance analysis...")
        os.system("python scripts/performance_analysis.py")
    
    elif args.demo:
        print("Running demonstration...")
        try:
            from calquity import HybridSearchEngine
            
            with HybridSearchEngine() as engine:
                print("✅ System initialized successfully")
                
                # Demo search
                results = engine.search("revenue growth strategy", limit=3)
                print(f"📊 Found {len(results)} results")
                
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result['company']} - {result['document_type']}")
                    print(f"   Score: {result.get('hybrid_score', 'N/A')}")
                    print(f"   Preview: {result['content_preview'][:100]}...")
                
        except Exception as e:
            print(f"❌ Demo failed: {e}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
