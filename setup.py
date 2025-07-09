#!/usr/bin/env python3
"""
Calquity Setup Script
====================

Professional setup and installation script for the Calquity system.
Handles environment setup, dependency installation, and system validation.

Author: Divyanshu Chaudhary
Version: 1.0.0
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a system command with error handling."""
    print(f"📋 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup routine."""
    print("=" * 60)
    print("CALQUITY SETUP - PROFESSIONAL INSTALLATION")
    print("Author: Divyanshu Chaudhary")
    print("Version: 1.0.0")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r config/requirements.txt", "Installing dependencies"):
        sys.exit(1)
    
    # Validate installation
    print("🔍 Validating installation...")
    
    # Add src to path for testing
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from calquity import HybridSearchEngine, ConfigManager
        print("✅ Core modules imported successfully")
        
        # Test configuration
        config = ConfigManager()
        print("✅ Configuration manager initialized")
        
        print("\n🚀 INSTALLATION COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Set up PostgreSQL database with PGVector extension")
        print("2. Configure environment variables (see docs/INSTALLATION.md)")
        print("3. Run system tests: python run.py --test")
        print("4. Launch web interface: python run.py --web")
        
    except Exception as e:
        print(f"❌ Installation validation failed: {e}")
        print("Please check the error messages above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
