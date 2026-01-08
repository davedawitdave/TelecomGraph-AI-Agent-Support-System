#!/usr/bin/env python3
"""
Simple launcher for RAG evaluation system
Usage: python run_eval.py
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import and run the evaluation
from evaluation.run_eval import main

if __name__ == "__main__":
    main()
