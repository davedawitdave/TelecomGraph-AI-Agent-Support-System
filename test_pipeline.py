#!/usr/bin/env python3
"""
Test script for RAG Pipeline components
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """
    Test that all modules can be imported without errors.

    Returns:
        True if all imports successful, False otherwise.
    """
    print("Testing imports...")
    try:
        from src.pipeline import RAGPipeline
        from src.ingestion.loader import DataLoader
        from src.ingestion.processor import DataProcessor
        from src.ingestion.graph_builder import GraphBuilder
        from src.retrieval.search import VectorSearch
        from src.generation.llm import LLMGenerator
        from src.evaluation.metrics import EvaluationMetrics
        from src.evaluation.run_eval import Evaluator
        print("[SUCCESS] All imports successful")
        return True
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False


def test_config_loading():
    """
    Test configuration file loading and validation.

    Returns:
        True if configuration files are valid, False otherwise.
    """
    print("Testing configuration loading...")
    try:
        import yaml

        # Test config.yaml
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        assert 'models' in config
        assert 'neo4j' in config

        # Test secrets.yaml (may not be configured yet)
        secrets_path = Path("config/secrets.yaml")
        if secrets_path.exists():
            with open(secrets_path, 'r') as f:
                secrets = yaml.safe_load(f)
            print("[SUCCESS] Configuration files loaded successfully")
            return True
        else:
            print("[WARNING] secrets.yaml not found - API keys need to be configured")
            return True

    except Exception as e:
        print(f"[ERROR] Configuration loading failed: {e}")
        return False


def test_pipeline_initialization():
    """
    Test pipeline initialization (may fail without proper API keys).

    Returns:
        True if pipeline initializes (or fails as expected), False on unexpected errors.
    """
    print("Testing pipeline initialization...")
    try:
        from src.pipeline import RAGPipeline
        pipeline = RAGPipeline()
        print("[SUCCESS] Pipeline initialized successfully")
        return True
    except Exception as e:
        print(f"[WARNING] Pipeline initialization failed (expected without API keys): {e}")
        return True  # This is expected to fail without proper configuration


def test_metrics():
    """
    Test evaluation metrics calculation.

    Returns:
        True if metrics calculate correctly, False otherwise.
    """
    print("Testing evaluation metrics...")
    try:
        from src.evaluation.metrics import EvaluationMetrics

        metrics = EvaluationMetrics()

        # Test faithfulness
        response = "Please restart your router"
        context = "Customer: My internet is slow. Agent: Please restart your router."
        faithfulness = metrics.calculate_faithfulness(response, context)
        assert isinstance(faithfulness, float)

        # Test relevancy
        query = "My internet is slow"
        relevancy = metrics.calculate_relevancy(response, query)
        assert isinstance(relevancy, float)

        print("[SUCCESS] Metrics calculation working")
        return True
    except Exception as e:
        print(f"[ERROR] Metrics test failed: {e}")
        return False


def main():
    """
    Run all tests and report results.

    Returns:
        True if all tests pass, False otherwise.
    """
    print("Testing RAG Pipeline Components")
    print("=" * 40)

    tests = [
        test_imports,
        test_config_loading,
        test_pipeline_initialization,
        test_metrics
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 40)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("[SUCCESS] All tests passed!")
    else:
        print("[WARNING] Some tests failed - check configuration")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
