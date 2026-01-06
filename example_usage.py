#!/usr/bin/env python3
"""
Example usage of the RAG Telecom Support Assistant
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def demo_basic_usage():
    """Demonstrate basic pipeline usage"""
    print("RAG Telecom Support Assistant - Demo")
    print("=" * 50)

    try:
        from src.pipeline import RAGPipeline

        # Initialize pipeline
        print("Initializing pipeline...")
        pipeline = RAGPipeline()

        # Example queries
        queries = [
            "My internet is very slow and keeps disconnecting",
            "I can't connect to the WiFi network",
            "My phone bill is much higher than usual",
            "I need help setting up mobile data"
        ]

        print("\nGenerating responses for example queries...\n")

        for i, query in enumerate(queries, 1):
            print(f"Query {i}: {query}")
            print("-" * 40)

            try:
                response = pipeline.generate_response(query)
                print(f"Response: {response}")
            except Exception as e:
                print(f"Error: {e}")

            print("\n" + "=" * 50 + "\n")

    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        print("Check your configuration in config/secrets.yaml")

def demo_evaluation():
    """Demonstrate evaluation capabilities"""
    print("Evaluation Demo")
    print("=" * 30)

    try:
        from src.evaluation.metrics import EvaluationMetrics
        from src.evaluation.run_eval import Evaluator

        # Create sample test data
        test_data = [
            {
                "query": "My internet is slow",
                "expected_answer": "Let me help you troubleshoot your slow internet connection."
            },
            {
                "query": "WiFi not connecting",
                "expected_answer": "I'll guide you through WiFi connection troubleshooting steps."
            }
        ]

        # Initialize components
        metrics = EvaluationMetrics()
        evaluator = Evaluator(None, metrics)  # Pipeline not needed for metrics demo

        print("Calculating metrics for sample responses...")

        for test_case in test_data:
            query = test_case["query"]
            expected = test_case["expected_answer"]

            # Mock generated response
            generated = f"I understand you're having issues with {query}. Let me help you resolve this."

            # Calculate metrics
            eval_metrics = metrics.evaluate_response(
                query=query,
                response=generated,
                expected_answer=expected
            )

            print(f"\nQuery: {query}")
            print(f"Generated: {generated}")
            print(f"Expected: {expected}")
            print("Metrics:")
            for metric, value in eval_metrics.items():
                print(".3f")

    except Exception as e:
        print(f"[ERROR] Evaluation demo failed: {e}")

def demo_data_ingestion():
    """Demonstrate data ingestion (without actually loading large dataset)"""
    print("Data Ingestion Demo")
    print("=" * 25)

    try:
        from src.ingestion.loader import DataLoader
        from src.ingestion.processor import DataProcessor

        print("This would load the 'eisenzopf/telecom-conversation-corpus' dataset")
        print("For demo purposes, we'll just show the configuration...")

        # Show configuration
        import yaml
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        print(f"Dataset: {config['data']['dataset_name']}")
        print(f"Cache dir: {config['data']['cache_dir']}")
        print(f"Embedding model: {config['models']['embedding_model']}")
        print(f"LLM model: {config['models']['llm_model']}")

        print("\nTo run actual data ingestion:")
        print("from src.pipeline import RAGPipeline")
        print("pipeline = RAGPipeline()")
        print("pipeline.ingest_data()")

    except Exception as e:
        print(f"[ERROR] Data ingestion demo failed: {e}")

def main():
    """Main demo function"""
    print("Choose a demo:")
    print("1. Basic pipeline usage")
    print("2. Evaluation metrics")
    print("3. Data ingestion overview")
    print("4. Run all demos")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        demo_basic_usage()
    elif choice == "2":
        demo_evaluation()
    elif choice == "3":
        demo_data_ingestion()
    elif choice == "4":
        demo_basic_usage()
        print("\n" + "="*50 + "\n")
        demo_evaluation()
        print("\n" + "="*50 + "\n")
        demo_data_ingestion()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    # If run directly, show menu
    if len(sys.argv) == 1:
        main()
    else:
        # If arguments provided, run specific demo
        arg = sys.argv[1]
        if arg == "basic":
            demo_basic_usage()
        elif arg == "eval":
            demo_evaluation()
        elif arg == "data":
            demo_data_ingestion()
        else:
            print("Usage: python example_usage.py [basic|eval|data]")
