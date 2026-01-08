# src/evaluation/run_eval.py
"""
Standalone RAG Pipeline Evaluation Script

This script evaluates the RAG pipeline by comparing generated responses against expected answers.
It reads test cases from a JSON file and calculates recall metrics.

Usage:
    python -m src.evaluation.run_eval --test-file evaluation/test_queries.json
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

class EvaluationMetrics:
    """
    Simple evaluation metrics focused on recall.
    """
    def __init__(self):
        pass

    def calculate_recall(self, predicted: str, expected: str) -> float:
        """
        Calculate recall score - how many key elements from expected answer
        are present in the predicted answer.

        Args:
            predicted: Generated response text
            expected: Expected/ground truth text

        Returns:
            Recall score between 0.0 and 1.0
        """
        if not predicted or not expected:
            return 0.0

        # Simple word-level recall
        expected_words = set(word.lower() for word in expected.split() if len(word) > 3)
        if not expected_words:
            return 0.0

        predicted_words = set(word.lower() for word in predicted.split())
        if not predicted_words:
            return 0.0

        # Calculate recall
        matching_words = expected_words.intersection(predicted_words)
        return len(matching_words) / len(expected_words)

class RAGEvaluator:
    """
    Standalone RAG pipeline evaluator.
    """
    def __init__(self):
        self.metrics = EvaluationMetrics()
        self.results_dir = Path("evaluation/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_test_cases(self, test_file: str) -> List[Dict[str, Any]]:
        """
        Load test cases from a JSON file.
        
        Expected format:
        [
            {
                "query": "What is the status of my order?",
                "expected_answer": "Your order #123 is in transit",
                "generated_response": "The order #123 is currently being shipped"
            },
            ...
        ]
        """
        try:
            with open(test_file, 'r') as f:
                test_cases = json.load(f)
            
            # Validate test cases
            valid_cases = []
            for i, case in enumerate(test_cases, 1):
                if not isinstance(case, dict):
                    print(f"Warning: Test case {i} is not a dictionary, skipping")
                    continue
                
                if 'query' not in case or 'expected_answer' not in case or 'generated_response' not in case:
                    print(f"Warning: Test case {i} is missing required fields, skipping")
                    continue
                
                valid_cases.append(case)
            
            return valid_cases
            
        except Exception as e:
            print(f"Error loading test cases: {str(e)}")
            return []

    def evaluate(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate test cases and calculate metrics.
        """
        results = {
            'evaluation_time': datetime.utcnow().isoformat(),
            'total_cases': len(test_cases),
            'evaluated_cases': 0,
            'average_recall': 0.0,
            'test_cases': []
        }
        
        total_recall = 0.0
        
        for case in test_cases:
            query = case.get('query', '')
            expected = case.get('expected_answer', '')
            generated = case.get('generated_response', '')
            
            try:
                # Calculate recall
                recall = self.metrics.calculate_recall(generated, expected)
                total_recall += recall
                results['evaluated_cases'] += 1
                
                # Store results
                result = {
                    'query': query,
                    'expected_answer': expected,
                    'generated_response': generated,
                    'recall': recall,
                    'success': True
                }
                
                results['test_cases'].append(result)
                print(f"Query: {query[:60]}... | Recall: {recall:.2f}")
                
            except Exception as e:
                print(f"Error evaluating query: {str(e)}")
                results['test_cases'].append({
                    'query': query,
                    'error': str(e),
                    'success': False
                })
        
        # Calculate average recall
        if results['evaluated_cases'] > 0:
            results['average_recall'] = total_recall / results['evaluated_cases']
        
        return results

    def save_results(self, results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """
        Save evaluation results to a JSON file.
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.results_dir / f"eval_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return str(output_file)

    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of the evaluation results."""
        print("\n" + "="*50)
        print("RAG Pipeline Evaluation Summary")
        print("="*50)
        print(f"Evaluation Time: {results.get('evaluation_time', 'N/A')}")
        print(f"Total Test Cases: {results.get('total_cases', 0)}")
        print(f"Successfully Evaluated: {results.get('evaluated_cases', 0)}")
        print(f"Average Recall: {results.get('average_recall', 0.0):.4f}")
        
        # Print top and bottom performing cases
        test_cases = results.get('test_cases', [])
        if test_cases:
            # Sort by recall (highest first)
            sorted_cases = sorted(
                [c for c in test_cases if c.get('success', False)],
                key=lambda x: x.get('recall', 0),
                reverse=True
            )
            
            if sorted_cases:
                print("\nTop Performing Queries:")
                for i, case in enumerate(sorted_cases[:3], 1):
                    print(f"  {i}. Recall: {case.get('recall', 0):.2f} - {case.get('query', '')[:60]}...")
                
                print("\nBottom Performing Queries:")
                for i, case in enumerate(sorted_cases[-3:], 1):
                    print(f"  {i}. Recall: {case.get('recall', 0):.2f} - {case.get('query', '')[:60]}...")

def main():
    """Command-line interface for running evaluations."""
    parser = argparse.ArgumentParser(description='Evaluate RAG Pipeline')
    parser.add_argument(
        '--test-file',
        type=str,
        required=True,
        help='Path to JSON file containing test cases with expected answers and generated responses'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for evaluation results (default: evaluation/results/eval_results_<timestamp>.json)'
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Load test cases
    test_cases = evaluator.load_test_cases(args.test_file)
    if not test_cases:
        print("No valid test cases found. Exiting.")
        return
    
    print(f"Loaded {len(test_cases)} test cases for evaluation")
    
    # Run evaluation
    results = evaluator.evaluate(test_cases)
    
    # Save results
    output_file = evaluator.save_results(results, args.output)
    
    # Print summary
    evaluator.print_summary(results)
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()