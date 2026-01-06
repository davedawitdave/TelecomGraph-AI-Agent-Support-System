"""
Evaluation Runner for RAG Pipeline
Runs the pipeline against test data and computes metrics
"""

from typing import List, Dict, Any, Tuple
import json
import os
from .metrics import EvaluationMetrics

class Evaluator:
    def __init__(self, pipeline, metrics_calculator: EvaluationMetrics):
        self.pipeline = pipeline
        self.metrics = metrics_calculator

    def evaluate(self, test_data: List[Dict[str, Any]],
                output_file: str = "evaluation_results.json") -> Dict[str, Any]:
        """Run evaluation on test data"""
        print(f"Starting evaluation on {len(test_data)} test samples...")

        results = []
        for i, test_case in enumerate(test_data):
            print(f"Evaluating sample {i+1}/{len(test_data)}")

            # Generate response
            query = test_case.get('query', '')
            expected_answer = test_case.get('expected_answer', '')

            try:
                generated_response = self.pipeline.generate_response(query)

                # Get context from similar issues (if available)
                similar_issues = self.pipeline.search.find_similar_issues(query)
                context = self._format_context(similar_issues)

                # Calculate simple recall-based metrics
                metrics = {}

                # Query coverage - how well response addresses the original query
                metrics['query_coverage'] = self.metrics.calculate_query_coverage(
                    generated_response, query
                )

                # Context recall - how well response uses provided context
                if context:
                    metrics['context_recall'] = self.metrics.calculate_context_recall(
                        generated_response, context
                    )

                # Answer recall - how well response matches expected answer (if available)
                if expected_answer:
                    metrics['answer_recall'] = self.metrics.calculate_recall(
                        generated_response, expected_answer
                    )

                result = {
                    'query': query,
                    'generated_response': generated_response,
                    'expected_answer': expected_answer,
                    'context': context,
                    'metrics': metrics
                }

                results.append(result)

            except Exception as e:
                print(f"Error evaluating sample {i+1}: {e}")
                results.append({
                    'query': query,
                    'error': str(e),
                    'metrics': {}
                })

        # Calculate simple aggregate metrics
        aggregate_scores = self._calculate_simple_aggregates(
            [r['metrics'] for r in results if 'metrics' in r]
        )

        final_results = {
            'individual_results': results,
            'aggregate_scores': aggregate_scores,
            'summary': {
                'total_samples': len(test_data),
                'successful_evaluations': len([r for r in results if 'generated_response' in r]),
                'failed_evaluations': len([r for r in results if 'error' in r])
            }
        }

        # Save results
        self._save_results(final_results, output_file)

        print("Evaluation complete!")
        self._print_summary(final_results)

        return final_results

    def _calculate_simple_aggregates(self, evaluations: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate simple aggregate statistics for evaluation metrics.

        Args:
            evaluations: List of evaluation result dictionaries

        Returns:
            Dictionary with mean scores for each metric
        """
        if not evaluations:
            return {}

        # Collect all metric names
        all_metrics = set()
        for eval_result in evaluations:
            all_metrics.update(eval_result.keys())

        aggregate_scores = {}
        for metric in all_metrics:
            values = [eval_result.get(metric, 0.0) for eval_result in evaluations if metric in eval_result]
            if values:
                aggregate_scores[f"{metric}_mean"] = sum(values) / len(values)
                aggregate_scores[f"{metric}_count"] = len(values)

        return aggregate_scores

    def _format_context(self, similar_issues: List[Dict[str, Any]]) -> str:
        """Format similar issues into context string"""
        if not similar_issues:
            return ""

        context_parts = []
        for issue in similar_issues:
            context_parts.append(f"Customer: {issue.get('client_message', '')}")
            context_parts.append(f"Agent: {issue.get('agent_response', '')}")

        return "\n".join(context_parts)

    def _save_results(self, results: Dict[str, Any], output_file: str):
        """Save evaluation results to file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")

    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary"""
        summary = results['summary']
        aggregates = results['aggregate_scores']

        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Total samples: {summary['total_samples']}")
        print(f"Successful evaluations: {summary['successful_evaluations']}")
        print(f"Failed evaluations: {summary['failed_evaluations']}")

        print("\nAGGREGATE METRICS:")
        for metric_name, value in aggregates.items():
            if isinstance(value, (int, float)):
                print(".3f")

        print("="*50)

    def load_test_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load test data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading test data: {e}")
            return []

    def create_sample_test_data(self) -> List[Dict[str, Any]]:
        """Create sample test data for evaluation"""
        return [
            {
                "query": "My internet is very slow",
                "expected_answer": "Let me help you troubleshoot your slow internet connection."
            },
            {
                "query": "I can't connect to WiFi",
                "expected_answer": "I'll guide you through WiFi connection troubleshooting steps."
            },
            {
                "query": "My phone bill is too high",
                "expected_answer": "I can help you review your bill and identify any unexpected charges."
            }
        ]
