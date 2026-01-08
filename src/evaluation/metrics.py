"""
Simple Evaluation Metrics for RAG Pipeline
Basic recall-based evaluation techniques
"""

import re
from typing import Dict, List, Any
from collections import Counter
import numpy as np

class EvaluationMetrics:
    """
    Simple evaluation metrics using basic recall techniques.
    Focuses on text overlap and keyword matching rather than complex embeddings.
    """

    def __init__(self):
        pass  # No complex initialization needed

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

        # Extract key words/phrases from expected answer
        expected_keywords = self._extract_keywords(expected)

        if not expected_keywords:
            return 0.0

        # Count how many expected keywords appear in predicted answer
        predicted_lower = predicted.lower()
        found_keywords = sum(1 for keyword in expected_keywords
                           if keyword.lower() in predicted_lower)

        return found_keywords / len(expected_keywords)

    def calculate_context_recall(self, response: str, context: str) -> float:
        """
        Calculate context recall - how well the response uses information
        from the provided context.

        Args:
            response: Generated response text
            context: Context information provided

        Returns:
            Context recall score between 0.0 and 1.0
        """
        if not response or not context:
            return 0.0

        # Extract key terms from context
        context_keywords = self._extract_keywords(context)

        if not context_keywords:
            return 0.0

        # Count how many context keywords appear in response
        response_lower = response.lower()
        found_keywords = sum(1 for keyword in context_keywords
                           if keyword.lower() in response_lower)

        return found_keywords / len(context_keywords)

    def calculate_query_coverage(self, response: str, query: str) -> float:
        """
        Calculate query coverage - how well the response addresses
        the original query terms.

        Args:
            response: Generated response text
            query: Original user query

        Returns:
            Query coverage score between 0.0 and 1.0
        """
        if not response or not query:
            return 0.0

        # Extract key terms from query
        query_keywords = self._extract_keywords(query)

        if not query_keywords:
            return 1.0  # If no specific keywords, consider it covered

        # Count how many query keywords appear in response
        response_lower = response.lower()
        found_keywords = sum(1 for keyword in query_keywords
                           if keyword.lower() in response_lower)

        return found_keywords / len(query_keywords)

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract key words and phrases from text.
        Simple approach focusing on important terms.

        Args:
            text: Input text

        Returns:
            List of extracted keywords
        """
        if not text:
            return []

        # Convert to lowercase and split into words
        words = re.findall(r'\b\w+\b', text.lower())

        # Filter out common stop words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}

        keywords = [word for word in words if len(word) > 2 and word not in stop_words]

        # Get most common keywords (simple term frequency)
        if keywords:
            keyword_counts = Counter(keywords)
            # Return top keywords (up to 10 most frequent)
            return [keyword for keyword, _ in keyword_counts.most_common(10)]

        return keywords

    def calculate_faithfulness(self, response: str, context: str) -> float:
        """
        Calculate faithfulness - how well the response stays true to the provided context.
        Measures whether the response contains information not present in the context.

        Args:
            response: Generated response text
            context: Context information provided

        Returns:
            Faithfulness score between 0.0 and 1.0
        """
        if not response or not context:
            return 0.0

        # Extract key facts/statements from response
        response_sentences = self._extract_sentences(response)
        context_sentences = self._extract_sentences(context)

        if not response_sentences:
            return 1.0  # Empty response is considered faithful

        # For each response sentence, check if it can be supported by context
        faithful_sentences = 0
        for resp_sentence in response_sentences:
            resp_keywords = set(self._extract_keywords(resp_sentence))

            # Check if this sentence's key information appears in context
            sentence_supported = False
            for ctx_sentence in context_sentences:
                ctx_keywords = set(self._extract_keywords(ctx_sentence))
                # If significant overlap in keywords, consider it supported
                if len(resp_keywords.intersection(ctx_keywords)) / len(resp_keywords) > 0.3:
                    sentence_supported = True
                    break

            if sentence_supported:
                faithful_sentences += 1

        return faithful_sentences / len(response_sentences)

    def calculate_relevancy(self, response: str, query: str) -> float:
        """
        Calculate relevancy - how relevant the response is to the query.

        Args:
            response: Generated response text
            query: Original user query

        Returns:
            Relevancy score between 0.0 and 1.0
        """
        return self.calculate_query_coverage(response, query)

    def calculate_answer_correctness(self, response: str, expected_answer: str) -> float:
        """
        Calculate answer correctness - semantic similarity to expected answer.

        Args:
            response: Generated response text
            expected_answer: Expected/ground truth answer

        Returns:
            Correctness score between 0.0 and 1.0
        """
        return self.calculate_recall(response, expected_answer)

    def calculate_answer_relevance(self, response: str, expected_answer: str) -> float:
        """
        Calculate answer relevance - how well the response addresses the expected answer's key points.

        Args:
            response: Generated response text
            expected_answer: Expected/ground truth answer

        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not response or not expected_answer:
            return 0.0

        # Extract key concepts from expected answer
        expected_keywords = set(self._extract_keywords(expected_answer))
        response_keywords = set(self._extract_keywords(response))

        if not expected_keywords:
            return 1.0 if response_keywords else 0.0

        # Calculate overlap
        overlap = len(expected_keywords.intersection(response_keywords))
        return overlap / len(expected_keywords)

    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def evaluate_response(self, query: str, response: str,
                         context: str = "", expected_answer: str = "") -> Dict[str, float]:
        """Comprehensive evaluation of a single response"""
        metrics = {}

        # Faithfulness (if context is provided)
        if context:
            metrics['faithfulness'] = self.calculate_faithfulness(response, context)

        # Relevancy
        metrics['relevancy'] = self.calculate_relevancy(response, query)

        # Answer correctness (if expected answer is provided)
        if expected_answer:
            metrics['answer_correctness'] = self.calculate_answer_correctness(response, expected_answer)
            metrics['answer_relevance'] = self.calculate_answer_relevance(response, expected_answer)

        return metrics

    def calculate_aggregate_scores(self, evaluations: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate aggregate scores across multiple evaluations"""
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
                aggregate_scores[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }

        return aggregate_scores
