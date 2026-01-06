"""
RAG Pipeline Orchestrator
Combines ingestion, retrieval, and generation components
"""

import yaml
import os
from typing import Dict, List, Any
from .ingestion.loader import DataLoader
from .ingestion.processor import DataProcessor
from .ingestion.graph_builder import GraphBuilder
from .retrieval.search import VectorSearch
from .generation.llm import LLMGenerator

class RAGPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load secrets
        with open("config/secrets.yaml", 'r') as f:
            self.secrets = yaml.safe_load(f)

        # Initialize components
        self.loader = DataLoader(self.config)
        self.processor = DataProcessor()
        self.graph_builder = GraphBuilder(self.secrets['neo4j'])
        self.search = VectorSearch(self.config, self.secrets)
        self.generator = LLMGenerator(self.config, self.secrets)

    def ingest_data(self):
        """Ingest data from Hugging Face and build knowledge graph"""
        print("Loading dataset...")
        dataset = self.loader.load_dataset()

        print("Processing conversations...")
        conversations = self.processor.process_conversations(dataset)

        print("Building knowledge graph...")
        self.graph_builder.build_graph(conversations)

        print("Ingestion complete!")

    def generate_response(self, query: str) -> str:
        """Generate response using RAG pipeline"""
        # Retrieve similar issues
        similar_issues = self.search.find_similar_issues(query)

        # Generate response using LLM
        response = self.generator.generate_response(query, similar_issues)

        return response

    def evaluate_pipeline(self, test_data: List[Dict[str, Any]]):
        """Evaluate the RAG pipeline performance"""
        from .evaluation.metrics import EvaluationMetrics
        from .evaluation.run_eval import Evaluator

        evaluator = Evaluator(self, EvaluationMetrics())
        results = evaluator.evaluate(test_data)

        return results
