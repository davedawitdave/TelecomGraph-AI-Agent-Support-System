"""
RAG Pipeline Orchestrator
Combines ingestion, retrieval, and generation components with lazy loading and caching
"""

import yaml
import os
import streamlit as st
from typing import Dict, List, Any, Optional
from pathlib import Path
from .ingestion.loader import DataLoader
from .ingestion.processor import DataProcessor
from .ingestion.graph_builder import GraphBuilder
from .retrieval.search import VectorSearch
from .generation.llm import LLMGenerator

class RAGPipeline:
    _instance = None
    
    def __new__(cls, config_path: str = "config/config.yaml"):
        if cls._instance is None:
            cls._instance = super(RAGPipeline, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: str = "config/config.yaml"):
        if self._initialized:
            return
            
        self.config_path = config_path
        self._load_configs()
        
        # Initialize components as None - will be loaded on first use
        self._loader = None
        self._processor = None
        self._graph_builder = None
        self._search = None
        self._generator = None
        self._initialized = True
        
    def _load_configs(self):
        """Load configuration files with error handling"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
            secrets_path = Path("config/secrets.yaml")
            if not secrets_path.exists():
                raise FileNotFoundError("secrets.yaml not found in config directory")
                
            with open(secrets_path, 'r') as f:
                self.secrets = yaml.safe_load(f)
                
        except Exception as e:
            st.error(f"Error loading configuration: {str(e)}")
            raise
    
    @property
    def loader(self):
        if self._loader is None:
            self._loader = DataLoader(self.config)
        return self._loader
        
    @property
    def processor(self):
        if self._processor is None:
            self._processor = DataProcessor()
        return self._processor
        
    @property
    def graph_builder(self):
        if self._graph_builder is None:
            self._graph_builder = GraphBuilder(self.secrets.get('neo4j', {}))
        return self._graph_builder
        
    @property
    def search(self):
        if self._search is None:
            self._search = VectorSearch(self.config, self.secrets)
        return self._search
        
    @property
    def generator(self):
        if self._generator is None:
            self._generator = LLMGenerator(self.config, self.secrets)
        return self._generator

    def is_initialized(self) -> bool:
        """Check if the pipeline is properly initialized"""
        try:
            # Check if Neo4j is accessible
            self.graph_builder.check_connection()
            # Check if vector store is initialized
            self.search.check_connection()
            return True
        except Exception as e:
            st.warning(f"Initialization check failed: {str(e)}")
            return False
            
    def initialize_if_needed(self):
        """Initialize the pipeline components if not already done"""
        if not self.is_initialized():
            with st.spinner("Initializing pipeline components..."):
                self._load_configs()
                # Access properties to trigger lazy loading
                _ = self.loader
                _ = self.processor
                _ = self.graph_builder
                _ = self.search
                _ = self.generator
                st.success("Pipeline initialized successfully!")

    def ingest_data(self):
        """Ingest data from Hugging Face and build knowledge graph"""
        with st.spinner("Loading dataset..."):
            dataset = self.loader.load_dataset()

        with st.spinner("Processing conversations..."):
            conversations = self.processor.process_conversations(dataset)

        with st.spinner("Building knowledge graph..."):
            self.graph_builder.build_graph(conversations)

        st.success("Data ingestion complete!")

    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Generate response using RAG pipeline with detailed outputs
        
        Returns:
            Dict containing:
                - vector_search_results: Results from vector similarity search
                - gemini_response: Raw response from Gemini
                - rag_response: Final response augmented with search results
        """
        # Initialize response structure
        response = {
            'vector_search_results': [],
            'gemini_response': None,
            'rag_response': None
        }
        
        try:
            # 1. Perform vector search
            with st.spinner("Searching for similar issues..."):
                similar_issues = self.search.find_similar_issues(query)
                response['vector_search_results'] = similar_issues
            
            # 2. Get raw Gemini response (without context)
            with st.spinner("Generating initial response..."):
                gemini_response = self.generator.generate_response(query)
                response['gemini_response'] = gemini_response
            
            # 3. Get RAG-augmented response (with context)
            with st.spinner("Preparing final response..."):
                # Convert similar_issues to a formatted context string
                context = self._format_similar_issues(similar_issues)
                rag_response = self.generator.generate_response(query, context)
                response['rag_response'] = rag_response
                
            return response
            
        except Exception as e:
            st.error(f"Error in generate_response: {str(e)}")
            response['error'] = str(e)
            return response

    def evaluate_pipeline(self, test_data: List[Dict[str, Any]]):
        """Evaluate the RAG pipeline performance"""
        from .evaluation.metrics import EvaluationMetrics
        from .evaluation.run_eval import Evaluator

        evaluator = Evaluator(self, EvaluationMetrics())
        results = evaluator.evaluate(test_data)

        return results

    def _format_similar_issues(self, similar_issues: List[Dict[str, Any]]) -> str:
        """Format similar issues into a context string."""
        if not similar_issues:
            return "No similar issues found."
            
        context_parts = []
        for i, issue in enumerate(similar_issues[:3]):  # Limit to top 3
            context_parts.append(f"""
--- Similar Issue {i+1} ---
[Customer Issue]
{issue.get('client_message', 'N/A')}

[Agent Response]
{issue.get('agent_response', 'N/A')}

[Additional Context]
Category: {issue.get('category', 'N/A')}
Resolution Time: {issue.get('resolution_time', 'N/A')}
Satisfaction: {issue.get('satisfaction', 'N/A')}
""")
        return "\n".join(context_parts)
