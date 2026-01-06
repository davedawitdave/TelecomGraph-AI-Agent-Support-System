"""
Vector Search for Similar Customer Issues
Uses embeddings to find relevant past conversations
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from ..ingestion.graph_builder import GraphBuilder

class VectorSearch:
    def __init__(self, config: Dict[str, Any], secrets: Dict[str, Any]):
        self.config = config
        self.embedder = SentenceTransformer(
            config.get('models', {}).get('embedding_model', 'text-embedding-3-small')
        )
        self.graph_builder = GraphBuilder(secrets['neo4j'])

    def find_similar_issues(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar customer issues using vector search"""
        try:
            # Use Neo4j for vector search if available
            results = self.graph_builder.search_similar_issues(query, top_k)
            return results
        except Exception as e:
            print(f"Neo4j search failed: {e}. Falling back to local search.")
            # Fallback to local search if Neo4j is not available
            return self._local_vector_search(query, top_k)

    def _local_vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback local vector search (requires pre-computed embeddings)"""
        # This would require storing embeddings locally
        # For now, return empty results
        print("Local vector search not implemented - requires pre-computed embeddings")
        return []

    def compute_similarity(self, query_embedding: np.ndarray,
                          stored_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and stored embeddings"""
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        stored_norm = stored_embeddings / np.linalg.norm(stored_embeddings, axis=1, keepdims=True)

        # Compute cosine similarity
        similarities = np.dot(stored_norm, query_norm)
        return similarities

    def rank_results(self, similarities: np.ndarray,
                    texts: List[str], top_k: int) -> List[Tuple[str, float]]:
        """Rank search results by similarity score"""
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((texts[idx], float(similarities[idx])))

        return results
