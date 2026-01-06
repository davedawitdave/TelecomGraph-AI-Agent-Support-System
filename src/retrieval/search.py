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
        """
        Find similar customer issues using Graph RAG approach.

        Combines entity-based graph search with vector similarity for comprehensive retrieval.
        """
        try:
            # First, try Graph RAG search (entity relationships)
            graph_results = self.graph_builder.search_graph_rag(query, top_k)

            if graph_results['graph_conversations']:
                # Convert graph results to expected format
                formatted_results = []
                for conv in graph_results['graph_conversations']:
                    formatted_results.append({
                        'client_message': conv['client_message'],
                        'agent_response': conv['agent_response'],
                        'similarity': conv.get('issue_match_count', 0.5),  # Use entity match count as similarity
                        'source': 'graph_rag'
                    })

                # If we have enough graph results, return them
                if len(formatted_results) >= top_k // 2:
                    return formatted_results[:top_k]

            # Fallback to vector search for remaining slots
            remaining_slots = top_k - len(graph_results.get('graph_conversations', []))
            if remaining_slots > 0:
                vector_results = self.graph_builder.search_similar_issues(query, remaining_slots)

                # Add source indicator
                for result in vector_results:
                    result['source'] = 'vector_search'

                formatted_results.extend(vector_results)

            return formatted_results[:top_k]

        except Exception as e:
            print(f"Graph RAG search failed: {e}. Falling back to vector search.")
            try:
                # Fallback to pure vector search
                results = self.graph_builder.search_similar_issues(query, top_k)
                for result in results:
                    result['source'] = 'vector_fallback'
                return results
            except Exception as e2:
                print(f"Vector search also failed: {e2}. Using local fallback.")
                # Final fallback to local search
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
