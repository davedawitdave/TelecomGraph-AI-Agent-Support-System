"""
Neo4j Graph Builder for Knowledge Graph Construction
Ingests processed conversation data into Neo4j graph database
"""

from neo4j import GraphDatabase
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

class GraphBuilder:
    def __init__(self, neo4j_config: Dict[str, str]):
        self.uri = neo4j_config['uri']
        self.username = neo4j_config['username']
        self.password = neo4j_config['password']
        self.driver = None
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedder

    def connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Test connection
            self.driver.verify_connectivity()
            print("Connected to Neo4j successfully")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            raise

    def disconnect(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            print("Disconnected from Neo4j")

    def build_graph(self, conversation_pairs: List[Dict[str, Any]]):
        """Build knowledge graph from conversation pairs"""
        if not self.driver:
            self.connect()

        with self.driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")

            # Create constraints
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Conversation) REQUIRE c.id IS UNIQUE")

            # Process conversation pairs in batches
            batch_size = 100
            for i in range(0, len(conversation_pairs), batch_size):
                batch = conversation_pairs[i:i + batch_size]
                session.execute_write(self._create_conversation_nodes, batch)

        print(f"Created knowledge graph with {len(conversation_pairs)} conversation pairs")

    def _create_conversation_nodes(self, tx, batch: List[Dict[str, Any]]):
        """Create conversation nodes and relationships in Neo4j"""
        for pair in batch:
            # Generate embeddings
            client_embedding = self.embedder.encode(pair['client_message']).tolist()
            agent_embedding = self.embedder.encode(pair['agent_response']).tolist()

            # Create nodes and relationships
            query = """
            MERGE (client_msg:Message {
                text: $client_msg,
                type: 'client',
                embedding: $client_embedding
            })
            MERGE (agent_msg:Message {
                text: $agent_msg,
                type: 'agent',
                embedding: $agent_embedding
            })
            MERGE (client_msg)-[:RESPONDS_TO]->(agent_msg)
            MERGE (conv:Conversation {id: $conv_id})
            MERGE (client_msg)-[:PART_OF]->(conv)
            MERGE (agent_msg)-[:PART_OF]->(conv)
            """

            tx.run(query,
                   client_msg=pair['client_message'],
                   agent_msg=pair['agent_response'],
                   client_embedding=client_embedding,
                   agent_embedding=agent_embedding,
                   conv_id=pair['conversation_id'])

    def search_similar_issues(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar client messages using vector similarity.

        Uses a simplified approach without GDS library for broader compatibility.
        """
        if not self.driver:
            self.connect()

        query_embedding = self.embedder.encode(query).tolist()

        with self.driver.session() as session:
            # Get all client messages and their embeddings
            result = session.run("""
                MATCH (msg:Message {type: 'client'})
                RETURN msg.text AS client_message, msg.embedding AS embedding,
                       id(msg) AS msg_id
                """)

            similarities = []
            for record in result:
                stored_embedding = record['embedding']
                if stored_embedding:
                    # Calculate cosine similarity manually
                    similarity = self._calculate_cosine_similarity(query_embedding, stored_embedding)
                    similarities.append({
                        'client_message': record['client_message'],
                        'embedding': stored_embedding,
                        'msg_id': record['msg_id'],
                        'similarity': similarity
                    })

            # Sort by similarity (higher is better for cosine similarity)
            similarities.sort(key=lambda x: x['similarity'], reverse=True)

            # Get top results and their responses
            top_results = []
            for sim in similarities[:limit]:
                # Get the agent response for this client message
                response_result = session.run("""
                    MATCH (msg)-[:RESPONDS_TO]->(response:Message)
                    WHERE id(msg) = $msg_id
                    RETURN response.text AS agent_response
                    """,
                    msg_id=sim['msg_id'])

                response_record = response_result.single()
                if response_record:
                    top_results.append({
                        'client_message': sim['client_message'],
                        'agent_response': response_record['agent_response'],
                        'similarity': sim['similarity']
                    })

            return top_results

    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (between -1 and 1)
        """
        import math

        # Convert to numpy arrays for easier computation
        v1 = [float(x) for x in vec1]
        v2 = [float(x) for x in vec2]

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(v1, v2))

        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in v1))
        magnitude2 = math.sqrt(sum(b * b for b in v2))

        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        # Calculate cosine similarity
        return dot_product / (magnitude1 * magnitude2)
