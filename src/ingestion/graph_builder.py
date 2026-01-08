"""
Advanced Graph RAG Builder for Knowledge Graph Construction
Creates true entity relationships for telecom support knowledge base

Implements Graph RAG by modeling:
- Products/Services (Router, Internet Plan, etc.)
- Issues (Slow Internet, Dropped Calls, etc.)
- Resolutions (Restart Device, Check Connections, etc.)
- Relationships between entities for context-aware retrieval
"""

from neo4j import GraphDatabase
from typing import List, Dict, Any, Set, Tuple
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from collections import defaultdict

class GraphBuilder:
    """
    Advanced Graph RAG Builder that creates entity relationships for intelligent retrieval.

    This implements true Graph RAG by:
    1. Extracting entities (Products, Issues, Resolutions) from conversations
    2. Creating structured relationships between entities
    3. Enabling graph traversal for context-aware retrieval
    """
        # Class-level entity patterns
    entity_patterns = {
        'product': [
            r'\b(?:router|modem|phone|mobile|internet|broadband|fiber|dsl|wifi|wireless|5g|4g|3g|sim|ethernet)\b',
            r'\b(?:iphone|android|samsung|huawei|motorola|lg|nokia|oneplus|google pixel|xiaomi)\b',
            r'\b(?:verizon|att|t-mobile|sprint|comcast|xfinity|spectrum|cox|centurylink)\b'
        ],
        'issue': [
            r'\b(?:slow|speed|connection|disconnect|drop|buffering|lag|latency|jitter|packet loss)\b',
            r'\b(?:bill|billing|charge|fee|cost|expensive|overcharge|payment|refund|credit)\b',
            r'\b(?:data|plan|unlimited|throttling|limit|cap|usage|overage)\b',
            r'\b(?:signal|reception|coverage|bars|network|dead zone|no service)\b'
        ],
        'resolution': [
            'restart', 'reset', 'reboot', 'power cycle', 'unplug', 'check connection',
            'update', 'factory reset', 'contact support', 'troubleshoot', 'diagnostic',
            'password reset', 'account verification', 'service refresh'
        ]
    }

    def __init__(self, neo4j_config: Dict[str, str]):
        self.uri = neo4j_config['uri']
        self.username = neo4j_config['username']
        self.password = neo4j_config['password']
        self.driver = None
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    

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
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text using enhanced pattern matching."""
        text_lower = text.lower()
        entities = {key: [] for key in self.entity_patterns.keys()}
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                entities[entity_type].extend(matches)
        
        # Remove duplicates while preserving order
        return {k: list(dict.fromkeys(v)) for k, v in entities.items()}
    def build_entity_relationships(self, conversation_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build entity relationships from conversation pairs for Graph RAG.

        Args:
            conversation_pairs: List of client-agent conversation pairs

        Returns:
            Dictionary with entity relationships and statistics
        """
        entity_graph = {
            'product_issue_links': defaultdict(list),  # Product -> Issues
            'issue_resolution_links': defaultdict(list),  # Issue -> Resolutions
            'conversation_entities': []  # Per-conversation entity tracking
        }

        for pair in conversation_pairs:
            client_msg = pair.get('client_message', '')
            agent_msg = pair.get('agent_response', '')
            conv_id = pair.get('conversation_id', '')

            # Extract entities from both messages
            client_entities = self.extract_entities(client_msg)
            agent_entities = self.extract_entities(agent_msg)

            # Combine entities
            all_products = client_entities['products'] + agent_entities['products']
            all_issues = client_entities['issues'] + agent_entities['issues']
            all_resolutions = agent_entities['resolutions']  # Resolutions typically from agent

            # Build relationships
            for product in all_products:
                for issue in all_issues:
                    entity_graph['product_issue_links'][product].append(issue)

                for resolution in all_resolutions:
                    # Link products to resolutions through issues
                    for issue in all_issues:
                        entity_graph['issue_resolution_links'][issue].append(resolution)

            # Track per-conversation entities
            entity_graph['conversation_entities'].append({
                'conversation_id': conv_id,
                'products': list(set(all_products)),
                'issues': list(set(all_issues)),
                'resolutions': list(set(all_resolutions)),
                'client_message': client_msg,
                'agent_response': agent_msg
            })

        return entity_graph



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

    def _create_entity_nodes(self, tx, entity_graph: Dict[str, Any]):
        """Create entity nodes (Products, Issues, Resolutions)"""
        # Create product nodes
        for product in set().union(*[set(issues) for issues in entity_graph['product_issue_links'].values()]):
            embedding = self.embedder.encode(f"telecom product {product}").tolist()
            tx.run("""
                MERGE (p:Product {name: $name})
                SET p.embedding = $embedding,
                    p.type = 'product',
                    p.created_at = datetime()
                """, name=product, embedding=embedding)

        # Create issue nodes
        all_issues = set()
        for issues in entity_graph['product_issue_links'].values():
            all_issues.update(issues)
        all_issues.update(entity_graph['issue_resolution_links'].keys())
        
        for issue in all_issues:
            embedding = self.embedder.encode(f"telecom issue {issue}").tolist()
            tx.run("""
                MERGE (i:Issue {name: $name})
                SET i.embedding = $embedding,
                    i.type = 'issue',
                    i.created_at = datetime()
                """, name=issue, embedding=embedding)

        # Create resolution nodes
        all_resolutions = set()
        for resolutions in entity_graph['issue_resolution_links'].values():
            all_resolutions.update(resolutions)
            
        for resolution in all_resolutions:
            embedding = self.embedder.encode(f"telecom resolution {resolution}").tolist()
            tx.run("""
                MERGE (r:Resolution {name: $name})
                SET r.embedding = $embedding,
                    r.type = 'resolution',
                    r.created_at = datetime()
                """, name=resolution, embedding=embedding)
    
    def _create_conversation_nodes_with_entities(self, tx, conversation_pairs: List[Dict[str, Any]], entity_graph: Dict[str, Any]):
        """Create conversation nodes linked to extracted entities"""
        for pair in conversation_pairs:
            # Find corresponding entity data
            conv_entity_data = None
            for entity_data in entity_graph['conversation_entities']:
                if entity_data['conversation_id'] == pair.get('conversation_id'):
                    conv_entity_data = entity_data
                    break

            if conv_entity_data:
                # Create conversation with entity links
                client_embedding = self.embedder.encode(pair['client_message']).tolist()
                agent_embedding = self.embedder.encode(pair['agent_response']).tolist()

                tx.run("""
                    MERGE (conv:Conversation {id: $conv_id})
                    SET conv.client_message = $client_msg,
                        conv.agent_response = $agent_msg,
                        conv.client_embedding = $client_embedding,
                        conv.agent_embedding = $agent_embedding,
                        conv.products = $products,
                        conv.issues = $issues,
                        conv.resolutions = $resolutions,
                        conv.created_at = datetime()

                    // Link to products
                    FOREACH (product IN $products |
                        MERGE (p:Product {name: product})
                        MERGE (conv)-[:MENTIONS_PRODUCT]->(p)
                    )

                    // Link to issues
                    FOREACH (issue IN $issues |
                        MERGE (i:Issue {name: issue})
                        MERGE (conv)-[:HAS_ISSUE]->(i)
                    )

                    // Link to resolutions
                    FOREACH (resolution IN $resolutions |
                        MERGE (r:Resolution {name: resolution})
                        MERGE (conv)-[:USES_RESOLUTION]->(r)
                    )
                    """,
                    conv_id=pair['conversation_id'],
                    client_msg=pair['client_message'],
                    agent_msg=pair['agent_response'],
                    client_embedding=client_embedding,
                    agent_embedding=agent_embedding,
                    products=conv_entity_data['products'],
                    issues=conv_entity_data['issues'],
                    resolutions=conv_entity_data['resolutions']
                )

    def _create_entity_relationships(self, tx, entity_graph: Dict[str, Any]):
        """Create relationships between entities (Product->Issue->Resolution)"""
        # Create Product -> Issue relationships
        for product, issues in entity_graph['product_issue_links'].items():
            issue_counts = defaultdict(int)
            for issue in issues:
                issue_counts[issue] += 1

            for issue, count in issue_counts.items():
                tx.run("""
                    MATCH (p:Product {name: $product})
                    MATCH (i:Issue {name: $issue})
                    MERGE (p)-[r:HAS_COMMON_ISSUE]->(i)
                    SET r.frequency = $count,
                        r.last_updated = datetime()
                    """,
                    product=product,
                    issue=issue,
                    count=count
                )

        # Create Issue -> Resolution relationships
        for issue, resolutions in entity_graph['issue_resolution_links'].items():
            resolution_counts = defaultdict(int)
            for resolution in resolutions:
                resolution_counts[resolution] += 1

            for resolution, count in resolution_counts.items():
                tx.run("""
                    MATCH (i:Issue {name: $issue})
                    MATCH (r:Resolution {name: $resolution})
                    MERGE (i)-[rel:HAS_RESOLUTION]->(r)
                    SET rel.frequency = $count,
                        rel.success_rate = 1.0,  // Could be learned from user feedback
                        rel.last_updated = datetime()
                    """,
                    issue=issue,
                    resolution=resolution,
                    count=count

                )

    def search_graph_rag(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Advanced Graph RAG search that uses entity relationships and vector similarity.

        Args:
            query: User query
            limit: Maximum number of results to return

        Returns:
            Dictionary containing matching conversations and related entities
        """
        if not self.driver:
            self.connect()

        # Extract entities from query
        query_entities = self.extract_entities(query)
        query_embedding = self.embedder.encode(query).tolist()

        with self.driver.session() as session:
            # Find similar conversations using vector search
            results = session.run("""
                MATCH (c:Conversation)
                WITH c, gds.similarity.cosine(c.client_embedding, $embedding) AS similarity
                WHERE similarity > 0.7
                RETURN c, similarity
                ORDER BY similarity DESC
                LIMIT $limit
                """, 
                embedding=query_embedding, 
                limit=limit
            )

            conversations = [{
                'conversation_id': record['c']['id'],
                'client_message': record['c']['client_message'],
                'agent_response': record['c']['agent_response'],
                'similarity': float(record['similarity'])
            } for record in results]

            # Find related entities
            related_entities = []
            if query_entities.get('product') or query_entities.get('issue'):
                # Find related issues and resolutions based on query entities
                related_entities = session.run("""
                    MATCH (p:Product)-[r1:HAS_ISSUE]->(i:Issue)-[r2:HAS_RESOLUTION]->(r:Resolution)
                    WHERE p.name IN $products OR i.name IN $issues
                    RETURN p.name as product, i.name as issue, r.name as resolution,
                           r1.frequency as issue_freq, r2.frequency as resolution_freq,
                           r1.frequency * r2.frequency as relevance_score
                    ORDER BY relevance_score DESC
                    LIMIT 10
                    """, 
                    products=query_entities.get('product', []),
                    issues=query_entities.get('issue', [])
                ).data()

            return {
                'conversations': conversations,
                'related_entities': related_entities,
                'query_entities': query_entities
            }
