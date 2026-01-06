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

    def __init__(self, neo4j_config: Dict[str, str]):
        self.uri = neo4j_config['uri']
        self.username = neo4j_config['username']
        self.password = neo4j_config['password']
        self.driver = None
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedder

        # Entity extraction patterns for telecom domain
        self.product_patterns = [
            r'\b(?:router|modem|phone|mobile|internet|broadband|fiber|dsl|wifi|wireless)\b',
            r'\b(?:iphone|android|samsung|huawei|motorola|lg)\b',
            r'\b(?:verizon|att|t-mobile|sprint|comcast|xfinity)\b'
        ]

        self.issue_patterns = [
            r'\b(?:slow|speed|connection|disconnect|drop|dropped|buffering|lag|latency)\b',
            r'\b(?:bill|billing|charge|fee|cost|expensive|high)\b',
            r'\b(?:data|plan|unlimited|throttling|limit)\b',
            r'\b(?:signal|reception|coverage|bars|network)\b'
        ]

        self.resolution_keywords = [
            'restart', 'reset', 'reboot', 'power cycle', 'unplug', 'check connection',
            'update', 'factory reset', 'contact support', 'troubleshoot', 'diagnostic'
        ]

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
        """
        Extract entities from text using pattern matching and keyword analysis.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with 'products', 'issues', 'resolutions' keys
        """
        text_lower = text.lower()

        # Extract products/services
        products = []
        for pattern in self.product_patterns:
            matches = re.findall(pattern, text_lower)
            products.extend(matches)

        # Extract issues
        issues = []
        for pattern in self.issue_patterns:
            matches = re.findall(pattern, text_lower)
            issues.extend(matches)

        # Extract resolutions
        resolutions = []
        for keyword in self.resolution_keywords:
            if keyword in text_lower:
                resolutions.append(keyword)

        # Remove duplicates while preserving order
        products = list(dict.fromkeys(products))
        issues = list(dict.fromkeys(issues))
        resolutions = list(dict.fromkeys(resolutions))

        return {
            'products': products,
            'issues': issues,
            'resolutions': resolutions
        }

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

    def build_graph(self, conversation_pairs: List[Dict[str, Any]]):
        """
        Build advanced Graph RAG knowledge graph with entity relationships.

        Creates three layers:
        1. Raw conversation storage (for vector search)
        2. Entity extraction (Products, Issues, Resolutions)
        3. Relationship modeling (how entities connect)
        """
        if not self.driver:
            self.connect()

        with self.driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")

            # Create constraints for entity uniqueness
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Conversation) REQUIRE c.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Product) REQUIRE p.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Issue) REQUIRE i.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Resolution) REQUIRE r.name IS UNIQUE")

            # Build entity relationships first
            print("Extracting entities and building relationships...")
            entity_graph = self.build_entity_relationships(conversation_pairs)

            # Create entity nodes
            session.execute_write(self._create_entity_nodes, entity_graph)

            # Create conversation nodes with entity links
            session.execute_write(self._create_conversation_nodes_with_entities, conversation_pairs, entity_graph)

            # Create relationship edges between entities
            session.execute_write(self._create_entity_relationships, entity_graph)

        total_conversations = len(conversation_pairs)
        total_entities = len(entity_graph['conversation_entities'])
        print(f"Created advanced Graph RAG with {total_conversations} conversations and {total_entities} entity-mapped conversations")

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
        # Collect all unique products
        all_products = set()
        for product in entity_graph['product_issue_links'].keys():
            all_products.add(product)

        for product in all_products:
            product_embedding = self.embedder.encode(f"telecom product {product}").tolist()
            tx.run("""
                MERGE (p:Product {name: $name})
                SET p.embedding = $embedding,
                    p.type = 'product',
                    p.created_at = datetime()
                """,
                name=product,
                embedding=product_embedding
            )

        # Collect all unique issues
        all_issues = set()
        for issues in entity_graph['product_issue_links'].values():
            all_issues.update(issues)
        all_issues.update(entity_graph['issue_resolution_links'].keys())

        for issue in all_issues:
            issue_embedding = self.embedder.encode(f"telecom issue {issue}").tolist()
            tx.run("""
                MERGE (i:Issue {name: $name})
                SET i.embedding = $embedding,
                    i.type = 'issue',
                    i.created_at = datetime()
                """,
                name=issue,
                embedding=issue_embedding
            )

        # Collect all unique resolutions
        all_resolutions = set()
        for resolutions in entity_graph['issue_resolution_links'].values():
            all_resolutions.update(resolutions)

        for resolution in all_resolutions:
            resolution_embedding = self.embedder.encode(f"telecom resolution {resolution}").tolist()
            tx.run("""
                MERGE (r:Resolution {name: $name})
                SET r.embedding = $embedding,
                    r.type = 'resolution',
                    r.created_at = datetime()
                """,
                name=resolution,
                embedding=resolution_embedding
            )

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
        Advanced Graph RAG search that uses entity relationships.

        Args:
            query: User query
            limit: Maximum results to return

        Returns:
            Dictionary with graph-based results and entity relationships
        """
        if not self.driver:
            self.connect()

        # Extract entities from query
        query_entities = self.extract_entities(query)

        with self.driver.session() as session:
            # Find conversations that match entity patterns
            graph_results = session.run("""
                MATCH (conv:Conversation)-[:HAS_ISSUE]->(issue:Issue)
                WHERE ANY(entity IN $query_issues WHERE entity IN conv.issues)
                RETURN conv.id as conversation_id,
                       conv.client_message as client_message,
                       conv.agent_response as agent_response,
                       conv.issues as issues,
                       conv.resolutions as resolutions,
                       size([x IN conv.issues WHERE x IN $query_issues]) as issue_match_count
                ORDER BY issue_match_count DESC, conv.created_at DESC
                LIMIT $limit
                """,
                query_issues=query_entities['issues'],
                limit=limit
            )

            # Get related resolutions for matched issues
            related_resolutions = []
            if query_entities['issues']:
                resolution_results = session.run("""
                    MATCH (issue:Issue)-[:HAS_RESOLUTION]->(resolution:Resolution)
                    WHERE issue.name IN $query_issues
                    RETURN resolution.name as resolution,
                           count(*) as frequency
                    ORDER BY frequency DESC
                    LIMIT 5
                    """,
                    query_issues=query_entities['issues']
                )
                related_resolutions = [record.data() for record in resolution_results]

            return {
                'graph_conversations': [record.data() for record in graph_results],
                'related_resolutions': related_resolutions,
                'query_entities': query_entities
            }
