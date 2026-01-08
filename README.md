# Telecom Support RAG System

A Retrieval-Augmented Generation (RAG) system for telecom customer support using Neo4j graph database and LLMs.

## System Architecture

### 1. Data Ingestion & Processing

```python
# In pipeline.py
def run(self, query: str) -> str:
    # 1. Load and preprocess data
    raw_data = self.loader.load_data()
    processed_data = self.processor.process(raw_data)
    
    # 2. Build knowledge graph
    self.graph_builder.build_knowledge_graph(processed_data)
```

### 2. Graph Construction

```python
def build_knowledge_graph(self, conversations: List[Dict]):
    with self.driver.session() as session:
        # 2.1 Create conversation and message nodes
        session.write_transaction(
            self._create_conversation_nodes, 
            conversations
        )
        
        # 2.2 Extract and link entities
        entity_graph = self.build_entity_relationships(conversations)
        session.write_transaction(
            self._create_entity_relationships,
            entity_graph
        )
```

### 3. Search Flow

```python
def find_similar_issues(self, query: str, top_k: int = 5):
    # 3.1 First try Graph RAG search
    graph_results = self.graph_builder.search_graph_rag(query, top_k)
    
    # 3.2 Fallback to vector search if needed
    if not graph_results['graph_conversations']:
        vector_results = self.graph_builder.search_similar_issues(query, top_k)
        return self._format_vector_results(vector_results)
    
    return self._format_graph_results(graph_results)
```

### 4. Graph-based Search

```python
def search_graph_rag(self, query: str, limit: int):
    # 4.1 Get query embedding
    query_embedding = self.embedder.encode(query).tolist()
    
    # 4.2 Find similar messages using vector search
    similar_messages = self._find_similar_messages(
        query_embedding, 
        limit * 2  # Get more results for filtering
    )
    
    # 4.3 Get agent responses for top matches
    results = []
    with self.driver.session() as session:
        for msg in similar_messages:
            response = session.read_transaction(
                self._get_agent_response,
                msg['msg_id']
            )
            if response:
                results.append({
                    'client_message': msg['text'],
                    'agent_response': response,
                    'similarity': msg['similarity']
                })
    
    return results[:limit]
```

### 5. Response Generation

```python
def generate_response(self, query: str, context: List[Dict]) -> str:
    # 5.1 Format context
    context_str = "\n\n".join(
        f"User: {item['client_message']}\nAgent: {item['agent_response']}"
        for item in context
    )
    
    # 5.2 Generate response using LLM
    prompt = f"""Based on the following support conversations, provide a helpful response to: {query}
    
    Previous Conversations:
    {context_str}"""
    
    return self.llm.generate(prompt)
```

## Data Model

### Core Nodes
- **Conversation**
  - `id`: Unique identifier
  - `timestamp`: Creation time
  - `customer_id`: Customer identifier
  - `status`: Conversation status

- **Message**
  - `id`: Unique identifier
  - `text`: Message content
  - `type`: "client" or "agent"
  - `timestamp`: Message time
  - `embedding`: Vector embedding (384d)

- **Entity Types**
  - `Product`: e.g., "Router", "5G Plan"
  - `Issue`: e.g., "Slow Internet", "Billing Error"
  - `Resolution`: e.g., "Reset Router", "Update Plan"

### Relationships
- `(:Message)-[:PART_OF]->(:Conversation)`
- `(:Message)-[:MENTIONS]->(:Product|Issue|Resolution)`
- `(:Message)-[:RESPONDS_TO]->(:Message)`
- `(:Product)-[:HAS_ISSUE]->(:Issue)`
- `(:Issue)-[:HAS_RESOLUTION]->(:Resolution)`

## Complete Flow

1. **User Query** → `pipeline.run(query)` 
   - Loads and processes data
   - Builds/updates knowledge graph

2. **Search** → `vector_search.find_similar_issues(query)` 
   - Tries graph-based search first
   - Falls back to vector search if needed

3. **Graph Search** → `graph_builder.search_graph_rag()`
   - Encodes query to vector
   - Finds similar messages using vector similarity
   - Retrieves corresponding agent responses

4. **Response Generation** → `llm_generator.generate_response()`
   - Formats context
   - Generates final response using LLM

### Key Data Flow:
```
User Query 
  → Vector Embedding 
  → Similarity Search 
  → Context Retrieval 
  → LLM Generation 
  → Final Response
```

## Project Structure

```
telecom-rag/
├── config/
│   ├── config.yaml         # App configuration
│   └── secrets.yaml        # API keys and credentials
├── data/                   # Data storage
│   ├── raw/               # Raw conversation data
│   └── processed/         # Processed datasets
├── src/
│   ├── ingestion/         # Data loading and processing
│   ├── retrieval/         # Vector and graph search
│   ├── generation/        # LLM response generation
│   └── evaluation/        # System evaluation
├── app.py                 # Streamlit web interface
├── requirements.txt       # Python dependencies
└── README.md             # This documentation
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure Neo4j:
   - Install Neo4j Desktop or Server
   - Update `config/secrets.yaml` with your credentials

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Quick Start

### Automated Setup (Recommended)

```bash
# Run the automated setup script
python3 setup.py
```

This will:
- Check Python version compatibility
- Install all dependencies
- Optionally create a virtual environment
- Guide you through configuration

### Manual Setup

#### 1. Environment Setup

```bash
# Navigate to the project directory
cd rag-project-root

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

#### 2. Test Installation

```bash
# Run basic tests
python3 test_pipeline.py
```

#### 3. Configure API Keys

Edit `config/secrets.yaml` and add your API keys:

```yaml
openai:
  api_key: "your-openai-api-key-here"

neo4j:
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "your-neo4j-password-here"
```

#### 4. Set up Neo4j Database

1. **Download Neo4j:**
   - Neo4j Desktop: https://neo4j.com/download/
   - Or Neo4j Server for production

2. **Start Neo4j:**
   - Default credentials: `neo4j` / `neo4j`
   - Change password to match `secrets.yaml`

3. **Install GDS Plugin:**
   - For vector similarity operations
   - Available in Neo4j Desktop

#### 5. Data Ingestion

```bash
# Run data ingestion
python3 -c "from src.pipeline import RAGPipeline; p = RAGPipeline(); p.ingest_data()"
```

#### 6. Launch the Application

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

### Testing the Installation

```bash
# Run comprehensive tests
python3 test_pipeline.py

# Run usage examples
python3 example_usage.py
```

### Example Usage

```bash
# Interactive examples menu
python3 example_usage.py

# Specific demo
python3 example_usage.py basic    # Basic pipeline usage
python3 example_usage.py eval     # Evaluation metrics
python3 example_usage.py data     # Data ingestion overview
```

## Usage

### Web Interface

1. Open the Streamlit app in your browser
2. Enter a customer issue in the text area
3. Click "Get Help" to receive an AI-generated response

### Programmatic Usage

```python
from src.pipeline import RAGPipeline

pipeline = RAGPipeline()
response = pipeline.generate_response("My internet is very slow")
print(response)
```

### Evaluation

```python
from src.evaluation.run_eval import Evaluator
from src.evaluation.metrics import EvaluationMetrics

# Create test data
test_data = [
    {"query": "My internet is slow", "expected_answer": "Let me help troubleshoot..."}
]

# Run evaluation
evaluator = Evaluator(pipeline, EvaluationMetrics())
results = evaluator.evaluate(test_data)
```

## Configuration

### config.yaml

- `models.embedding_model`: Sentence transformer model for embeddings
- `models.llm_model`: OpenAI model for response generation
- `retrieval.top_k_similar`: Number of similar issues to retrieve
- `neo4j.*`: Neo4j connection settings

### secrets.yaml

- `openai.api_key`: OpenAI API key
- `neo4j.*`: Neo4j credentials
- `huggingface.token`: Optional Hugging Face token

## Dependencies

- **streamlit**: Web UI framework
- **datasets**: Hugging Face datasets library
- **neo4j**: Neo4j Python driver
- **sentence-transformers**: Text embedding models
- **openai**: OpenAI API client
- **pyyaml**: YAML configuration parsing

## Dataset

Uses the `eisenzopf/telecom-conversation-corpus` dataset from Hugging Face, which contains customer-agent conversation pairs from telecom support interactions.

**Local Storage**: The dataset is automatically downloaded and stored locally in `data/telecom_conversation_corpus/` for faster subsequent loading. The loader checks for local storage first before falling back to Hugging Face download.

**Dataset Statistics**:
- **Source**: Hugging Face (`eisenzopf/telecom-conversation-corpus`)
- **Size**: ~3.7 million conversation samples
- **Format**: Customer-agent message pairs
- **Domain**: Telecom customer support conversations

## Evaluation Metrics

- **Faithfulness**: How well the response is supported by retrieved context
- **Relevancy**: How well the response addresses the original query
- **Answer Correctness**: Semantic similarity to expected answer
- **Answer Relevance**: Additional relevance scoring

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src/
flake8 src/
```

## License

This project is for educational and research purposes. Please ensure compliance with API terms of service and data usage policies.
