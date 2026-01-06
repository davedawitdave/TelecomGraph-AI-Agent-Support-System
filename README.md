# RAG Telecom Support Assistant

A Retrieval-Augmented Generation (RAG) system for telecom customer support that uses Neo4j graph database and OpenAI GPT models to provide intelligent responses based on historical conversation data.

## Features

- **Data Ingestion**: Load and process telecom conversation datasets from Hugging Face
- **Graph Database**: Store conversation pairs in Neo4j with vector embeddings
- **Vector Search**: Find similar customer issues using semantic similarity
- **LLM Generation**: Generate contextual responses using GPT models
- **Evaluation**: Comprehensive metrics including faithfulness and relevancy
- **Streamlit UI**: User-friendly web interface for customer support

## Project Structure

```
rag-project-root/
├── config/
│   ├── config.yaml          # Application configuration
│   └── secrets.yaml         # API keys and secrets
├── data/
│   ├── 01_cache/            # Hugging Face dataset cache
│   └── telecom_conversation_corpus/  # Local dataset storage
├── src/
│   ├── ingestion/
│   │   ├── loader.py        # Local/remote dataset loading with fallback
│   │   ├── processor.py     # Conversation pair extraction
│   │   └── graph_builder.py # Neo4j graph construction
│   ├── retrieval/
│   │   └── search.py        # Vector similarity search
│   ├── generation/
│   │   └── llm.py           # LLM response generation
│   ├── evaluation/          # Evaluation framework
│   │   ├── metrics.py       # Faithfulness, relevancy metrics
│   │   └── run_eval.py      # Evaluation pipeline
│   └── pipeline.py          # Main RAG orchestrator
├── app.py                   # Streamlit web application
├── setup.py                 # Automated setup script
├── test_pipeline.py         # Component testing script
├── example_usage.py         # Usage examples and demos
├── .gitignore              # Git ignore rules
├── README.md               # This documentation
└── requirements.txt        # Python dependencies
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
