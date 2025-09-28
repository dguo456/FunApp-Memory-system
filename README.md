# Sekai Memory System

A sophisticated multi-character memory system designed for narrative applications. Unlike traditional single-character chatbots, Sekai's Memory System manages three distinct types of memory relationships:

1. **Character-to-User (C2U)**: Each character maintains separate memories with the user
2. **Inter-Character (IC)**: Characters remember interactions with each other
3. **World Memory (WM)**: Characters retain memories about the evolving world state

## Architecture

The system uses a pragmatic and fast stack:

- **Python 3.12** - Modern Python with async support
- **PostgreSQL + pgvector** - Dense retrieval with vector similarity search + built-in full-text indexes for sparse retrieval
- **FastAPI** - Simple CRUD + retrieve API
- **SentenceTransformers/all-MiniLM-L6-v2** - Cheap, decent embeddings (384 dimensions)
- **Optional LLM integration** - For advanced contradiction checks in evaluation

## Features

### Core Functionality
- **Hybrid Retrieval**: Combines semantic (vector) and keyword (full-text) search
- **Memory Consistency Checking**: Detects contradictions and timeline issues
- **Multi-character Support**: Handles complex relationship dynamics
- **Chapter-based Processing**: Supports narrative progression over time
- **Importance Scoring**: Weighted memory retrieval based on significance

### Evaluation System
- **Precision/Recall Metrics**: Measures retrieval accuracy
- **Consistency Analysis**: Detects cross-talk and forgotten updates
- **System Health Monitoring**: Real-time metrics and diagnostics
- **Benchmark Suite**: Comprehensive testing framework

## Quick Start

### 1. Prerequisites

- Python 3.12+
- PostgreSQL 15+ with pgvector extension
- 4GB+ RAM (for embedding model)

### 2. Database Setup

```bash
# Install PostgreSQL and pgvector
# Ubuntu/Debian:
sudo apt-get install postgresql postgresql-contrib
sudo apt-get install postgresql-15-pgvector

# macOS (with Homebrew):
brew install postgresql
brew install pgvector

# Create database
createdb sekai_memory
psql sekai_memory -c "CREATE EXTENSION vector;"
```

### 3. Installation

```bash
# Clone and setup
git clone <repository-url>
cd sekai-memory

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://user:password@localhost:5432/sekai_memory"
export LOG_LEVEL="INFO"

# Initialize database schema
psql sekai_memory < db/ddl.sql

# Seed with characters and initial data
python db/seed.py
```

### 4. Load Data and Start Server

```bash
# Load chapter data from JSON (use --local if file is in current directory)
python scripts/load_json.py --local

# Start the API server
python app/main.py
# OR
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Verify Installation

Visit http://localhost:8000/docs for the interactive API documentation.

Check system health: http://localhost:8000/health

## API Usage

### Create a Memory

```python
import requests

memory_data = {
    "memory_type": "C2U",
    "character_id": 1,
    "related_entity_id": 1,
    "content": "I had an interesting conversation with the user about their preferences.",
    "summary": "User preference discussion",
    "chapter_number": 5,
    "context_tags": ["conversation", "preferences"],
    "importance_score": 6.5
}

response = requests.post("http://localhost:8000/api/v1/memories", json=memory_data)
print(response.json())
```

### Search Memories

```python
search_query = {
    "query_text": "Byleth Dimitri relationship",
    "memory_types": ["C2U", "IC"],
    "limit": 3,
    "similarity_threshold": 0.7
}

response = requests.post("http://localhost:8000/api/v1/memories/search", json=search_query)
results = response.json()

for result in results["results"]:
    print(f"Score: {result['similarity_score']:.3f}")
    print(f"Content: {result['memory']['content'][:100]}...")
    print("---")
```

### Get Character Memories

```python
# Get all memories for character ID 1
response = requests.get("http://localhost:8000/api/v1/characters/1/memories")
memories = response.json()

print(f"Character has {len(memories)} memories")
```

## Evaluation

### Run Precision/Recall Evaluation

```python
# Run comprehensive evaluation
python eval/runner.py

# Check consistency
response = requests.post("http://localhost:8000/api/v1/eval/consistency")
consistency_results = response.json()

# Get system metrics
response = requests.get("http://localhost:8000/api/v1/eval/system-metrics")
metrics = response.json()
```

### Monitor System Health

```python
response = requests.get("http://localhost:8000/health")
health = response.json()

print(f"Database: {health['database_status']}")
print(f"Embeddings: {health['embedding_service_status']}")
print(f"Total memories: {health['total_memories']}")
print(f"Consistency score: {health['consistency_score']:.3f}")
```

### Get System Statistics

```python
response = requests.get("http://localhost:8000/stats")
stats = response.json()

print(f"Total memories: {stats['total_memories']}")
print(f"Memories by type: {stats['memories_by_type']}")
print(f"Memories by character: {stats['memories_by_character']}")
print(f"Average importance: {stats['avg_importance_score']:.2f}")
```

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL="postgresql://user:password@localhost:5432/sekai_memory"

# Embedding model
EMBEDDING_MODEL="all-MiniLM-L6-v2"
VECTOR_DIMENSION=384

# System settings
MAX_MEMORY_CONTENT_LENGTH=2000
DEFAULT_SIMILARITY_THRESHOLD=0.7
LOG_LEVEL="INFO"

# Optional: LLM for advanced consistency checking
LLM_API_KEY="your-api-key"
LLM_MODEL="gpt-3.5-turbo"
```

### Database Configuration

For production deployments, tune PostgreSQL for vector operations:

```sql
-- postgresql.conf
shared_preload_libraries = 'vector'
max_connections = 100
shared_buffers = 1GB
effective_cache_size = 3GB
random_page_cost = 1.1  # For SSD storage

-- Optimize for vector operations
SET maintenance_work_mem = '512MB';
SET max_parallel_workers_per_gather = 2;
```

## Memory Types Explained

### Character-to-User (C2U) Memories
Each character maintains separate memories about their interactions with the user. This allows for personalized relationship development.

Example: Dimitri remembers his growing attraction to Byleth, while Felix remains suspicious of Byleth's motives.

### Inter-Character (IC) Memories
Characters remember interactions with each other, creating an interconnected web of relationships.

Example: Sylvain remembers Annette's trust and happiness, while also maintaining memories of his secret affair with Byleth.

### World Memory (WM)
Characters retain memories about the evolving world state and environmental changes.

Example: All characters remember the company health alert about the virus, but with different levels of concern and interpretation.

## Consistency Checking

The system automatically detects several types of inconsistencies:

- **Contradictory Statements**: "I trust them completely" vs "I'm suspicious of their motives"
- **Timeline Issues**: Events happening out of logical sequence
- **Relationship Contradictions**: Different characters having incompatible views of the same relationship
- **Cross-talk**: Information bleeding between characters who shouldn't know it

## Performance Considerations

### Vector Search Optimization
- Uses HNSW index for fast approximate nearest neighbor search
- Embedding dimension optimized at 384 for speed/accuracy balance
- Hybrid search combines vector and full-text for best results

### Memory Management
- Importance scoring prioritizes relevant memories
- Access count tracking for usage-based optimization
- Configurable limits prevent unbounded growth

### Scalability
- Async architecture supports high concurrency
- Database connection pooling
- Batch processing for bulk operations
- Background consistency checking

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-mock

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=core --cov=app
```

### Database Migrations

```bash
# Reset database (warning: deletes all data)
python db/seed.py --reset

# Seed only
python db/seed.py
```

### Adding New Characters

```sql
INSERT INTO characters (name, description) VALUES
('NewCharacter', 'Description of the new character');
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Checklist

- [ ] Set up PostgreSQL with pgvector
- [ ] Configure connection pooling
- [ ] Set up monitoring and logging
- [ ] Configure CORS appropriately
- [ ] Set up backup strategy
- [ ] Monitor embedding service performance
- [ ] Set up consistency checking schedules

## Troubleshooting

### Common Issues

**Database Connection Failed**
- Verify PostgreSQL is running
- Check DATABASE_URL format
- Ensure pgvector extension is installed

**Embedding Model Loading Slow**
- First load downloads ~80MB model
- Consider model caching in production
- Monitor memory usage (model requires ~400MB RAM)

**Vector Search Returns No Results**
- Check similarity threshold (try lowering to 0.5)
- Verify embeddings are being generated
- Check if memories have null embeddings

**Consistency Check False Positives**
- Review contradiction patterns in core/consistency.py
- Adjust severity thresholds
- Consider domain-specific rules

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built for the Sekai narrative system
- Uses Sentence Transformers for embeddings
- PostgreSQL pgvector for efficient vector operations
- FastAPI for modern async web framework