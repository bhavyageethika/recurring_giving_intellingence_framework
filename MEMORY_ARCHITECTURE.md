# Agent Memory Architecture

## Overview

The multi-agent system uses a hybrid memory architecture combining:
1. **Vector Memory** - Semantic search and similarity matching
2. **Graph Memory** - Causal relationships and knowledge graphs
3. **In-Memory Working Memory** - Current context and task state

## Vector Memory (`src/core/vector_memory.py`)

### Purpose
- **Semantic Search**: Find similar past experiences using embeddings
- **Context Retrieval**: Retrieve relevant memories for current goals
- **Cross-Agent Learning**: Share knowledge across agents via embeddings
- **Pattern Recognition**: Identify similar reasoning patterns and outcomes

### Use Cases

#### 1. Episodic Memory
Stores past interactions, goals, and outcomes:
```python
await vector_store.store(
    agent_id="profiler_agent",
    content="Successfully built donor profile for user_123",
    memory_type="episodic",
    metadata={"donor_id": "user_123", "outcome": "success"}
)
```

#### 2. Reasoning Memory
Stores reasoning chains for future reference:
```python
await vector_store.store_reasoning_chain(
    agent_id="intelligence_agent",
    reasoning_steps=[
        "Analyzed campaign quality",
        "Identified key weaknesses",
        "Generated improvement suggestions"
    ],
    task_id="task_456"
)
```

#### 3. Semantic Memory
Stores learned facts and patterns:
```python
await vector_store.store(
    agent_id="curator_agent",
    content="Campaigns with regular updates have 40% higher success rate",
    memory_type="semantic",
    metadata={"pattern": "update_frequency", "metric": "success_rate"}
)
```

### Search Capabilities

#### Context-Aware Retrieval
```python
# Get relevant past experiences for current goal
contexts = await vector_store.get_context(
    agent_id="profiler_agent",
    current_goal="Build donor profile",
    limit=5
)
```

#### Similarity Search
```python
# Find similar memories
results = await vector_store.search(
    query="donor with high engagement",
    agent_id="profiler_agent",
    memory_type="episodic",
    min_similarity=0.7,
    limit=10
)
```

## Graph Memory (`src/core/graph_memory.py`)

### Purpose
- **Causal Relationships**: Track cause-and-effect chains
- **Agent Interactions**: Record collaboration patterns
- **Knowledge Graphs**: Build structured knowledge representations
- **Pattern Discovery**: Find similar causal patterns

### Relationship Types

#### Causal Relationships
- `CAUSES`: Direct causation (A causes B)
- `LEADS_TO`: Sequential causation (A leads to B)
- `INFLUENCES`: Indirect influence (A influences B)
- `PREDICTS`: Predictive relationship (A predicts B)

#### Agent Relationships
- `COLLABORATES_WITH`: Agent collaboration
- `REQUESTS_FROM`: Agent requesting service
- `PROVIDES_TO`: Agent providing service

#### Entity Relationships
- `RELATED_TO`: General relationship
- `SIMILAR_TO`: Similarity relationship
- `PART_OF`: Composition relationship
- `DEPENDS_ON`: Dependency relationship

#### Temporal Relationships
- `PRECEDES`: Temporal ordering
- `FOLLOWS`: Temporal sequence

#### Learning Relationships
- `LEARNS_FROM`: Learning relationship
- `IMPROVES`: Improvement relationship

### Use Cases

#### 1. Causal Chain Recording
```python
# Record: High quality campaign → Increased donations → Campaign success
graph_store.record_causal_chain([
    ("campaign_123", "donation_increase", RelationshipType.CAUSES),
    ("donation_increase", "campaign_success", RelationshipType.LEADS_TO),
], metadata={"campaign_id": "campaign_123"})
```

#### 2. Agent Interaction Tracking
```python
# Record: Profiler agent requests legitimacy check from Matcher agent
graph_store.record_agent_interaction(
    agent_a_id="profiler_agent",
    agent_b_id="matcher_agent",
    interaction_type="legitimacy_check",
    outcome="success",
    metadata={"campaign_id": "campaign_123"}
)
```

#### 3. Causal Path Finding
```python
# Find: What leads to campaign success?
paths = graph_store.find_causal_paths(
    source_id="high_quality_campaign",
    target_id="campaign_success",
    max_depth=5,
    relationship_types=[RelationshipType.CAUSES, RelationshipType.LEADS_TO]
)
```

#### 4. Causal Predecessors/Successors
```python
# What causes campaign success?
predecessors = graph_store.get_causal_predecessors(
    node_id="campaign_success",
    relationship_type=RelationshipType.CAUSES,
    max_depth=3
)

# What does high quality lead to?
successors = graph_store.get_causal_successors(
    node_id="high_quality_campaign",
    relationship_type=RelationshipType.LEADS_TO,
    max_depth=3
)
```

## Integration with Agents

### Automatic Memory Storage

The `AutonomousAgent` class automatically:
1. **Stores goals** in vector memory when `run()` is called
2. **Stores reasoning steps** in vector memory when `think()` tool is used
3. **Retrieves relevant context** from vector memory when building context summaries
4. **Queries graph memory** for causal insights when available

### Example: Campaign Intelligence Agent

```python
# Agent automatically stores analysis in vector memory
intelligence = await intelligence_agent.analyze_campaign(campaign_data)

# Later, similar campaigns can be found via semantic search
similar_analyses = await vector_store.search(
    query="campaign with low quality score",
    agent_id="intelligence_agent",
    memory_type="episodic"
)

# Causal relationships are recorded in graph memory
graph_store.record_causal_chain([
    ("low_quality_score", "low_donations", RelationshipType.PREDICTS),
    ("low_donations", "campaign_failure", RelationshipType.LEADS_TO),
])
```

## Architecture Benefits

### 1. Semantic Understanding
- Vector embeddings capture semantic meaning
- Similar concepts are found even with different wording
- Cross-agent knowledge sharing via embeddings

### 2. Causal Reasoning
- Graph structure enables causal chain traversal
- Predict outcomes based on causal patterns
- Learn from cause-and-effect relationships

### 3. Context Enrichment
- Agents retrieve relevant past experiences
- Context summaries include vector and graph insights
- Better decision-making with historical context

### 4. Pattern Discovery
- Find similar patterns across different contexts
- Identify recurring causal chains
- Learn from successful and failed patterns

## Performance Optimizations

### Vector Memory
- **In-memory storage** for fast access (can be replaced with ChromaDB/Pinecone)
- **Cosine similarity** for efficient similarity search
- **Metadata filtering** for targeted searches
- **Embedding caching** to avoid redundant LLM calls

### Graph Memory
- **NetworkX** for in-memory graph operations (can be replaced with Neo4j)
- **Efficient path finding** using graph algorithms
- **Relationship type filtering** for targeted queries
- **Graph statistics** for monitoring and optimization

## Future Enhancements

### Production-Ready Backends
1. **Vector DB**: Replace in-memory with ChromaDB, Pinecone, or Weaviate
2. **Graph DB**: Replace NetworkX with Neo4j or ArangoDB
3. **Hybrid Search**: Combine vector and graph queries for richer results

### Advanced Features
1. **Temporal Memory**: Time-aware memory retrieval
2. **Multi-Modal Memory**: Store images, audio, structured data
3. **Memory Compression**: Summarize old memories to save space
4. **Memory Forgetting**: Remove irrelevant or outdated memories
5. **Memory Attribution**: Track which agent learned what

## Usage Examples

### Recording a Successful Pattern
```python
# Vector: Store the successful outcome
await vector_store.store(
    agent_id="curator_agent",
    content="Recurring opportunity identified: Medical emergency campaign with ongoing need",
    memory_type="episodic",
    metadata={"campaign_id": "campaign_123", "outcome": "success"}
)

# Graph: Record the causal chain
graph_store.record_causal_chain([
    ("medical_emergency", "ongoing_need", RelationshipType.CAUSES),
    ("ongoing_need", "recurring_opportunity", RelationshipType.LEADS_TO),
    ("recurring_opportunity", "donor_engagement", RelationshipType.INFLUENCES),
])
```

### Retrieving Relevant Context
```python
# When analyzing a new campaign, retrieve similar past analyses
similar_campaigns = await vector_store.search(
    query="medical emergency campaign with ongoing need",
    agent_id="curator_agent",
    memory_type="episodic",
    limit=5
)

# Find what typically leads to recurring opportunities
causal_paths = graph_store.find_causal_paths(
    source_id="medical_emergency",
    target_id="recurring_opportunity",
    max_depth=3
)
```

## Configuration

Both memory stores are initialized automatically when agents are created. They use in-memory storage by default for development, but can be configured to use external databases:

```python
# Vector memory with ChromaDB
from chromadb import Client
vector_store = VectorMemoryStore(collection_name="agent_memory", chroma_client=client)

# Graph memory with Neo4j
from neo4j import GraphDatabase
graph_store = GraphMemoryStore(neo4j_driver=driver)
```





