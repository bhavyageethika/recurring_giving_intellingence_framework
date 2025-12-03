"""
Vector Memory Store

Provides semantic search and similarity matching for agent memory using vector embeddings.
Stores agent reasoning, learned facts, and episodic memories as embeddings for fast retrieval.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import structlog

from src.core.llm_service import get_llm_service

logger = structlog.get_logger()


@dataclass
class MemoryVector:
    """A vectorized memory entry."""
    memory_id: str
    agent_id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    memory_type: str = "episodic"  # episodic, semantic, reasoning, causal


class VectorMemoryStore:
    """
    Vector database for semantic memory storage and retrieval.
    
    Uses embeddings to enable:
    - Semantic search across agent memories
    - Similarity matching for learned patterns
    - Context retrieval for relevant past experiences
    - Cross-agent knowledge sharing
    """
    
    def __init__(self, collection_name: str = "agent_memory"):
        self.collection_name = collection_name
        self._llm = get_llm_service()
        self._vectors: Dict[str, MemoryVector] = {}  # In-memory for now, can be replaced with ChromaDB
        self._logger = logger.bind(component="vector_memory")
    
    async def store(
        self,
        agent_id: str,
        content: str,
        memory_type: str = "episodic",
        metadata: Optional[Dict[str, Any]] = None,
        use_fast_embedding: bool = False,
    ) -> str:
        """
        Store a memory with vector embedding.
        
        Args:
            agent_id: ID of the agent storing the memory
            content: Text content to store
            memory_type: Type of memory (episodic, semantic, reasoning, causal)
            metadata: Additional metadata (task_id, campaign_id, etc.)
            use_fast_embedding: If True, use hash-based embedding (faster but less semantic)
        
        Returns:
            memory_id: Unique ID of the stored memory
        """
        memory_id = str(uuid.uuid4())
        
        # Generate embedding (use fast hash-based for reasoning to avoid API calls)
        if use_fast_embedding or memory_type == "reasoning":
            # Fast hash-based embedding (no API call)
            import hashlib
            hash_bytes = hashlib.sha256(content.encode()).digest()
            embedding = [(hash_bytes[i % len(hash_bytes)] - 128) / 128.0 for i in range(384)]
        else:
            # Full semantic embedding (may be slower with API calls)
            embedding = await self._llm.generate_embedding(content)
        
        vector = MemoryVector(
            memory_id=memory_id,
            agent_id=agent_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            memory_type=memory_type,
        )
        
        self._vectors[memory_id] = vector
        self._logger.info(
            "memory_stored",
            agent_id=agent_id,
            memory_id=memory_id,
            memory_type=memory_type,
            content_preview=content[:100],
        )
        
        return memory_id
    
    async def search(
        self,
        query: str,
        agent_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 10,
        min_similarity: float = 0.7,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[MemoryVector, float]]:
        """
        Semantic search across stored memories.
        
        Args:
            query: Search query text
            agent_id: Filter by agent ID (None = all agents)
            memory_type: Filter by memory type (None = all types)
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold (0-1)
            metadata_filter: Filter by metadata key-value pairs
        
        Returns:
            List of (MemoryVector, similarity_score) tuples, sorted by similarity
        """
        # Generate query embedding
        query_embedding = await self._llm.generate_embedding(query)
        
        # Calculate similarities
        results = []
        for vector in self._vectors.values():
            # Apply filters
            if agent_id and vector.agent_id != agent_id:
                continue
            if memory_type and vector.memory_type != memory_type:
                continue
            if metadata_filter:
                if not all(
                    vector.metadata.get(k) == v
                    for k, v in metadata_filter.items()
                ):
                    continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, vector.embedding)
            
            if similarity >= min_similarity:
                results.append((vector, similarity))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:limit]
    
    async def find_similar_memories(
        self,
        memory_id: str,
        limit: int = 5,
        min_similarity: float = 0.7,
    ) -> List[Tuple[MemoryVector, float]]:
        """Find memories similar to a given memory."""
        if memory_id not in self._vectors:
            return []
        
        source_vector = self._vectors[memory_id]
        
        results = []
        for vid, vector in self._vectors.items():
            if vid == memory_id:
                continue
            
            similarity = self._cosine_similarity(
                source_vector.embedding,
                vector.embedding
            )
            
            if similarity >= min_similarity:
                results.append((vector, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    async def get_context(
        self,
        agent_id: str,
        current_goal: str,
        limit: int = 5,
    ) -> List[str]:
        """
        Get relevant context for an agent's current goal.
        
        Searches for similar past experiences, learned facts, and reasoning patterns.
        Optimized to return quickly even with many memories.
        """
        # If no memories, return immediately
        if len(self._vectors) == 0:
            return []
        
        # Only search if we have memories for this agent
        agent_memories = [v for v in self._vectors.values() if v.agent_id == agent_id]
        if len(agent_memories) == 0:
            return []
        
        # Use a shorter query for faster embedding (first 100 chars)
        short_goal = current_goal[:100] if len(current_goal) > 100 else current_goal
        
        results = await self.search(
            query=short_goal,
            agent_id=agent_id,
            limit=limit,
            min_similarity=0.6,
        )
        
        return [vector.content for vector, _ in results]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    async def store_reasoning_chain(
        self,
        agent_id: str,
        reasoning_steps: List[str],
        task_id: Optional[str] = None,
    ) -> List[str]:
        """Store a chain of reasoning steps as connected memories."""
        memory_ids = []
        
        for i, step in enumerate(reasoning_steps):
            metadata = {
                "task_id": task_id,
                "step_index": i,
                "total_steps": len(reasoning_steps),
            }
            
            memory_id = await self.store(
                agent_id=agent_id,
                content=step,
                memory_type="reasoning",
                metadata=metadata,
            )
            memory_ids.append(memory_id)
        
        return memory_ids
    
    def get_memory(self, memory_id: str) -> Optional[MemoryVector]:
        """Retrieve a memory by ID."""
        return self._vectors.get(memory_id)
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        if memory_id in self._vectors:
            del self._vectors[memory_id]
            return True
        return False


# Global vector memory store instance
_vector_store: Optional[VectorMemoryStore] = None


def get_vector_memory() -> VectorMemoryStore:
    """Get or create the global vector memory store."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorMemoryStore()
    return _vector_store

