"""
Graph Memory Store

Provides graph-based storage for causal relationships, agent interactions, and knowledge graphs.
Uses NetworkX for in-memory graph operations (can be replaced with Neo4j for production).
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import structlog

try:
    import networkx as nx
except ImportError:
    nx = None
    logger = structlog.get_logger()
    logger.warning("networkx not installed, graph memory will use dict-based storage")

logger = structlog.get_logger()


class RelationshipType(str, Enum):
    """Types of relationships in the knowledge graph."""
    # Causal relationships
    CAUSES = "causes"
    LEADS_TO = "leads_to"
    INFLUENCES = "influences"
    PREDICTS = "predicts"
    
    # Agent relationships
    COLLABORATES_WITH = "collaborates_with"
    REQUESTS_FROM = "requests_from"
    PROVIDES_TO = "provides_to"
    
    # Entity relationships
    RELATED_TO = "related_to"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"
    DEPENDS_ON = "depends_on"
    
    # Temporal relationships
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    
    # Learning relationships
    LEARNS_FROM = "learns_from"
    IMPROVES = "improves"


@dataclass
class GraphNode:
    """A node in the knowledge graph."""
    node_id: str
    node_type: str  # agent, campaign, donor, task, outcome, pattern, etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GraphEdge:
    """An edge (relationship) in the knowledge graph."""
    edge_id: str
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0  # Strength of relationship
    timestamp: datetime = field(default_factory=datetime.utcnow)


class GraphMemoryStore:
    """
    Graph database for causal relationships and knowledge graphs.
    
    Stores:
    - Causal chains (e.g., "high quality campaign → increased donations → campaign success")
    - Agent interaction patterns
    - Learning relationships (what works, what doesn't)
    - Temporal sequences (task dependencies, workflow steps)
    - Entity relationships (campaigns, donors, communities)
    """
    
    def __init__(self):
        if nx:
            self._graph = nx.MultiDiGraph()  # Directed multigraph for multiple relationship types
        else:
            self._graph = None
            self._nodes: Dict[str, GraphNode] = {}
            self._edges: Dict[str, GraphEdge] = {}
            self._adjacency: Dict[str, Dict[str, List[str]]] = {}  # source -> target -> [edge_ids]
        
        self._logger = logger.bind(component="graph_memory")
    
    def add_node(
        self,
        node_type: str,
        properties: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> str:
        """
        Add a node to the graph.
        
        Args:
            node_type: Type of node (agent, campaign, donor, task, outcome, etc.)
            properties: Node properties
            node_id: Optional custom ID (auto-generated if not provided)
        
        Returns:
            node_id: Unique ID of the created node
        """
        if node_id is None:
            node_id = str(uuid.uuid4())
        
        node = GraphNode(
            node_id=node_id,
            node_type=node_type,
            properties=properties or {},
        )
        
        if nx and self._graph:
            self._graph.add_node(node_id, **{
                "node_type": node_type,
                **node.properties,
                "timestamp": node.timestamp.isoformat(),
            })
        else:
            self._nodes[node_id] = node
            if node_id not in self._adjacency:
                self._adjacency[node_id] = {}
        
        self._logger.info("node_added", node_id=node_id, node_type=node_type)
        return node_id
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: RelationshipType,
        properties: Optional[Dict[str, Any]] = None,
        weight: float = 1.0,
        edge_id: Optional[str] = None,
    ) -> str:
        """
        Add an edge (relationship) to the graph.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_type: Type of relationship
            properties: Edge properties
            weight: Relationship strength (0-1)
            edge_id: Optional custom ID
        
        Returns:
            edge_id: Unique ID of the created edge
        """
        if edge_id is None:
            edge_id = str(uuid.uuid4())
        
        edge = GraphEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            properties=properties or {},
            weight=weight,
        )
        
        if nx and self._graph:
            self._graph.add_edge(
                source_id,
                target_id,
                key=edge_id,
                relationship_type=relationship_type.value,
                weight=weight,
                **edge.properties,
                timestamp=edge.timestamp.isoformat(),
            )
        else:
            self._edges[edge_id] = edge
            if source_id not in self._adjacency:
                self._adjacency[source_id] = {}
            if target_id not in self._adjacency[source_id]:
                self._adjacency[source_id][target_id] = []
            self._adjacency[source_id][target_id].append(edge_id)
        
        self._logger.info(
            "edge_added",
            edge_id=edge_id,
            source=source_id,
            target=target_id,
            relationship=relationship_type.value,
        )
        return edge_id
    
    def record_causal_chain(
        self,
        chain: List[Tuple[str, str, RelationshipType]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Record a causal chain: A → B → C → D.
        
        Args:
            chain: List of (source_id, target_id, relationship_type) tuples
            metadata: Metadata to attach to the chain
        
        Returns:
            List of edge IDs created
        """
        edge_ids = []
        
        for source_id, target_id, rel_type in chain:
            edge_props = metadata.copy() if metadata else {}
            edge_props["chain_index"] = len(edge_ids)
            
            edge_id = self.add_edge(
                source_id=source_id,
                target_id=target_id,
                relationship_type=rel_type,
                properties=edge_props,
            )
            edge_ids.append(edge_id)
        
        return edge_ids
    
    def find_causal_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
        relationship_types: Optional[List[RelationshipType]] = None,
    ) -> List[List[str]]:
        """
        Find all causal paths from source to target.
        
        Returns:
            List of paths, where each path is a list of node IDs
        """
        if nx and self._graph:
            try:
                paths = list(nx.all_simple_paths(
                    self._graph,
                    source_id,
                    target_id,
                    cutoff=max_depth,
                ))
                return paths
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return []
        else:
            # Simple DFS for dict-based storage
            paths = []
            visited = set()
            
            def dfs(current: str, path: List[str]):
                if current == target_id:
                    paths.append(path.copy())
                    return
                
                if len(path) >= max_depth:
                    return
                
                if current in visited:
                    return
                
                visited.add(current)
                
                if current in self._adjacency:
                    for neighbor_id, edge_ids in self._adjacency[current].items():
                        # Check relationship type filter
                        if relationship_types:
                            edge = self._edges.get(edge_ids[0])
                            if edge and edge.relationship_type not in relationship_types:
                                continue
                        
                        path.append(neighbor_id)
                        dfs(neighbor_id, path)
                        path.pop()
                
                visited.remove(current)
            
            dfs(source_id, [source_id])
            return paths
    
    def get_causal_predecessors(
        self,
        node_id: str,
        relationship_type: Optional[RelationshipType] = None,
        max_depth: int = 3,
    ) -> List[str]:
        """Get all nodes that causally influence this node."""
        if nx and self._graph:
            predecessors = []
            for depth in range(1, max_depth + 1):
                for pred in nx.ancestors(self._graph, node_id):
                    # Check relationship type if specified
                    if relationship_type:
                        edges = self._graph.get_edge_data(pred, node_id)
                        if edges:
                            for edge_data in edges.values():
                                if edge_data.get("relationship_type") == relationship_type.value:
                                    predecessors.append(pred)
                                    break
                    else:
                        predecessors.append(pred)
            return list(set(predecessors))
        else:
            # Simple BFS backwards
            predecessors = []
            queue = [(node_id, 0)]
            visited = {node_id}
            
            while queue:
                current, depth = queue.pop(0)
                if depth >= max_depth:
                    continue
                
                # Find all nodes that have edges to current
                for source_id, targets in self._adjacency.items():
                    if current in targets:
                        edge_ids = targets[current]
                        for edge_id in edge_ids:
                            edge = self._edges.get(edge_id)
                            if edge:
                                if relationship_type and edge.relationship_type != relationship_type:
                                    continue
                                if source_id not in visited:
                                    visited.add(source_id)
                                    predecessors.append(source_id)
                                    queue.append((source_id, depth + 1))
            
            return predecessors
    
    def get_causal_successors(
        self,
        node_id: str,
        relationship_type: Optional[RelationshipType] = None,
        max_depth: int = 3,
    ) -> List[str]:
        """Get all nodes that this node causally influences."""
        if nx and self._graph:
            successors = []
            for depth in range(1, max_depth + 1):
                for succ in nx.descendants(self._graph, node_id):
                    # Check relationship type if specified
                    if relationship_type:
                        edges = self._graph.get_edge_data(node_id, succ)
                        if edges:
                            for edge_data in edges.values():
                                if edge_data.get("relationship_type") == relationship_type.value:
                                    successors.append(succ)
                                    break
                    else:
                        successors.append(succ)
            return list(set(successors))
        else:
            # Simple BFS forwards
            successors = []
            queue = [(node_id, 0)]
            visited = {node_id}
            
            while queue:
                current, depth = queue.pop(0)
                if depth >= max_depth:
                    continue
                
                if current in self._adjacency:
                    for target_id, edge_ids in self._adjacency[current].items():
                        for edge_id in edge_ids:
                            edge = self._edges.get(edge_id)
                            if edge:
                                if relationship_type and edge.relationship_type != relationship_type:
                                    continue
                                if target_id not in visited:
                                    visited.add(target_id)
                                    successors.append(target_id)
                                    queue.append((target_id, depth + 1))
            
            return successors
    
    def record_agent_interaction(
        self,
        agent_a_id: str,
        agent_b_id: str,
        interaction_type: str,
        outcome: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record an interaction between two agents."""
        # Ensure agent nodes exist
        if not self.node_exists(agent_a_id):
            self.add_node("agent", {"agent_id": agent_a_id}, node_id=agent_a_id)
        if not self.node_exists(agent_b_id):
            self.add_node("agent", {"agent_id": agent_b_id}, node_id=agent_b_id)
        
        edge_props = metadata or {}
        edge_props["interaction_type"] = interaction_type
        if outcome:
            edge_props["outcome"] = outcome
        
        return self.add_edge(
            source_id=agent_a_id,
            target_id=agent_b_id,
            relationship_type=RelationshipType.COLLABORATES_WITH,
            properties=edge_props,
        )
    
    def find_similar_patterns(
        self,
        pattern_nodes: List[str],
        min_match: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Find similar patterns in the graph.
        
        Args:
            pattern_nodes: List of node IDs forming a pattern
            min_match: Minimum number of nodes that must match
        
        Returns:
            List of similar patterns with their contexts
        """
        # This is a simplified implementation
        # In production, use graph isomorphism or subgraph matching
        similar = []
        
        # Find nodes that appear in similar contexts
        for node_id in pattern_nodes:
            if nx and self._graph:
                neighbors = list(self._graph.neighbors(node_id))
                # Find other nodes with similar neighbor sets
                for other_node in self._graph.nodes():
                    if other_node == node_id:
                        continue
                    other_neighbors = list(self._graph.neighbors(other_node))
                    overlap = len(set(neighbors) & set(other_neighbors))
                    if overlap >= min_match:
                        similar.append({
                            "pattern_node": node_id,
                            "similar_node": other_node,
                            "overlap": overlap,
                        })
        
        return similar
    
    def node_exists(self, node_id: str) -> bool:
        """Check if a node exists."""
        if nx and self._graph:
            return node_id in self._graph
        return node_id in self._nodes
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        if nx and self._graph:
            if node_id in self._graph:
                data = self._graph.nodes[node_id]
                return GraphNode(
                    node_id=node_id,
                    node_type=data.get("node_type", "unknown"),
                    properties={k: v for k, v in data.items() if k not in ["node_type", "timestamp"]},
                )
            return None
        return self._nodes.get(node_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if nx and self._graph:
            return {
                "nodes": self._graph.number_of_nodes(),
                "edges": self._graph.number_of_edges(),
                "density": nx.density(self._graph),
                "is_connected": nx.is_weakly_connected(self._graph),
            }
        else:
            return {
                "nodes": len(self._nodes),
                "edges": len(self._edges),
                "density": len(self._edges) / max(len(self._nodes) * (len(self._nodes) - 1), 1),
            }


# Global graph memory store instance
_graph_store: Optional[GraphMemoryStore] = None


def get_graph_memory() -> GraphMemoryStore:
    """Get or create the global graph memory store."""
    global _graph_store
    if _graph_store is None:
        _graph_store = GraphMemoryStore()
    return _graph_store





