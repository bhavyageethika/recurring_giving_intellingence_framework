"""
Agent-to-Agent (A2A) Protocol Implementation.
Handles message routing, agent discovery, and coordination.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

import structlog

from src.core.base_agent import AgentCapability, AgentMessage, BaseAgent

logger = structlog.get_logger()


class MessageType(str, Enum):
    """Types of messages in the A2A protocol."""
    
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    EVENT = "event"
    ERROR = "error"


@dataclass
class AgentRegistration:
    """Registration information for an agent."""
    
    agent_id: str
    name: str
    capabilities: List[AgentCapability]
    endpoint: Optional[str] = None
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentRegistry:
    """
    Registry for agent discovery and management.
    Maintains a catalog of available agents and their capabilities.
    """
    
    def __init__(self):
        self._agents: Dict[str, AgentRegistration] = {}
        self._capability_index: Dict[AgentCapability, Set[str]] = defaultdict(set)
        self._logger = logger.bind(component="agent_registry")
    
    def register(
        self,
        agent_id: str,
        name: str,
        capabilities: List[AgentCapability],
        endpoint: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentRegistration:
        """
        Register an agent with the registry.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name
            capabilities: List of capabilities the agent provides
            endpoint: Optional network endpoint for remote agents
            metadata: Additional metadata
            
        Returns:
            The registration record
        """
        registration = AgentRegistration(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities,
            endpoint=endpoint,
            metadata=metadata or {},
        )
        
        self._agents[agent_id] = registration
        
        # Update capability index
        for capability in capabilities:
            self._capability_index[capability].add(agent_id)
        
        self._logger.info(
            "agent_registered",
            agent_id=agent_id,
            name=name,
            capabilities=[c.value for c in capabilities],
        )
        
        return registration
    
    def unregister(self, agent_id: str) -> bool:
        """
        Remove an agent from the registry.
        
        Args:
            agent_id: The agent to remove
            
        Returns:
            True if the agent was removed, False if not found
        """
        if agent_id not in self._agents:
            return False
        
        registration = self._agents.pop(agent_id)
        
        # Update capability index
        for capability in registration.capabilities:
            self._capability_index[capability].discard(agent_id)
        
        self._logger.info("agent_unregistered", agent_id=agent_id)
        return True
    
    def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get registration for a specific agent."""
        return self._agents.get(agent_id)
    
    def find_by_capability(self, capability: AgentCapability) -> List[str]:
        """Find all agents with a specific capability."""
        return list(self._capability_index.get(capability, set()))
    
    def find_by_capabilities(
        self,
        capabilities: List[AgentCapability],
        require_all: bool = False,
    ) -> List[str]:
        """
        Find agents by capabilities.
        
        Args:
            capabilities: List of capabilities to search for
            require_all: If True, agent must have all capabilities
            
        Returns:
            List of agent IDs matching the criteria
        """
        if not capabilities:
            return list(self._agents.keys())
        
        agent_sets = [self._capability_index.get(cap, set()) for cap in capabilities]
        
        if require_all:
            result = set.intersection(*agent_sets) if agent_sets else set()
        else:
            result = set.union(*agent_sets) if agent_sets else set()
        
        return list(result)
    
    def list_all(self) -> List[AgentRegistration]:
        """List all registered agents."""
        return list(self._agents.values())
    
    def update_heartbeat(self, agent_id: str) -> bool:
        """Update the heartbeat timestamp for an agent."""
        if agent_id in self._agents:
            self._agents[agent_id].last_heartbeat = datetime.utcnow()
            return True
        return False


class A2AProtocol:
    """
    Agent-to-Agent Protocol coordinator.
    Manages message routing between agents and orchestrates workflows.
    """
    
    def __init__(self):
        self.registry = AgentRegistry()
        self._agents: Dict[str, BaseAgent] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._logger = logger.bind(component="a2a_protocol")
        self._running = False
    
    def register_agent(self, agent) -> None:
        """
        Register a local agent with the protocol.
        
        Args:
            agent: The agent instance to register (BaseAgent or AutonomousAgent)
        """
        self._agents[agent.agent_id] = agent
        
        # Get capabilities - BaseAgent has it, AutonomousAgent might not
        capabilities = getattr(agent, "capabilities", [])
        
        self.registry.register(
            agent_id=agent.agent_id,
            name=agent.name,
            capabilities=capabilities,
            metadata={"local": True},
        )
        self._logger.info("local_agent_registered", agent_id=agent.agent_id)
    
    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the protocol."""
        self._agents.pop(agent_id, None)
        self.registry.unregister(agent_id)
    
    async def send_message(
        self,
        sender_id: str,
        recipient_id: str,
        action: str,
        payload: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> AgentMessage:
        """
        Send a message from one agent to another.
        
        Args:
            sender_id: The sending agent's ID
            recipient_id: The receiving agent's ID
            action: The action to perform
            payload: The message payload
            context: Optional context data
            timeout: Timeout in seconds
            
        Returns:
            The response message
        """
        message = AgentMessage(
            sender=sender_id,
            recipient=recipient_id,
            message_type=MessageType.REQUEST.value,
            action=action,
            payload=payload,
            context=context or {},
        )
        
        self._logger.info(
            "sending_message",
            message_id=message.id,
            sender=sender_id,
            recipient=recipient_id,
            action=action,
        )
        
        # Route to local agent
        if recipient_id in self._agents:
            recipient = self._agents[recipient_id]
            response = await asyncio.wait_for(
                recipient.process_message(message),
                timeout=timeout,
            )
            return response
        
        # Check if it's a remote agent
        registration = self.registry.get_agent(recipient_id)
        if registration and registration.endpoint:
            # TODO: Implement remote agent communication via HTTP/WebSocket
            raise NotImplementedError("Remote agent communication not yet implemented")
        
        raise ValueError(f"Unknown recipient agent: {recipient_id}")
    
    async def broadcast(
        self,
        sender_id: str,
        action: str,
        payload: Dict[str, Any],
        target_capability: Optional[AgentCapability] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[AgentMessage]:
        """
        Broadcast a message to multiple agents.
        
        Args:
            sender_id: The sending agent's ID
            action: The action to perform
            payload: The message payload
            target_capability: Optional capability filter
            context: Optional context data
            
        Returns:
            List of response messages
        """
        if target_capability:
            recipient_ids = self.registry.find_by_capability(target_capability)
        else:
            recipient_ids = list(self._agents.keys())
        
        # Exclude sender
        recipient_ids = [r for r in recipient_ids if r != sender_id]
        
        self._logger.info(
            "broadcasting_message",
            sender=sender_id,
            action=action,
            recipient_count=len(recipient_ids),
        )
        
        # Send to all recipients concurrently
        tasks = [
            self.send_message(sender_id, recipient_id, action, payload, context)
            for recipient_id in recipient_ids
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid responses
        valid_responses = [r for r in responses if isinstance(r, AgentMessage)]
        return valid_responses
    
    def subscribe_event(self, event_type: str, handler: Callable) -> None:
        """Subscribe to a specific event type."""
        self._event_handlers[event_type].append(handler)
    
    async def emit_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        source: Optional[str] = None,
    ) -> None:
        """Emit an event to all subscribers."""
        handlers = self._event_handlers.get(event_type, [])
        
        self._logger.info(
            "emitting_event",
            event_type=event_type,
            handler_count=len(handlers),
            source=source,
        )
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_type, payload, source)
                else:
                    handler(event_type, payload, source)
            except Exception as e:
                self._logger.error(
                    "event_handler_error",
                    event_type=event_type,
                    error=str(e),
                )
    
    async def initialize_all_agents(self) -> None:
        """Initialize all registered local agents."""
        self._logger.info("initializing_all_agents", count=len(self._agents))
        
        tasks = [agent.initialize() for agent in self._agents.values()]
        await asyncio.gather(*tasks)
        
        self._logger.info("all_agents_initialized")
    
    async def shutdown_all_agents(self) -> None:
        """Shutdown all registered local agents."""
        self._logger.info("shutting_down_all_agents")
        
        tasks = [agent.shutdown() for agent in self._agents.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self._logger.info("all_agents_shutdown")
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get a local agent by ID."""
        return self._agents.get(agent_id)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents with their cards."""
        return [agent.get_agent_card() for agent in self._agents.values()]


# Global protocol instance
_protocol: Optional[A2AProtocol] = None


def get_protocol() -> A2AProtocol:
    """Get or create the global A2A protocol instance."""
    global _protocol
    if _protocol is None:
        _protocol = A2AProtocol()
    return _protocol

# Alias for consistency
get_a2a_protocol = get_protocol

