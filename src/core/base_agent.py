"""
Base Agent class that all specialized agents inherit from.
Implements the Agent-to-Agent (A2A) communication protocol.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar, Generic
from uuid import uuid4

import structlog

logger = structlog.get_logger()


class AgentCapability(str, Enum):
    """Capabilities that agents can advertise."""
    
    DONOR_PROFILING = "donor_profiling"
    CAMPAIGN_MATCHING = "campaign_matching"
    COMMUNITY_DISCOVERY = "community_discovery"
    RECURRING_CURATION = "recurring_curation"
    GIVING_CIRCLE = "giving_circle"
    ENGAGEMENT = "engagement"
    EMBEDDING_GENERATION = "embedding_generation"
    SIMILARITY_SEARCH = "similarity_search"


@dataclass
class AgentMessage:
    """Message format for agent-to-agent communication."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    sender: str = ""
    recipient: str = ""
    message_type: str = "request"  # request, response, broadcast, event
    action: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    priority: int = 5  # 1 (highest) to 10 (lowest)
    ttl_seconds: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type,
            "action": self.action,
            "payload": self.payload,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "priority": self.priority,
            "ttl_seconds": self.ttl_seconds,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary."""
        data = data.copy()
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class AgentState:
    """Represents the current state of an agent."""
    
    is_ready: bool = False
    is_processing: bool = False
    current_task: Optional[str] = None
    last_activity: Optional[datetime] = None
    error_count: int = 0
    processed_count: int = 0


T = TypeVar("T")


class BaseAgent(ABC, Generic[T]):
    """
    Abstract base class for all agents in the Giving Intelligence system.
    
    Each agent has:
    - A unique identifier
    - A set of capabilities it provides
    - Methods for processing requests and communicating with other agents
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        capabilities: List[AgentCapability],
    ):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.state = AgentState()
        self._message_handlers: Dict[str, callable] = {}
        self._logger = logger.bind(agent_id=agent_id, agent_name=name)
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        self._message_handlers["ping"] = self._handle_ping
        self._message_handlers["status"] = self._handle_status
        self._message_handlers["capabilities"] = self._handle_capabilities
    
    async def _handle_ping(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle ping requests."""
        return {"status": "alive", "agent_id": self.agent_id}
    
    async def _handle_status(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle status requests."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "state": {
                "is_ready": self.state.is_ready,
                "is_processing": self.state.is_processing,
                "current_task": self.state.current_task,
                "last_activity": self.state.last_activity.isoformat() if self.state.last_activity else None,
                "error_count": self.state.error_count,
                "processed_count": self.state.processed_count,
            }
        }
    
    async def _handle_capabilities(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle capabilities query."""
        return {
            "agent_id": self.agent_id,
            "capabilities": [cap.value for cap in self.capabilities],
        }
    
    def register_handler(self, action: str, handler: callable) -> None:
        """Register a handler for a specific action."""
        self._message_handlers[action] = handler
        self._logger.info("handler_registered", action=action)
    
    async def initialize(self) -> None:
        """Initialize the agent (load models, connect to services, etc.)."""
        self._logger.info("initializing_agent")
        await self._initialize()
        self.state.is_ready = True
        self._logger.info("agent_initialized")
    
    @abstractmethod
    async def _initialize(self) -> None:
        """Agent-specific initialization logic."""
        pass
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Process an incoming message and return a response.
        
        Args:
            message: The incoming agent message
            
        Returns:
            Response message
        """
        self.state.is_processing = True
        self.state.current_task = message.action
        self.state.last_activity = datetime.utcnow()
        
        self._logger.info(
            "processing_message",
            message_id=message.id,
            action=message.action,
            sender=message.sender,
        )
        
        try:
            # Check if we have a handler for this action
            handler = self._message_handlers.get(message.action)
            
            if handler:
                result = await handler(message)
            else:
                # Fall back to the generic process method
                result = await self._process(message)
            
            self.state.processed_count += 1
            
            # Create response message
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                message_type="response",
                action=f"{message.action}_response",
                payload=result,
                context=message.context,
                correlation_id=message.id,
            )
            
            self._logger.info(
                "message_processed",
                message_id=message.id,
                response_id=response.id,
            )
            
            return response
            
        except Exception as e:
            self.state.error_count += 1
            self._logger.error(
                "message_processing_failed",
                message_id=message.id,
                error=str(e),
            )
            
            return AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                message_type="error",
                action=f"{message.action}_error",
                payload={"error": str(e), "error_type": type(e).__name__},
                context=message.context,
                correlation_id=message.id,
            )
            
        finally:
            self.state.is_processing = False
            self.state.current_task = None
    
    @abstractmethod
    async def _process(self, message: AgentMessage) -> Dict[str, Any]:
        """
        Agent-specific message processing logic.
        
        Args:
            message: The incoming message
            
        Returns:
            Dictionary containing the response payload
        """
        pass
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the agent."""
        self._logger.info("shutting_down_agent")
        await self._shutdown()
        self.state.is_ready = False
        self._logger.info("agent_shutdown_complete")
    
    async def _shutdown(self) -> None:
        """Agent-specific shutdown logic. Override if needed."""
        pass
    
    def get_agent_card(self) -> Dict[str, Any]:
        """
        Get the agent's card (metadata for discovery).
        
        Returns:
            Dictionary containing agent metadata
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": [cap.value for cap in self.capabilities],
            "is_ready": self.state.is_ready,
            "version": "1.0.0",
        }





