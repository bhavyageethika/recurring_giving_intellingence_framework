"""
Agent Collaboration Utilities

Provides helper functions for agents to easily collaborate with each other
via the A2A protocol.
"""

from typing import Any, Dict, Optional
import structlog

from src.core.a2a_protocol import get_protocol
from src.core.base_agent import AgentMessage

logger = structlog.get_logger()


class AgentCollaborator:
    """
    Helper class for agents to collaborate with other agents.
    
    Provides easy-to-use methods for common agent-to-agent interactions.
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.protocol = get_protocol()
        self._logger = logger.bind(agent_id=agent_id, component="agent_collaboration")
    
    async def ask_profiler_for_insights(self, donor_id: str) -> Dict[str, Any]:
        """Ask Donor Affinity Profiler for donor insights."""
        try:
            response = await self.protocol.send_message(
                sender_id=self.agent_id,
                recipient_id="donor_affinity_profiler",
                action="get_donor_insights",
                payload={"donor_id": donor_id},
            )
            # Handle both AgentMessage and dict responses
            if hasattr(response, "payload"):
                return response.payload.get("insights", {})
            elif isinstance(response, dict):
                return response.get("insights", {})
            return {}
        except Exception as e:
            self._logger.warning("profiler_request_failed", error=str(e))
            return {}
    
    async def ask_matcher_for_legitimacy(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ask Campaign Matching Engine to evaluate campaign legitimacy."""
        try:
            response = await self.protocol.send_message(
                sender_id=self.agent_id,
                recipient_id="campaign_matching_engine",
                action="evaluate_legitimacy",
                payload={"campaign_data": campaign_data},
            )
            # Handle both AgentMessage and dict responses
            if hasattr(response, "payload"):
                return response.payload
            elif isinstance(response, dict):
                return response
            return {"is_legitimate": True, "legitimacy_score": 0.5}
        except Exception as e:
            self._logger.warning("matcher_request_failed", error=str(e))
            return {"is_legitimate": True, "legitimacy_score": 0.5}
    
    async def ask_matcher_for_analysis(self, campaign_data: Dict[str, Any], donor_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ask Campaign Matching Engine to analyze a campaign."""
        try:
            response = await self.protocol.send_message(
                sender_id=self.agent_id,
                recipient_id="campaign_matching_engine",
                action="analyze_campaign",
                payload={
                    "campaign_data": campaign_data,
                    "donor_profile": donor_profile,
                },
            )
            # Handle both AgentMessage and dict responses
            if hasattr(response, "payload"):
                return response.payload
            elif isinstance(response, dict):
                return response
            return {}
        except Exception as e:
            self._logger.warning("matcher_analysis_request_failed", error=str(e))
            return {}
    
    async def ask_matcher_for_match(self, campaign_data: Dict[str, Any], donor_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Ask Campaign Matching Engine to match a campaign to a donor."""
        try:
            response = await self.protocol.send_message(
                sender_id=self.agent_id,
                recipient_id="campaign_matching_engine",
                action="match_campaign",
                payload={
                    "campaign_data": campaign_data,
                    "donor_profile": donor_profile,
                },
            )
            # Handle both AgentMessage and dict responses
            if hasattr(response, "payload"):
                return response.payload
            elif isinstance(response, dict):
                return response
            return {"match_score": 0.5, "reasons": []}
        except Exception as e:
            self._logger.warning("matcher_match_request_failed", error=str(e))
            return {"match_score": 0.5, "reasons": []}
    
    async def ask_community_for_connections(self, donor_id: str, location: str) -> Dict[str, Any]:
        """Ask Community Discovery Agent for community connections."""
        try:
            response = await self.protocol.send_message(
                sender_id=self.agent_id,
                recipient_id="community_discovery",
                action="discover_communities",
                payload={
                    "donor_id": donor_id,
                    "location": location,
                },
            )
            # Handle both AgentMessage and dict responses
            if hasattr(response, "payload"):
                return response.payload
            elif isinstance(response, dict):
                return response
            return {}
        except Exception as e:
            self._logger.warning("community_request_failed", error=str(e))
            return {}
    
    async def ask_engagement_for_plan(self, donor_id: str, donor_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Ask Engagement Agent for engagement plan."""
        try:
            response = await self.protocol.send_message(
                sender_id=self.agent_id,
                recipient_id="engagement_agent",
                action="create_engagement_plan",
                payload={
                    "donor_id": donor_id,
                    "donor_profile": donor_profile,
                },
            )
            # Handle both AgentMessage and dict responses
            if hasattr(response, "payload"):
                return response.payload
            elif isinstance(response, dict):
                return response
            return {}
        except Exception as e:
            self._logger.warning("engagement_request_failed", error=str(e))
            return {}


def get_collaborator(agent_id: str) -> AgentCollaborator:
    """Get an AgentCollaborator instance for an agent."""
    return AgentCollaborator(agent_id)

