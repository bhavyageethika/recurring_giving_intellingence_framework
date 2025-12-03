"""
Specialized autonomous agents for the Giving Intelligence platform.
Each agent uses LLM-powered planning, reasoning, and tool execution.
"""

from src.agents.donor_affinity_profiler import DonorAffinityProfiler
from src.agents.campaign_matching_engine import CampaignMatchingEngine
from src.agents.community_discovery import CommunityDiscoveryAgent
from src.agents.recurring_curator import RecurringCuratorAgent
from src.agents.giving_circle_orchestrator import GivingCircleOrchestrator
from src.agents.engagement_agent import EngagementAgent
from src.agents.campaign_data_agent import CampaignDataAgent

# Lazy imports for chat orchestrator
def get_chat_orchestrator():
    from src.agents.chat_orchestrator import ChatOrchestrator
    return ChatOrchestrator()

__all__ = [
    "DonorAffinityProfiler",
    "CampaignMatchingEngine",
    "CommunityDiscoveryAgent",
    "RecurringCuratorAgent",
    "GivingCircleOrchestrator",
    "EngagementAgent",
    "CampaignDataAgent",
]

# Lazy import for CampaignIntelligenceAgent to avoid circular imports
def get_campaign_intelligence_agent():
    """Lazy import for CampaignIntelligenceAgent."""
    from src.agents.campaign_intelligence_agent import CampaignIntelligenceAgent
    return CampaignIntelligenceAgent
