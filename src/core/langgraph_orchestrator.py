"""
LangGraph Orchestrator for Multi-Agent Workflows

Creates state machines that orchestrate agent interactions using LangGraph.
Enables conditional routing, parallel execution, and agent-to-agent communication.
"""

from typing import Annotated, TypedDict, Literal, Optional
from datetime import datetime
import asyncio

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

import structlog

from src.core.a2a_protocol import A2AProtocol, get_protocol
from src.core.base_agent import BaseAgent, AgentMessage

# Lazy imports to avoid circular dependency
# Agents are imported inside methods/__init__ where needed

logger = structlog.get_logger()


class WorkflowState(TypedDict):
    """State for the LangGraph workflow."""
    
    # Input data
    donor_id: str
    donations: list
    donor_metadata: dict
    campaigns: list
    
    # Agent outputs (using Annotated to allow concurrent updates)
    donor_profile: Annotated[dict, lambda x, y: {**x, **y}]
    campaign_analyses: Annotated[list, lambda x, y: x + y if isinstance(y, list) else x + [y]]
    community_insights: Annotated[dict, lambda x, y: {**x, **y}]
    recurring_opportunities: Annotated[list, lambda x, y: x + y if isinstance(y, list) else x + [y]]
    giving_circle_suggestions: Annotated[list, lambda x, y: x + y if isinstance(y, list) else x + [y]]
    engagement_plan: Annotated[dict, lambda x, y: {**x, **y}]
    
    # Workflow control
    current_step: str
    completed_steps: Annotated[list, lambda x, y: x + y if isinstance(y, list) else x + [y]]
    errors: Annotated[list, lambda x, y: x + y if isinstance(y, list) else x + [y]]
    
    # A2A messages
    agent_messages: Annotated[list, add_messages]
    
    # Metadata
    workflow_id: str
    started_at: str
    completed_at: str


class LangGraphOrchestrator:
    """
    Orchestrates multi-agent workflows using LangGraph.
    
    Creates state machines that:
    - Route tasks to appropriate agents
    - Handle conditional branching
    - Enable parallel execution
    - Facilitate agent-to-agent communication
    """
    
    def __init__(self):
        self.protocol: A2AProtocol = get_protocol()
        self.graph = None
        self.checkpointer = MemorySaver()
        self._logger = logger.bind(component="langgraph_orchestrator")
        
        # Lazy import to avoid circular dependency
        from src.agents.donor_affinity_profiler import DonorAffinityProfiler
        from src.agents.campaign_matching_engine import CampaignMatchingEngine
        from src.agents.community_discovery import CommunityDiscoveryAgent
        from src.agents.recurring_curator import RecurringCuratorAgent
        from src.agents.giving_circle_orchestrator import GivingCircleOrchestrator
        from src.agents.engagement_agent import EngagementAgent
        
        # Initialize agents
        self.profiler = DonorAffinityProfiler()
        self.matcher = CampaignMatchingEngine()
        self.community = CommunityDiscoveryAgent()
        self.curator = RecurringCuratorAgent()
        self.giving_circle = GivingCircleOrchestrator()
        self.engagement = EngagementAgent()
        
        # Register agents with A2A protocol
        self._register_agents()
        
        # Build workflow graph
        self._build_graph()
    
    def _register_agents(self):
        """Register all agents with the A2A protocol."""
        agents = [
            self.profiler,
            self.matcher,
            self.community,
            self.curator,
            self.giving_circle,
            self.engagement,
        ]
        
        for agent in agents:
            self.protocol.register_agent(agent)
            self._logger.info("agent_registered_for_orchestration", agent_id=agent.agent_id)
    
    def _build_graph(self):
        """Build the LangGraph workflow state machine."""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes (agent execution steps)
        workflow.add_node("profile_donor", self._node_profile_donor)
        workflow.add_node("analyze_campaigns", self._node_analyze_campaigns)
        workflow.add_node("discover_community", self._node_discover_community)
        workflow.add_node("curate_opportunities", self._node_curate_opportunities)
        workflow.add_node("suggest_giving_circles", self._node_suggest_giving_circles)
        workflow.add_node("plan_engagement", self._node_plan_engagement)
        
        # Define workflow edges
        workflow.set_entry_point("profile_donor")
        
        # Sequential flow with conditional routing
        workflow.add_edge("profile_donor", "analyze_campaigns")
        workflow.add_conditional_edges(
            "analyze_campaigns",
            self._should_continue_after_analysis,
            {
                "continue": "discover_community",
                "skip": "curate_opportunities",
            }
        )
        workflow.add_edge("discover_community", "curate_opportunities")
        
        # Parallel paths after curation
        workflow.add_edge("curate_opportunities", "suggest_giving_circles")
        workflow.add_edge("curate_opportunities", "plan_engagement")
        
        # Both paths converge to END
        workflow.add_edge("suggest_giving_circles", END)
        workflow.add_edge("plan_engagement", END)
        
        # Compile graph with checkpointer
        self.graph = workflow.compile(checkpointer=self.checkpointer)
        self._logger.info("workflow_graph_built")
    
    async def _node_profile_donor(self, state: WorkflowState) -> WorkflowState:
        """Node: Profile the donor."""
        self._logger.info("executing_node", node="profile_donor", donor_id=state.get("donor_id"))
        
        try:
            profile = await self.profiler.build_profile(
                donor_id=state["donor_id"],
                donations=state.get("donations", []),
                metadata=state.get("donor_metadata", {}),
            )
            
            profile_dict = profile.to_dict() if hasattr(profile, "to_dict") else profile
            return {
                **state,
                "donor_profile": profile_dict,
                "completed_steps": state.get("completed_steps", []) + ["profile_donor"],
            }
        except Exception as e:
            self._logger.error("node_failed", node="profile_donor", error=str(e))
            return {
                **state,
                "errors": state.get("errors", []) + [{"step": "profile_donor", "error": str(e)}],
            }
    
    async def _node_analyze_campaigns(self, state: WorkflowState) -> WorkflowState:
        """Node: Analyze campaigns using Campaign Matching Engine."""
        self._logger.info("executing_node", node="analyze_campaigns")
        
        try:
            campaigns = state.get("campaigns", [])
            donor_profile = state.get("donor_profile", {})
            
            analyses = []
            for campaign in campaigns:
                # Use A2A: Curator can ask Matcher for analysis
                analysis = await self.matcher.analyze_campaign(
                    campaign_data=campaign,
                )
                analyses.append(analysis.to_dict() if hasattr(analysis, "to_dict") else analysis)
            
            return {
                **state,
                "campaign_analyses": analyses,
                "completed_steps": state.get("completed_steps", []) + ["analyze_campaigns"],
            }
        except Exception as e:
            self._logger.error("node_failed", node="analyze_campaigns", error=str(e))
            return {
                **state,
                "errors": state.get("errors", []) + [{"step": "analyze_campaigns", "error": str(e)}],
            }
    
    async def _node_discover_community(self, state: WorkflowState) -> WorkflowState:
        """Node: Discover local community opportunities (with A2A communication)."""
        self._logger.info("executing_node", node="discover_community")
        
        try:
            donor_profile = state.get("donor_profile", {})
            donor_metadata = state.get("donor_metadata", {})
            donor_id = state.get("donor_id", "")
            
            # AGENT COLLABORATION: Community Discovery asks Profiler for donor insights
            try:
                profiler_insights = await self.protocol.send_message(
                    sender_id=self.community.agent_id,
                    recipient_id=self.profiler.agent_id,
                    action="get_donor_insights",
                    payload={"donor_id": donor_id},
                )
                
                # Enhance donor profile with insights from Profiler
                if profiler_insights.payload.get("insights"):
                    donor_profile = {**donor_profile, **profiler_insights.payload["insights"]}
                
                self._logger.info(
                    "a2a_communication",
                    sender=self.community.agent_id,
                    recipient=self.profiler.agent_id,
                    action="get_donor_insights",
                )
            except Exception as e:
                self._logger.warning("a2a_communication_failed", error=str(e))
            
            insights = await self.community.discover_communities(
                donor_profile=donor_profile,
                location=donor_metadata.get("location", ""),
            )
            
            insights_dict = insights.to_dict() if hasattr(insights, "to_dict") else insights
            return {
                **state,
                "community_insights": insights_dict,
                "completed_steps": state.get("completed_steps", []) + ["discover_community"],
            }
        except Exception as e:
            self._logger.error("node_failed", node="discover_community", error=str(e))
            return {
                **state,
                "errors": state.get("errors", []) + [{"step": "discover_community", "error": str(e)}],
            }
    
    async def _node_curate_opportunities(self, state: WorkflowState) -> WorkflowState:
        """Node: Curate recurring opportunities (with A2A communication)."""
        self._logger.info("executing_node", node="curate_opportunities")
        
        try:
            donor_profile = state.get("donor_profile", {})
            campaigns = state.get("campaigns", [])
            campaign_analyses = state.get("campaign_analyses", [])
            
            # Example A2A: Curator asks Matcher for legitimacy check
            # This demonstrates agent-to-agent communication
            validated_campaigns = []
            for campaign in campaigns:
                # Use A2A protocol to ask Matcher for legitimacy assessment
                try:
                    message = AgentMessage(
                        sender=self.curator.agent_id,
                        recipient=self.matcher.agent_id,
                        message_type="request",
                        action="evaluate_legitimacy",
                        payload={"campaign_data": campaign},
                    )
                    
                    response = await self.protocol.send_message(
                        sender_id=self.curator.agent_id,
                        recipient_id=self.matcher.agent_id,
                        action="evaluate_legitimacy",
                        payload={"campaign_data": campaign},
                    )
                    
                    if response.payload.get("is_legitimate", True):
                        validated_campaigns.append(campaign)
                    
                    self._logger.info(
                        "a2a_communication",
                        sender=self.curator.agent_id,
                        recipient=self.matcher.agent_id,
                        action="evaluate_legitimacy",
                    )
                except Exception as e:
                    self._logger.warning("a2a_communication_failed", error=str(e))
                    # Fallback: include campaign anyway
                    validated_campaigns.append(campaign)
            
            # Curate opportunities (agent uses A2A internally to ask Matcher for legitimacy)
            opportunities = await self.curator.curate_opportunities(
                campaigns=validated_campaigns,
            )
            
            opportunities_list = [
                opp.to_dict() if hasattr(opp, "to_dict") else opp
                for opp in opportunities
            ]
            return {
                **state,
                "recurring_opportunities": opportunities_list,
                "completed_steps": state.get("completed_steps", []) + ["curate_opportunities"],
            }
        except Exception as e:
            self._logger.error("node_failed", node="curate_opportunities", error=str(e))
            return {
                **state,
                "errors": state.get("errors", []) + [{"step": "curate_opportunities", "error": str(e)}],
            }
    
    async def _node_suggest_giving_circles(self, state: WorkflowState) -> WorkflowState:
        """Node: Suggest giving circles (with A2A communication)."""
        self._logger.info("executing_node", node="suggest_giving_circles")
        
        try:
            donor_profile = state.get("donor_profile", {})
            community_insights = state.get("community_insights", {})
            donor_id = state.get("donor_id", "")
            donor_metadata = state.get("donor_metadata", {})
            
            # AGENT COLLABORATION: Giving Circle asks Community Discovery for connections
            try:
                community_result = await self.protocol.send_message(
                    sender_id=self.giving_circle.agent_id,
                    recipient_id=self.community.agent_id,
                    action="discover_communities",
                    payload={
                        "donor_id": donor_id,
                        "location": donor_metadata.get("location", ""),
                        "donor_profile": donor_profile,
                    },
                )
                
                # Enhance community insights with results from Community Discovery
                if community_result.payload:
                    community_insights = {**community_insights, **community_result.payload}
                
                self._logger.info(
                    "a2a_communication",
                    sender=self.giving_circle.agent_id,
                    recipient=self.community.agent_id,
                    action="discover_communities",
                )
            except Exception as e:
                self._logger.warning("a2a_communication_failed", error=str(e))
            
            suggestions = await self.giving_circle.orchestrate_circle(
                donor_profile=donor_profile,
                community_context=community_insights,
            )
            
            suggestions_list = [
                sug.to_dict() if hasattr(sug, "to_dict") else sug
                for sug in suggestions
            ]
            return {
                **state,
                "giving_circle_suggestions": suggestions_list,
                "completed_steps": state.get("completed_steps", []) + ["suggest_giving_circles"],
            }
        except Exception as e:
            self._logger.error("node_failed", node="suggest_giving_circles", error=str(e))
            return {
                **state,
                "errors": state.get("errors", []) + [{"step": "suggest_giving_circles", "error": str(e)}],
            }
    
    async def _node_plan_engagement(self, state: WorkflowState) -> WorkflowState:
        """Node: Plan engagement (with A2A communication)."""
        self._logger.info("executing_node", node="plan_engagement")
        
        try:
            donor_profile = state.get("donor_profile", {})
            
            # Example A2A: Engagement Agent asks Profiler for donor insights
            try:
                message = AgentMessage(
                    sender=self.engagement.agent_id,
                    recipient=self.profiler.agent_id,
                    message_type="request",
                    action="get_donor_insights",
                    payload={"donor_id": state["donor_id"]},
                )
                
                response = await self.protocol.send_message(
                    sender_id=self.engagement.agent_id,
                    recipient_id=self.profiler.agent_id,
                    action="get_donor_insights",
                    payload={"donor_id": state["donor_id"]},
                )
                
                # Use insights from Profiler
                donor_insights = response.payload.get("insights", {})
                
                self._logger.info(
                    "a2a_communication",
                    sender=self.engagement.agent_id,
                    recipient=self.profiler.agent_id,
                    action="get_donor_insights",
                )
            except Exception as e:
                self._logger.warning("a2a_communication_failed", error=str(e))
                donor_insights = {}
            
            plan = await self.engagement.create_engagement_plan(
                donor_profile=donor_profile,
                additional_context=donor_insights,
            )
            
            plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else plan
            return {
                **state,
                "engagement_plan": plan_dict,
                "completed_steps": state.get("completed_steps", []) + ["plan_engagement"],
            }
        except Exception as e:
            self._logger.error("node_failed", node="plan_engagement", error=str(e))
            return {
                **state,
                "errors": state.get("errors", []) + [{"step": "plan_engagement", "error": str(e)}],
            }
    
    def _should_continue_after_analysis(self, state: WorkflowState) -> Literal["continue", "skip"]:
        """Conditional routing: Continue to community discovery if we have location data."""
        donor_metadata = state.get("donor_metadata", {})
        location = donor_metadata.get("location", "")
        
        if location:
            return "continue"
        return "skip"
    
    async def run_workflow(
        self,
        donor_id: str,
        donations: list,
        campaigns: list,
        donor_metadata: dict = None,
        config: dict = None,
    ) -> WorkflowState:
        """
        Run the complete donor journey workflow.
        
        Args:
            donor_id: Unique donor identifier
            donations: List of donation records
            campaigns: List of campaign data
            donor_metadata: Optional donor metadata
            config: Optional LangGraph config
            
        Returns:
            Final workflow state
        """
        initial_state: WorkflowState = {
            "donor_id": donor_id,
            "donations": donations,
            "campaigns": campaigns,
            "donor_metadata": donor_metadata or {},
            "donor_profile": {},
            "campaign_analyses": [],
            "community_insights": {},
            "recurring_opportunities": [],
            "giving_circle_suggestions": [],
            "engagement_plan": {},
            "current_step": "start",
            "completed_steps": [],
            "errors": [],
            "agent_messages": [],
            "workflow_id": f"workflow_{datetime.utcnow().isoformat()}",
            "started_at": datetime.utcnow().isoformat(),
            "completed_at": "",
        }
        
        self._logger.info("starting_workflow", workflow_id=initial_state["workflow_id"], donor_id=donor_id)
        
        config = config or {"configurable": {"thread_id": initial_state["workflow_id"]}}
        
        try:
            # Run the graph
            final_state = None
            async for state in self.graph.astream(initial_state, config=config):
                final_state = state
                self._logger.debug("workflow_step", state_keys=list(state.keys()))
            
            if final_state:
                final_state["completed_at"] = datetime.utcnow().isoformat()
                self._logger.info(
                    "workflow_completed",
                    workflow_id=initial_state["workflow_id"],
                    completed_steps=final_state.get("completed_steps", []),
                )
                return final_state
            
            return initial_state
            
        except Exception as e:
            self._logger.error("workflow_failed", error=str(e))
            initial_state["errors"] = [{"step": "workflow", "error": str(e)}]
            return initial_state
    
    def visualize_workflow(self) -> str:
        """Generate a visual representation of the workflow graph."""
        if self.graph:
            return self.graph.get_graph().draw_mermaid()
        return "Graph not built"


# Singleton instance
_orchestrator: Optional[LangGraphOrchestrator] = None


def get_orchestrator() -> LangGraphOrchestrator:
    """Get the singleton orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = LangGraphOrchestrator()
    return _orchestrator

