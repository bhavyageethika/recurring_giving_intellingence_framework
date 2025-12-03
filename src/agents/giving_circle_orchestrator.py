"""
Agent 5: Giving Circle Orchestrator (Autonomous Agent)

An LLM-based autonomous agent that facilitates collective giving.
Uses planning, reasoning, and tool-use to:
- Form giving circles based on shared interests
- Facilitate group decision-making
- Orchestrate collective donations
- Build community around shared giving
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict
import json
import re
import math

import structlog

from src.core.autonomous_agent import AutonomousAgent, Tool, Task, ReasoningStep, ReasoningType
from src.core.llm_service import get_llm_service
from src.core.agent_collaboration import get_collaborator

logger = structlog.get_logger()


class CircleType(str, Enum):
    """Types of giving circles."""
    INTEREST_BASED = "interest_based"
    WORKPLACE = "workplace"
    ALUMNI = "alumni"
    NEIGHBORHOOD = "neighborhood"
    FAMILY = "family"
    FAITH_BASED = "faith_based"
    CUSTOM = "custom"


class MemberRole(str, Enum):
    """Roles within a giving circle."""
    FOUNDER = "founder"
    ADMIN = "admin"
    MEMBER = "member"
    INVITED = "invited"


class CircleStatus(str, Enum):
    """Status of a giving circle."""
    FORMING = "forming"
    ACTIVE = "active"
    VOTING = "voting"
    DISBURSING = "disbursing"
    PAUSED = "paused"
    COMPLETED = "completed"


class VoteType(str, Enum):
    """Types of votes in a giving circle."""
    CAMPAIGN_SELECTION = "campaign_selection"
    AMOUNT_ALLOCATION = "amount_allocation"
    NEW_MEMBER = "new_member"
    RULE_CHANGE = "rule_change"


@dataclass
class CircleMember:
    """A member of a giving circle."""
    user_id: str
    display_name: str
    role: MemberRole = MemberRole.MEMBER
    joined_at: datetime = field(default_factory=datetime.utcnow)
    contribution_total: float = 0.0
    votes_cast: int = 0
    is_active: bool = True


@dataclass
class CircleVote:
    """A vote within a giving circle."""
    vote_id: str
    circle_id: str
    vote_type: VoteType
    title: str
    description: str
    options: List[Dict[str, Any]] = field(default_factory=list)
    votes: Dict[str, str] = field(default_factory=dict)  # user_id -> option_id
    created_at: datetime = field(default_factory=datetime.utcnow)
    closes_at: Optional[datetime] = None
    is_closed: bool = False
    result: Optional[str] = None


@dataclass
class GivingCircle:
    """A collective giving circle."""
    circle_id: str
    name: str
    description: str
    circle_type: CircleType = CircleType.INTEREST_BASED
    status: CircleStatus = CircleStatus.FORMING
    
    # Members
    members: List[CircleMember] = field(default_factory=list)
    max_members: int = 20
    min_members: int = 3
    
    # Finances
    pool_balance: float = 0.0
    contribution_frequency: str = "monthly"
    contribution_amount: float = 25.0
    total_disbursed: float = 0.0
    
    # Focus
    cause_focus: List[str] = field(default_factory=list)
    geographic_focus: Optional[str] = None
    
    # Voting
    active_votes: List[CircleVote] = field(default_factory=list)
    voting_threshold: float = 0.5  # Majority
    
    # History
    campaigns_supported: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # LLM-generated
    mission_statement: str = ""
    impact_summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "circle_id": self.circle_id,
            "name": self.name,
            "description": self.description,
            "circle_type": self.circle_type.value,
            "status": self.status.value,
            "member_count": len(self.members),
            "pool_balance": self.pool_balance,
            "contribution_amount": self.contribution_amount,
            "total_disbursed": self.total_disbursed,
            "cause_focus": self.cause_focus,
            "campaigns_supported": len(self.campaigns_supported),
            "mission_statement": self.mission_statement,
            "impact_summary": self.impact_summary,
        }


class GivingCircleOrchestrator(AutonomousAgent[GivingCircle]):
    """
    Autonomous agent for orchestrating giving circles.
    
    This agent:
    1. Forms circles based on shared interests
    2. Facilitates democratic decision-making
    3. Manages pooled funds and disbursements
    4. Builds community through collective giving
    5. Generates impact reports and narratives
    """
    
    SYSTEM_PROMPT = """You are an expert in collective giving and community building.
Your expertise includes:
- Forming effective giving groups
- Facilitating democratic decision-making
- Managing group dynamics
- Building community around shared values
- Measuring and communicating collective impact

You understand that giving circles:
- Amplify individual impact through pooled resources
- Create social bonds through shared purpose
- Democratize philanthropy
- Build giving habits through community accountability

You help donors experience the joy and power of giving together."""

    def __init__(self):
        super().__init__(
            agent_id="giving_circle_orchestrator",
            name="Giving Circle Orchestrator",
            description="Autonomous agent for facilitating collective giving",
            system_prompt=self.SYSTEM_PROMPT,
        )
        
        # Data stores
        self._circles: Dict[str, GivingCircle] = {}
        self._user_circles: Dict[str, List[str]] = defaultdict(list)
        self._campaigns: Dict[str, Dict[str, Any]] = {}
        
        # Register domain-specific tools
        for tool in self._get_domain_tools():
            self.register_tool(tool)
        
        # Initialize collaborator for A2A communication
        self._collaborator = get_collaborator(self.agent_id)
    
    def _get_domain_system_prompt(self) -> str:
        return """
## Domain Expertise: Giving Circles

You specialize in:

1. **Circle Formation**: Creating effective giving groups
   - Matching members by interests and values
   - Setting appropriate group size
   - Establishing governance rules
   - Creating compelling missions

2. **Democratic Facilitation**: Managing group decisions
   - Campaign nomination and voting
   - Allocation decisions
   - Conflict resolution
   - Consensus building

3. **Community Building**: Fostering connection
   - Shared identity creation
   - Celebration of collective impact
   - Member engagement
   - Story sharing

4. **Impact Amplification**: Maximizing collective power
   - Strategic campaign selection
   - Coordinated giving timing
   - Matching fund opportunities
   - Public recognition
"""
    
    def _get_domain_tools(self) -> List[Tool]:
        return [
            Tool(
                name="form_circle",
                description="Form a new giving circle with initial members",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "circle_type": {"type": "string"},
                        "founder_id": {"type": "string"},
                        "cause_focus": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name", "founder_id"],
                },
                function=self._tool_form_circle,
            ),
            Tool(
                name="suggest_circle_matches",
                description="Suggest potential circles for a user based on their profile",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "user_profile": {"type": "object"},
                    },
                    "required": ["user_id", "user_profile"],
                },
                function=self._tool_suggest_matches,
            ),
            Tool(
                name="add_member",
                description="Add a member to a giving circle",
                parameters={
                    "type": "object",
                    "properties": {
                        "circle_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        "display_name": {"type": "string"},
                    },
                    "required": ["circle_id", "user_id", "display_name"],
                },
                function=self._tool_add_member,
            ),
            Tool(
                name="create_vote",
                description="Create a vote for the circle to decide on",
                parameters={
                    "type": "object",
                    "properties": {
                        "circle_id": {"type": "string"},
                        "vote_type": {"type": "string"},
                        "title": {"type": "string"},
                        "options": {"type": "array"},
                    },
                    "required": ["circle_id", "vote_type", "title", "options"],
                },
                function=self._tool_create_vote,
            ),
            Tool(
                name="cast_vote",
                description="Cast a vote in a circle decision",
                parameters={
                    "type": "object",
                    "properties": {
                        "circle_id": {"type": "string"},
                        "vote_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        "option_id": {"type": "string"},
                    },
                    "required": ["circle_id", "vote_id", "user_id", "option_id"],
                },
                function=self._tool_cast_vote,
            ),
            Tool(
                name="tally_votes",
                description="Tally votes and determine the result",
                parameters={
                    "type": "object",
                    "properties": {
                        "circle_id": {"type": "string"},
                        "vote_id": {"type": "string"},
                    },
                    "required": ["circle_id", "vote_id"],
                },
                function=self._tool_tally_votes,
            ),
            Tool(
                name="recommend_campaigns",
                description="Recommend campaigns for the circle to consider",
                parameters={
                    "type": "object",
                    "properties": {
                        "circle_id": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["circle_id"],
                },
                function=self._tool_recommend_campaigns,
            ),
            Tool(
                name="disburse_funds",
                description="Disburse pooled funds to a campaign",
                parameters={
                    "type": "object",
                    "properties": {
                        "circle_id": {"type": "string"},
                        "campaign_id": {"type": "string"},
                        "amount": {"type": "number"},
                    },
                    "required": ["circle_id", "campaign_id", "amount"],
                },
                function=self._tool_disburse,
            ),
            Tool(
                name="generate_impact_report",
                description="Generate an impact report for the circle",
                parameters={
                    "type": "object",
                    "properties": {
                        "circle_id": {"type": "string"},
                    },
                    "required": ["circle_id"],
                },
                function=self._tool_generate_impact,
            ),
            Tool(
                name="generate_mission_statement",
                description="Generate a mission statement for the circle",
                parameters={
                    "type": "object",
                    "properties": {
                        "circle_id": {"type": "string"},
                    },
                    "required": ["circle_id"],
                },
                function=self._tool_generate_mission,
            ),
        ]
    
    # ==================== Tool Implementations ====================
    
    async def _tool_form_circle(
        self,
        name: str,
        founder_id: str,
        description: str = "",
        circle_type: str = "interest_based",
        cause_focus: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Form a new giving circle."""
        circle_id = f"circle_{founder_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        circle = GivingCircle(
            circle_id=circle_id,
            name=name,
            description=description or f"A giving circle focused on making a difference together.",
            circle_type=CircleType(circle_type),
            cause_focus=cause_focus or [],
            members=[
                CircleMember(
                    user_id=founder_id,
                    display_name=f"Founder",
                    role=MemberRole.FOUNDER,
                )
            ],
        )
        
        self._circles[circle_id] = circle
        self._user_circles[founder_id].append(circle_id)
        
        # Generate mission statement
        await self._tool_generate_mission(circle_id)
        
        return {
            "status": "created",
            "circle_id": circle_id,
            "name": name,
            "founder_id": founder_id,
        }
    
    async def _tool_suggest_matches(
        self,
        user_id: str,
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Suggest circles that match a user's interests."""
        suggestions = []
        
        user_causes = [
            a.get("category") for a in user_profile.get("cause_affinities", [])
        ]
        user_location = user_profile.get("city", "")
        
        for circle_id, circle in self._circles.items():
            if circle.status != CircleStatus.ACTIVE:
                continue
            if len(circle.members) >= circle.max_members:
                continue
            if user_id in [m.user_id for m in circle.members]:
                continue
            
            score = 0
            reasons = []
            
            # Cause match
            common_causes = set(user_causes) & set(circle.cause_focus)
            if common_causes:
                score += len(common_causes) * 0.3
                reasons.append(f"Shared interest in {', '.join(common_causes)}")
            
            # Geographic match
            if circle.geographic_focus and user_location:
                if user_location.lower() in circle.geographic_focus.lower():
                    score += 0.3
                    reasons.append(f"Local to {circle.geographic_focus}")
            
            # Size preference (smaller circles = more intimate)
            if len(circle.members) < 10:
                score += 0.1
                reasons.append("Intimate group size")
            
            if score > 0:
                suggestions.append({
                    "circle_id": circle_id,
                    "name": circle.name,
                    "description": circle.description,
                    "member_count": len(circle.members),
                    "match_score": round(score, 3),
                    "reasons": reasons,
                })
        
        # Sort by score
        suggestions.sort(key=lambda x: x["match_score"], reverse=True)
        
        return {
            "user_id": user_id,
            "suggestions": suggestions[:5],
        }
    
    async def _tool_add_member(
        self,
        circle_id: str,
        user_id: str,
        display_name: str
    ) -> Dict[str, Any]:
        """Add a member to a circle."""
        circle = self._circles.get(circle_id)
        if not circle:
            return {"error": "Circle not found"}
        
        if len(circle.members) >= circle.max_members:
            return {"error": "Circle is full"}
        
        if any(m.user_id == user_id for m in circle.members):
            return {"error": "User already a member"}
        
        member = CircleMember(
            user_id=user_id,
            display_name=display_name,
            role=MemberRole.MEMBER,
        )
        
        circle.members.append(member)
        self._user_circles[user_id].append(circle_id)
        
        # Activate circle if minimum members reached
        if len(circle.members) >= circle.min_members and circle.status == CircleStatus.FORMING:
            circle.status = CircleStatus.ACTIVE
        
        return {
            "status": "added",
            "circle_id": circle_id,
            "user_id": user_id,
            "member_count": len(circle.members),
            "circle_status": circle.status.value,
        }
    
    async def _tool_create_vote(
        self,
        circle_id: str,
        vote_type: str,
        title: str,
        options: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create a vote in the circle."""
        circle = self._circles.get(circle_id)
        if not circle:
            return {"error": "Circle not found"}
        
        vote_id = f"vote_{circle_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        vote = CircleVote(
            vote_id=vote_id,
            circle_id=circle_id,
            vote_type=VoteType(vote_type),
            title=title,
            description=f"Vote on: {title}",
            options=options,
            closes_at=datetime.utcnow() + timedelta(days=3),
        )
        
        circle.active_votes.append(vote)
        circle.status = CircleStatus.VOTING
        
        return {
            "status": "created",
            "vote_id": vote_id,
            "title": title,
            "options": options,
            "closes_at": vote.closes_at.isoformat(),
        }
    
    async def _tool_cast_vote(
        self,
        circle_id: str,
        vote_id: str,
        user_id: str,
        option_id: str
    ) -> Dict[str, Any]:
        """Cast a vote."""
        circle = self._circles.get(circle_id)
        if not circle:
            return {"error": "Circle not found"}
        
        vote = next((v for v in circle.active_votes if v.vote_id == vote_id), None)
        if not vote:
            return {"error": "Vote not found"}
        
        if vote.is_closed:
            return {"error": "Vote is closed"}
        
        if not any(m.user_id == user_id for m in circle.members):
            return {"error": "User not a member"}
        
        vote.votes[user_id] = option_id
        
        # Update member stats
        member = next(m for m in circle.members if m.user_id == user_id)
        member.votes_cast += 1
        
        return {
            "status": "recorded",
            "vote_id": vote_id,
            "user_id": user_id,
            "votes_cast": len(vote.votes),
            "total_members": len(circle.members),
        }
    
    async def _tool_tally_votes(
        self,
        circle_id: str,
        vote_id: str
    ) -> Dict[str, Any]:
        """Tally votes and determine result."""
        circle = self._circles.get(circle_id)
        if not circle:
            return {"error": "Circle not found"}
        
        vote = next((v for v in circle.active_votes if v.vote_id == vote_id), None)
        if not vote:
            return {"error": "Vote not found"}
        
        # Count votes
        vote_counts = defaultdict(int)
        for option_id in vote.votes.values():
            vote_counts[option_id] += 1
        
        # Determine winner
        total_votes = len(vote.votes)
        total_members = len(circle.members)
        participation = total_votes / total_members if total_members > 0 else 0
        
        if vote_counts:
            winner_id = max(vote_counts, key=vote_counts.get)
            winner_count = vote_counts[winner_id]
            winner_pct = winner_count / total_votes if total_votes > 0 else 0
            
            # Check if threshold met
            if winner_pct >= circle.voting_threshold:
                vote.result = winner_id
                vote.is_closed = True
                
                # Update circle status
                circle.status = CircleStatus.ACTIVE
                
                return {
                    "status": "decided",
                    "winner": winner_id,
                    "vote_count": winner_count,
                    "percentage": round(winner_pct * 100, 1),
                    "participation": round(participation * 100, 1),
                }
        
        return {
            "status": "no_decision",
            "vote_counts": dict(vote_counts),
            "participation": round(participation * 100, 1),
            "threshold_needed": circle.voting_threshold * 100,
        }
    
    async def _tool_recommend_campaigns(
        self,
        circle_id: str,
        limit: int = 5
    ) -> Dict[str, Any]:
        """Recommend campaigns for the circle to consider."""
        circle = self._circles.get(circle_id)
        if not circle:
            return {"error": "Circle not found"}
        
        recommendations = []
        
        for campaign_id, campaign in self._campaigns.items():
            if campaign_id in circle.campaigns_supported:
                continue
            
            score = 0
            reasons = []
            
            # Cause match
            campaign_category = campaign.get("category", "")
            if campaign_category in circle.cause_focus:
                score += 0.5
                reasons.append(f"Matches circle focus: {campaign_category}")
            
            # Geographic match
            if circle.geographic_focus:
                campaign_location = f"{campaign.get('city', '')}, {campaign.get('state', '')}"
                if circle.geographic_focus.lower() in campaign_location.lower():
                    score += 0.3
                    reasons.append("Local campaign")
            
            # Funding gap (prioritize campaigns that need help)
            goal = campaign.get("goal_amount", 1)
            raised = campaign.get("raised_amount", 0)
            funding_pct = raised / goal if goal > 0 else 1
            if 0.3 <= funding_pct <= 0.8:
                score += 0.2
                reasons.append(f"{funding_pct*100:.0f}% funded - your support matters")
            
            if score > 0:
                recommendations.append({
                    "campaign_id": campaign_id,
                    "title": campaign.get("title", ""),
                    "category": campaign_category,
                    "goal": campaign.get("goal_amount", 0),
                    "raised": campaign.get("raised_amount", 0),
                    "match_score": round(score, 3),
                    "reasons": reasons,
                })
        
        # Sort by score
        recommendations.sort(key=lambda x: x["match_score"], reverse=True)
        
        # Use LLM to add context
        if recommendations:
            prompt = f"""Write a brief recommendation (1-2 sentences) for why the circle "{circle.name}" should consider supporting the top campaign:

Campaign: {recommendations[0]['title']}
Reasons: {recommendations[0]['reasons']}
Circle Focus: {circle.cause_focus}

Be enthusiastic but not pushy."""

            recommendation_text = await self._llm._provider.complete(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.7,
            )
            recommendations[0]["recommendation"] = recommendation_text.strip()
        
        return {
            "circle_id": circle_id,
            "recommendations": recommendations[:limit],
        }
    
    async def _tool_disburse(
        self,
        circle_id: str,
        campaign_id: str,
        amount: float
    ) -> Dict[str, Any]:
        """Disburse funds to a campaign."""
        circle = self._circles.get(circle_id)
        if not circle:
            return {"error": "Circle not found"}
        
        if amount > circle.pool_balance:
            return {"error": "Insufficient funds", "available": circle.pool_balance}
        
        circle.pool_balance -= amount
        circle.total_disbursed += amount
        circle.campaigns_supported.append(campaign_id)
        circle.status = CircleStatus.ACTIVE
        
        campaign = self._campaigns.get(campaign_id, {})
        
        return {
            "status": "disbursed",
            "circle_id": circle_id,
            "campaign_id": campaign_id,
            "campaign_title": campaign.get("title", ""),
            "amount": amount,
            "remaining_balance": circle.pool_balance,
            "total_disbursed": circle.total_disbursed,
        }
    
    async def _tool_generate_impact(
        self,
        circle_id: str
    ) -> Dict[str, Any]:
        """Generate an impact report for the circle."""
        circle = self._circles.get(circle_id)
        if not circle:
            return {"error": "Circle not found"}
        
        # Gather stats
        stats = {
            "member_count": len(circle.members),
            "total_disbursed": circle.total_disbursed,
            "campaigns_supported": len(circle.campaigns_supported),
            "pool_balance": circle.pool_balance,
        }
        
        prompt = f"""Generate an inspiring impact report for this giving circle:

Circle: {circle.name}
Mission: {circle.mission_statement}
Members: {stats['member_count']}
Total Given: ${stats['total_disbursed']:,.0f}
Campaigns Supported: {stats['campaigns_supported']}

Create a 3-4 sentence impact summary that:
1. Celebrates the collective achievement
2. Highlights the power of giving together
3. Inspires continued participation"""

        impact_summary = await self._llm._provider.complete(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
        )
        
        circle.impact_summary = impact_summary.strip()
        
        return {
            "circle_id": circle_id,
            "stats": stats,
            "impact_summary": circle.impact_summary,
        }
    
    async def _tool_generate_mission(
        self,
        circle_id: str
    ) -> Dict[str, Any]:
        """Generate a mission statement for the circle."""
        circle = self._circles.get(circle_id)
        if not circle:
            return {"error": "Circle not found"}
        
        prompt = f"""Create a compelling mission statement for this giving circle:

Name: {circle.name}
Type: {circle.circle_type.value}
Focus Areas: {circle.cause_focus}
Description: {circle.description}

Write a 1-2 sentence mission statement that:
1. Captures the circle's purpose
2. Inspires collective action
3. Reflects shared values

Be aspirational but authentic."""

        mission = await self._llm._provider.complete(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
        )
        
        circle.mission_statement = mission.strip()
        
        return {
            "circle_id": circle_id,
            "mission_statement": circle.mission_statement,
        }
    
    # ==================== Public API ====================
    
    def register_campaign(self, campaign: Dict[str, Any]) -> None:
        """Register a campaign for circle consideration."""
        campaign_id = campaign.get("campaign_id", str(hash(campaign.get("title", ""))))
        self._campaigns[campaign_id] = campaign
    
    async def create_circle(
        self,
        name: str,
        founder_id: str,
        description: str = "",
        circle_type: str = "interest_based",
        cause_focus: Optional[List[str]] = None
    ) -> GivingCircle:
        """
        Create a new giving circle autonomously.
        
        The agent will:
        1. Form the circle with initial settings
        2. Generate a mission statement
        3. Recommend initial campaigns
        """
        goal = f"Create and initialize giving circle '{name}'"
        
        context = {
            "name": name,
            "founder_id": founder_id,
            "description": description,
            "circle_type": circle_type,
            "cause_focus": cause_focus,
        }
        
        # Run autonomous creation
        await self.run(goal, context)
        
        # Return the created circle
        for circle in self._circles.values():
            if circle.name == name:
                return circle
        
        return GivingCircle(
            circle_id="unknown",
            name=name,
            description=description,
        )
    
    async def find_circles_for_user(
        self,
        user_id: str,
        user_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find circles that match a user's interests."""
        result = await self._tool_suggest_matches(user_id, user_profile)
        return result.get("suggestions", [])
    
    async def join_circle(
        self,
        circle_id: str,
        user_id: str,
        display_name: str
    ) -> Dict[str, Any]:
        """Join a giving circle."""
        return await self._tool_add_member(circle_id, user_id, display_name)
    
    async def start_campaign_vote(
        self,
        circle_id: str,
        campaign_options: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Start a vote to select a campaign to support."""
        options = [
            {"id": c.get("campaign_id"), "label": c.get("title")}
            for c in campaign_options
        ]
        
        return await self._tool_create_vote(
            circle_id=circle_id,
            vote_type="campaign_selection",
            title="Which campaign should we support next?",
            options=options,
        )
    
    async def get_circle(self, circle_id: str) -> Optional[GivingCircle]:
        """Get a circle by ID."""
        return self._circles.get(circle_id)
    
    async def get_user_circles(self, user_id: str) -> List[GivingCircle]:
        """Get all circles a user belongs to."""
        circle_ids = self._user_circles.get(user_id, [])
        return [self._circles[cid] for cid in circle_ids if cid in self._circles]
    
    async def orchestrate_circle(
        self,
        donor_profile: Dict[str, Any],
        community_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Orchestrate giving circle suggestions for a donor.
        
        Uses agent collaboration to:
        - Get community connections from Community Discovery
        - Create personalized circle suggestions
        """
        donor_id = donor_profile.get("donor_id", "unknown")
        
        # AGENT COLLABORATION: Ask Community Discovery for connections
        community_insights = community_context or {}
        if not community_insights.get("proximities"):
            try:
                location = donor_profile.get("location", "")
                community_result = await self._collaborator.ask_community_for_connections(
                    donor_id=donor_id,
                    location=location,
                )
                if community_result:
                    community_insights = community_result
            except Exception as e:
                self._logger.warning("community_collaboration_failed", error=str(e))
        
        # Create circle suggestions based on donor profile and community
        suggestions = await self.find_circles_for_user(
            user_id=donor_id,
            user_profile=donor_profile,
        )
        
        # Enhance suggestions with community context
        for suggestion in suggestions:
            if community_insights:
                suggestion["community_context"] = community_insights
        
        return suggestions
