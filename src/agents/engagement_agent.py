"""
Agent 6: Engagement & Re-activation Agent (Autonomous Agent)

An LLM-based autonomous agent for donor engagement and re-activation.
Uses planning, reasoning, and tool-use to:
- Monitor donor engagement levels
- Identify at-risk donors
- Generate personalized outreach
- Orchestrate re-engagement campaigns
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import json
import re
import math

import structlog

from src.core.autonomous_agent import AutonomousAgent, Tool, Task, ReasoningStep, ReasoningType
from src.core.llm_service import get_llm_service
from src.core.mcp_server import get_mcp_server

logger = structlog.get_logger()


class EngagementLevel(str, Enum):
    """Donor engagement levels."""
    HIGHLY_ENGAGED = "highly_engaged"
    ENGAGED = "engaged"
    COOLING = "cooling"
    AT_RISK = "at_risk"
    LAPSED = "lapsed"
    CHURNED = "churned"


class NudgeType(str, Enum):
    """Types of engagement nudges."""
    IMPACT_UPDATE = "impact_update"
    MILESTONE_CELEBRATION = "milestone_celebration"
    CAMPAIGN_RECOMMENDATION = "campaign_recommendation"
    COMMUNITY_INVITATION = "community_invitation"
    ANNIVERSARY_REMINDER = "anniversary_reminder"
    MATCHING_OPPORTUNITY = "matching_opportunity"
    THANK_YOU = "thank_you"
    RE_ENGAGEMENT = "re_engagement"


class ChannelType(str, Enum):
    """Communication channels."""
    EMAIL = "email"
    PUSH_NOTIFICATION = "push_notification"
    IN_APP = "in_app"
    SMS = "sms"


@dataclass
class EngagementSignal:
    """A signal indicating donor engagement."""
    signal_type: str
    timestamp: datetime
    value: Any
    weight: float = 1.0


@dataclass
class DonorEngagement:
    """Engagement state for a donor."""
    donor_id: str
    level: EngagementLevel = EngagementLevel.ENGAGED
    score: float = 0.5
    
    # Activity metrics
    last_donation: Optional[datetime] = None
    last_login: Optional[datetime] = None
    last_interaction: Optional[datetime] = None
    
    # Engagement signals
    signals: List[EngagementSignal] = field(default_factory=list)
    
    # Risk assessment
    churn_risk: float = 0.0
    days_since_activity: int = 0
    
    # LLM insights
    engagement_summary: str = ""
    recommended_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "donor_id": self.donor_id,
            "level": self.level.value,
            "score": self.score,
            "last_donation": self.last_donation.isoformat() if self.last_donation else None,
            "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None,
            "churn_risk": self.churn_risk,
            "days_since_activity": self.days_since_activity,
            "engagement_summary": self.engagement_summary,
            "recommended_actions": self.recommended_actions,
        }


@dataclass
class EngagementNudge:
    """A personalized engagement nudge."""
    nudge_id: str
    donor_id: str
    nudge_type: NudgeType
    channel: ChannelType
    
    # Content
    subject: str = ""
    message: str = ""
    call_to_action: str = ""
    
    # Targeting
    priority: int = 1
    scheduled_at: Optional[datetime] = None
    
    # Tracking
    sent_at: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    clicked_at: Optional[datetime] = None
    converted: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nudge_id": self.nudge_id,
            "donor_id": self.donor_id,
            "nudge_type": self.nudge_type.value,
            "channel": self.channel.value,
            "subject": self.subject,
            "message": self.message,
            "call_to_action": self.call_to_action,
            "priority": self.priority,
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
        }


class EngagementAgent(AutonomousAgent[DonorEngagement]):
    """
    Autonomous agent for donor engagement and re-activation.
    
    This agent:
    1. Monitors engagement signals across donors
    2. Identifies at-risk and lapsed donors
    3. Generates personalized outreach content
    4. Orchestrates multi-channel campaigns
    5. Measures and optimizes engagement strategies
    """
    
    SYSTEM_PROMPT = """You are an expert in donor engagement and retention.
Your expertise includes:
- Understanding donor lifecycle and engagement patterns
- Identifying early signs of disengagement
- Crafting personalized, meaningful outreach
- Optimizing communication timing and channels
- Re-activating lapsed donors with empathy

You understand that engagement is about:
- Building genuine relationships, not transactions
- Showing impact and appreciation
- Respecting donor preferences and boundaries
- Creating value in every interaction

You help maintain strong donor relationships through thoughtful, personalized engagement."""

    def __init__(self):
        super().__init__(
            agent_id="engagement_agent",
            name="Engagement & Re-activation Agent",
            description="Autonomous agent for donor engagement and retention",
            system_prompt=self.SYSTEM_PROMPT,
        )
        
        # Data stores
        self._donor_engagement: Dict[str, DonorEngagement] = {}
        self._nudge_queue: List[EngagementNudge] = []
        self._donor_profiles: Dict[str, Dict[str, Any]] = {}
        self._campaigns: Dict[str, Dict[str, Any]] = {}
        
        # Engagement thresholds
        self._thresholds = {
            "highly_engaged": 0.8,
            "engaged": 0.6,
            "cooling": 0.4,
            "at_risk": 0.2,
            "lapsed": 0.1,
        }
        
        # Register domain-specific tools
        for tool in self._get_domain_tools():
            self.register_tool(tool)
    
    def _get_domain_system_prompt(self) -> str:
        return """
## Domain Expertise: Donor Engagement

You specialize in:

1. **Engagement Monitoring**: Tracking donor health
   - Activity signals (donations, logins, interactions)
   - Engagement scoring and trending
   - Early warning detection
   - Cohort analysis

2. **Personalized Outreach**: Crafting meaningful messages
   - Impact updates that resonate
   - Milestone celebrations
   - Thoughtful recommendations
   - Re-engagement with empathy

3. **Channel Optimization**: Right message, right channel
   - Email for detailed updates
   - Push for timely nudges
   - In-app for active users
   - SMS for urgent matters

4. **Re-activation Strategies**: Winning back lapsed donors
   - Understanding why they left
   - Personalized win-back offers
   - Gradual re-engagement
   - Respectful follow-up
"""
    
    def _get_domain_tools(self) -> List[Tool]:
        return [
            Tool(
                name="calculate_engagement_score",
                description="Calculate engagement score for a donor",
                parameters={
                    "type": "object",
                    "properties": {
                        "donor_id": {"type": "string"},
                        "activity_data": {"type": "object"},
                    },
                    "required": ["donor_id", "activity_data"],
                },
                function=self._tool_calculate_score,
            ),
            Tool(
                name="assess_churn_risk",
                description="Assess the risk of a donor churning",
                parameters={
                    "type": "object",
                    "properties": {
                        "donor_id": {"type": "string"},
                        "engagement": {"type": "object"},
                    },
                    "required": ["donor_id", "engagement"],
                },
                function=self._tool_assess_churn,
            ),
            Tool(
                name="identify_at_risk_donors",
                description="Identify donors at risk of churning",
                parameters={
                    "type": "object",
                    "properties": {
                        "threshold": {"type": "number"},
                    },
                    "required": [],
                },
                function=self._tool_identify_at_risk,
            ),
            Tool(
                name="generate_nudge_content",
                description="Generate personalized nudge content using LLM",
                parameters={
                    "type": "object",
                    "properties": {
                        "donor_id": {"type": "string"},
                        "nudge_type": {"type": "string"},
                        "context": {"type": "object"},
                    },
                    "required": ["donor_id", "nudge_type"],
                },
                function=self._tool_generate_nudge,
            ),
            Tool(
                name="select_optimal_channel",
                description="Select the optimal communication channel for a donor",
                parameters={
                    "type": "object",
                    "properties": {
                        "donor_id": {"type": "string"},
                        "nudge_type": {"type": "string"},
                    },
                    "required": ["donor_id", "nudge_type"],
                },
                function=self._tool_select_channel,
            ),
            Tool(
                name="schedule_nudge",
                description="Schedule a nudge for delivery",
                parameters={
                    "type": "object",
                    "properties": {
                        "nudge": {"type": "object"},
                    },
                    "required": ["nudge"],
                },
                function=self._tool_schedule_nudge,
            ),
            Tool(
                name="generate_impact_update",
                description="Generate an impact update for a donor",
                parameters={
                    "type": "object",
                    "properties": {
                        "donor_id": {"type": "string"},
                        "campaigns": {"type": "array"},
                    },
                    "required": ["donor_id"],
                },
                function=self._tool_generate_impact,
            ),
            Tool(
                name="create_reengagement_plan",
                description="Create a re-engagement plan for a lapsed donor",
                parameters={
                    "type": "object",
                    "properties": {
                        "donor_id": {"type": "string"},
                        "donor_profile": {"type": "object"},
                    },
                    "required": ["donor_id"],
                },
                function=self._tool_create_reengagement,
            ),
            Tool(
                name="send_personalized_thank_you",
                description="Send a personalized thank you email using MCP. Accesses contacts to personalize the message and sends via email.",
                parameters={
                    "type": "object",
                    "properties": {
                        "donor_id": {"type": "string", "description": "Donor ID"},
                        "donor_email": {"type": "string", "description": "Donor email address"},
                        "donation_amount": {"type": "number", "description": "Donation amount"},
                        "campaign_title": {"type": "string", "description": "Campaign title"},
                        "use_contact_info": {"type": "boolean", "description": "Whether to enrich with contact information", "default": True},
                    },
                    "required": ["donor_id", "donor_email", "donation_amount", "campaign_title"],
                },
                function=self._tool_send_personalized_thank_you,
            ),
            Tool(
                name="get_donor_contact_info",
                description="Get contact information for a donor from MCP contacts. Helps personalize messages.",
                parameters={
                    "type": "object",
                    "properties": {
                        "donor_email": {"type": "string", "description": "Donor email to look up"},
                        "donor_name": {"type": "string", "description": "Optional donor name for matching"},
                    },
                    "required": ["donor_email"],
                },
                function=self._tool_get_donor_contact_info,
            ),
            Tool(
                name="analyze_engagement_trends",
                description="Analyze engagement trends across donors",
                parameters={
                    "type": "object",
                    "properties": {
                        "time_period_days": {"type": "integer"},
                    },
                    "required": [],
                },
                function=self._tool_analyze_trends,
            ),
        ]
    
    # ==================== Tool Implementations ====================
    
    async def _tool_calculate_score(
        self,
        donor_id: str,
        activity_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate engagement score for a donor."""
        now = datetime.utcnow()
        score = 0.0
        signals = []
        
        # Recency score (last donation)
        last_donation = activity_data.get("last_donation")
        if last_donation:
            if isinstance(last_donation, str):
                last_donation = datetime.fromisoformat(last_donation.replace("Z", "+00:00"))
            days_since = (now - last_donation.replace(tzinfo=None)).days
            recency_score = math.exp(-days_since / 90)  # Decay over 90 days
            score += recency_score * 0.3
            signals.append(EngagementSignal(
                signal_type="donation_recency",
                timestamp=last_donation,
                value=days_since,
                weight=0.3,
            ))
        
        # Frequency score
        donation_count = activity_data.get("donation_count", 0)
        frequency_score = min(1.0, donation_count / 10)
        score += frequency_score * 0.2
        signals.append(EngagementSignal(
            signal_type="donation_frequency",
            timestamp=now,
            value=donation_count,
            weight=0.2,
        ))
        
        # Login recency
        last_login = activity_data.get("last_login")
        if last_login:
            if isinstance(last_login, str):
                last_login = datetime.fromisoformat(last_login.replace("Z", "+00:00"))
            days_since = (now - last_login.replace(tzinfo=None)).days
            login_score = math.exp(-days_since / 30)
            score += login_score * 0.2
            signals.append(EngagementSignal(
                signal_type="login_recency",
                timestamp=last_login,
                value=days_since,
                weight=0.2,
            ))
        
        # Interaction score
        interactions = activity_data.get("interactions", 0)
        interaction_score = min(1.0, interactions / 20)
        score += interaction_score * 0.15
        
        # Email engagement
        email_opens = activity_data.get("email_opens", 0)
        emails_sent = activity_data.get("emails_sent", 1)
        open_rate = email_opens / emails_sent if emails_sent > 0 else 0
        score += open_rate * 0.15
        
        # Determine level
        if score >= self._thresholds["highly_engaged"]:
            level = EngagementLevel.HIGHLY_ENGAGED
        elif score >= self._thresholds["engaged"]:
            level = EngagementLevel.ENGAGED
        elif score >= self._thresholds["cooling"]:
            level = EngagementLevel.COOLING
        elif score >= self._thresholds["at_risk"]:
            level = EngagementLevel.AT_RISK
        elif score >= self._thresholds["lapsed"]:
            level = EngagementLevel.LAPSED
        else:
            level = EngagementLevel.CHURNED
        
        # Calculate days since activity
        last_activity = max(
            filter(None, [
                activity_data.get("last_donation"),
                activity_data.get("last_login"),
                activity_data.get("last_interaction"),
            ]),
            default=None,
            key=lambda x: datetime.fromisoformat(x.replace("Z", "+00:00")) if isinstance(x, str) else x
        )
        
        days_since_activity = 0
        if last_activity:
            if isinstance(last_activity, str):
                last_activity = datetime.fromisoformat(last_activity.replace("Z", "+00:00"))
            days_since_activity = (now - last_activity.replace(tzinfo=None)).days
        
        # Store engagement
        engagement = DonorEngagement(
            donor_id=donor_id,
            level=level,
            score=round(score, 3),
            last_donation=activity_data.get("last_donation"),
            last_login=activity_data.get("last_login"),
            signals=signals,
            days_since_activity=days_since_activity,
        )
        
        self._donor_engagement[donor_id] = engagement
        
        return {
            "donor_id": donor_id,
            "score": round(score, 3),
            "level": level.value,
            "days_since_activity": days_since_activity,
        }
    
    async def _tool_assess_churn(
        self,
        donor_id: str,
        engagement: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess churn risk for a donor."""
        score = engagement.get("score", 0.5)
        days_since = engagement.get("days_since_activity", 0)
        level = engagement.get("level", "engaged")
        
        # Calculate churn risk
        risk = 0.0
        factors = []
        
        # Score-based risk
        if score < 0.3:
            risk += 0.3
            factors.append("Low engagement score")
        
        # Days since activity
        if days_since > 90:
            risk += 0.3
            factors.append(f"Inactive for {days_since} days")
        elif days_since > 60:
            risk += 0.2
            factors.append(f"Cooling off ({days_since} days inactive)")
        elif days_since > 30:
            risk += 0.1
            factors.append("Reduced activity")
        
        # Level-based risk
        level_risks = {
            "at_risk": 0.2,
            "lapsed": 0.3,
            "churned": 0.4,
        }
        if level in level_risks:
            risk += level_risks[level]
            factors.append(f"Current status: {level}")
        
        risk = min(1.0, risk)
        
        # Update engagement record
        if donor_id in self._donor_engagement:
            self._donor_engagement[donor_id].churn_risk = risk
        
        return {
            "donor_id": donor_id,
            "churn_risk": round(risk, 3),
            "risk_level": "high" if risk > 0.6 else "medium" if risk > 0.3 else "low",
            "factors": factors,
        }
    
    async def _tool_identify_at_risk(
        self,
        threshold: float = 0.4
    ) -> Dict[str, Any]:
        """Identify donors at risk of churning."""
        at_risk = []
        
        for donor_id, engagement in self._donor_engagement.items():
            if engagement.churn_risk >= threshold:
                at_risk.append({
                    "donor_id": donor_id,
                    "level": engagement.level.value,
                    "score": engagement.score,
                    "churn_risk": engagement.churn_risk,
                    "days_inactive": engagement.days_since_activity,
                })
        
        # Sort by risk
        at_risk.sort(key=lambda x: x["churn_risk"], reverse=True)
        
        return {
            "at_risk_count": len(at_risk),
            "threshold": threshold,
            "donors": at_risk[:20],
        }
    
    async def _tool_generate_nudge(
        self,
        donor_id: str,
        nudge_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate personalized nudge content."""
        context = context or {}
        profile = self._donor_profiles.get(donor_id, {})
        engagement = self._donor_engagement.get(donor_id)
        
        nudge_prompts = {
            "impact_update": """Write a brief, warm impact update for a donor.
Show them the difference their giving has made. Be specific and personal.""",
            
            "milestone_celebration": """Write a celebratory message for a donor milestone.
Make them feel appreciated and valued. Be genuine, not salesy.""",
            
            "campaign_recommendation": """Write a personalized campaign recommendation.
Connect to their interests. Be helpful, not pushy.""",
            
            "re_engagement": """Write a warm re-engagement message for a lapsed donor.
Acknowledge their past support. Invite them back gently. No guilt.""",
            
            "thank_you": """Write a heartfelt thank you message.
Be specific about their impact. Make it personal and memorable.""",
        }
        
        base_prompt = nudge_prompts.get(nudge_type, nudge_prompts["thank_you"])
        
        prompt = f"""{base_prompt}

Donor Info:
- Name: {profile.get('display_name', 'Valued Donor')}
- Interests: {profile.get('cause_affinities', [])}
- Last donation: {engagement.last_donation if engagement else 'Unknown'}
- Engagement level: {engagement.level.value if engagement else 'Unknown'}

Context: {json.dumps(context, default=str)[:500]}

Generate:
1. Subject line (for email)
2. Main message (2-3 sentences)
3. Call to action (one sentence)

Respond in JSON:
{{
    "subject": "...",
    "message": "...",
    "call_to_action": "..."
}}"""

        response = await self._llm._provider.complete(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
        )
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                content = json.loads(json_match.group())
                return {
                    "donor_id": donor_id,
                    "nudge_type": nudge_type,
                    **content,
                }
        except:
            pass
        
        return {
            "donor_id": donor_id,
            "nudge_type": nudge_type,
            "subject": "We appreciate you",
            "message": "Thank you for being part of our community.",
            "call_to_action": "Continue making a difference",
        }
    
    async def _tool_select_channel(
        self,
        donor_id: str,
        nudge_type: str
    ) -> Dict[str, Any]:
        """Select optimal communication channel."""
        profile = self._donor_profiles.get(donor_id, {})
        engagement = self._donor_engagement.get(donor_id)
        
        # Channel preferences based on nudge type
        type_preferences = {
            "impact_update": [ChannelType.EMAIL, ChannelType.IN_APP],
            "milestone_celebration": [ChannelType.EMAIL, ChannelType.PUSH_NOTIFICATION],
            "campaign_recommendation": [ChannelType.IN_APP, ChannelType.EMAIL],
            "re_engagement": [ChannelType.EMAIL],
            "thank_you": [ChannelType.EMAIL, ChannelType.PUSH_NOTIFICATION],
            "matching_opportunity": [ChannelType.PUSH_NOTIFICATION, ChannelType.SMS],
        }
        
        preferences = type_preferences.get(nudge_type, [ChannelType.EMAIL])
        
        # Adjust based on engagement
        if engagement:
            if engagement.level in [EngagementLevel.HIGHLY_ENGAGED, EngagementLevel.ENGAGED]:
                # Active users: in-app or push
                if ChannelType.IN_APP in preferences:
                    selected = ChannelType.IN_APP
                elif ChannelType.PUSH_NOTIFICATION in preferences:
                    selected = ChannelType.PUSH_NOTIFICATION
                else:
                    selected = preferences[0]
            else:
                # Less active: email
                selected = ChannelType.EMAIL
        else:
            selected = preferences[0]
        
        return {
            "donor_id": donor_id,
            "nudge_type": nudge_type,
            "selected_channel": selected.value,
            "reasoning": f"Based on engagement level and nudge type",
        }
    
    async def _tool_schedule_nudge(
        self,
        nudge: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Schedule a nudge for delivery."""
        nudge_id = f"nudge_{nudge.get('donor_id')}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        scheduled_nudge = EngagementNudge(
            nudge_id=nudge_id,
            donor_id=nudge.get("donor_id", ""),
            nudge_type=NudgeType(nudge.get("nudge_type", "thank_you")),
            channel=ChannelType(nudge.get("channel", "email")),
            subject=nudge.get("subject", ""),
            message=nudge.get("message", ""),
            call_to_action=nudge.get("call_to_action", ""),
            scheduled_at=datetime.utcnow() + timedelta(hours=1),
        )
        
        self._nudge_queue.append(scheduled_nudge)
        
        return {
            "status": "scheduled",
            "nudge_id": nudge_id,
            "scheduled_at": scheduled_nudge.scheduled_at.isoformat(),
        }
    
    async def _tool_generate_impact(
        self,
        donor_id: str,
        campaigns: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Generate impact update for a donor."""
        profile = self._donor_profiles.get(donor_id, {})
        
        # Get campaigns they've supported
        supported_campaigns = campaigns or []
        if not supported_campaigns:
            # Use sample data
            supported_campaigns = [
                {"title": "Local Food Bank", "raised": 5000, "impact": "Fed 200 families"},
            ]
        
        prompt = f"""Create a personalized impact update for this donor:

Donor: {profile.get('display_name', 'Valued Donor')}
Their total giving: ${profile.get('total_lifetime_giving', 0):,.0f}

Campaigns they've supported:
{json.dumps(supported_campaigns[:3], indent=2, default=str)}

Write a 2-3 sentence impact update that:
1. Shows specific, tangible impact
2. Makes them feel like a hero
3. Connects their giving to real outcomes

Be warm and specific, not generic."""

        impact_message = await self._llm._provider.complete(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
        )
        
        return {
            "donor_id": donor_id,
            "impact_message": impact_message.strip(),
            "campaigns_referenced": len(supported_campaigns),
        }
    
    async def _tool_create_reengagement(
        self,
        donor_id: str,
        donor_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a re-engagement plan for a lapsed donor."""
        donor_profile = donor_profile or self._donor_profiles.get(donor_id, {})
        engagement = self._donor_engagement.get(donor_id)
        
        # Analyze why they might have lapsed
        prompt = f"""Create a re-engagement plan for this lapsed donor:

Donor Profile:
- Interests: {donor_profile.get('cause_affinities', [])}
- Last donation: {engagement.last_donation if engagement else 'Unknown'}
- Days inactive: {engagement.days_since_activity if engagement else 'Unknown'}
- Previous engagement: {engagement.level.value if engagement else 'Unknown'}

Create a 3-step re-engagement plan:
1. Initial gentle outreach (what to say)
2. Follow-up if no response (different angle)
3. Final attempt (special offer or appeal)

For each step, provide timing and message theme.

Respond in JSON:
{{
    "analysis": "brief analysis of why they may have lapsed",
    "steps": [
        {{"timing": "day 0", "theme": "...", "message_idea": "..."}},
        {{"timing": "day 7", "theme": "...", "message_idea": "..."}},
        {{"timing": "day 14", "theme": "...", "message_idea": "..."}}
    ]
}}"""

        response = await self._llm._provider.complete(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.5,
        )
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                plan = json.loads(json_match.group())
                return {
                    "donor_id": donor_id,
                    "plan": plan,
                }
        except:
            pass
        
        return {
            "donor_id": donor_id,
            "plan": {
                "analysis": "Unable to generate detailed analysis",
                "steps": [
                    {"timing": "day 0", "theme": "We miss you", "message_idea": "Gentle check-in"},
                    {"timing": "day 7", "theme": "Impact update", "message_idea": "Show what they've helped achieve"},
                    {"timing": "day 14", "theme": "Special opportunity", "message_idea": "Matching gift or special campaign"},
                ],
            },
        }
    
    async def _tool_analyze_trends(
        self,
        time_period_days: int = 30
    ) -> Dict[str, Any]:
        """Analyze engagement trends across donors."""
        levels = defaultdict(int)
        risk_distribution = {"high": 0, "medium": 0, "low": 0}
        
        for engagement in self._donor_engagement.values():
            levels[engagement.level.value] += 1
            
            if engagement.churn_risk > 0.6:
                risk_distribution["high"] += 1
            elif engagement.churn_risk > 0.3:
                risk_distribution["medium"] += 1
            else:
                risk_distribution["low"] += 1
        
        total = len(self._donor_engagement)
        
        return {
            "total_donors": total,
            "level_distribution": dict(levels),
            "risk_distribution": risk_distribution,
            "healthy_percentage": round(
                (levels.get("highly_engaged", 0) + levels.get("engaged", 0)) / total * 100
                if total > 0 else 0,
                1
            ),
            "at_risk_percentage": round(
                (levels.get("at_risk", 0) + levels.get("lapsed", 0) + levels.get("churned", 0)) / total * 100
                if total > 0 else 0,
                1
            ),
        }
    
    # ==================== Public API ====================
    
    def register_donor(self, donor_id: str, profile: Dict[str, Any]) -> None:
        """Register a donor profile."""
        self._donor_profiles[donor_id] = profile
    
    def register_campaign(self, campaign: Dict[str, Any]) -> None:
        """Register a campaign."""
        campaign_id = campaign.get("campaign_id", str(hash(campaign.get("title", ""))))
        self._campaigns[campaign_id] = campaign
    
    async def assess_donor_engagement(
        self,
        donor_id: str,
        activity_data: Dict[str, Any]
    ) -> DonorEngagement:
        """
        Autonomously assess a donor's engagement level.
        
        The agent will:
        1. Calculate engagement score
        2. Assess churn risk
        3. Generate insights
        4. Recommend actions
        """
        goal = f"Assess engagement and churn risk for donor {donor_id}"
        
        context = {
            "donor_id": donor_id,
            "activity_data": activity_data,
        }
        
        # Run autonomous assessment
        await self.run(goal, context)
        
        return self._donor_engagement.get(donor_id, DonorEngagement(donor_id=donor_id))
    
    async def _tool_get_donor_contact_info(
        self,
        donor_email: str,
        donor_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get contact information for a donor from MCP contacts."""
        mcp = get_mcp_server()
        
        # Search contacts by email
        filter_dict = {"email": donor_email}
        if donor_name:
            filter_dict["name"] = donor_name
        
        try:
            result = await mcp.call_tool("get_contacts", {"filter": filter_dict, "limit": 1})
            contacts = result.get("contacts", [])
            
            if contacts:
                contact = contacts[0]
                return {
                    "found": True,
                    "contact": {
                        "name": contact.get("name", ""),
                        "email": contact.get("email", donor_email),
                        "phone": contact.get("phone", ""),
                        "tags": contact.get("tags", []),
                        "notes": contact.get("notes", ""),
                    },
                }
            else:
                return {
                    "found": False,
                    "contact": {
                        "email": donor_email,
                        "name": donor_name or "Valued Donor",
                    },
                }
        except Exception as e:
            self._logger.warning("mcp_contact_lookup_failed", error=str(e))
            return {
                "found": False,
                "contact": {
                    "email": donor_email,
                    "name": donor_name or "Valued Donor",
                },
                "error": str(e),
            }
    
    async def _tool_send_personalized_thank_you(
        self,
        donor_id: str,
        donor_email: str,
        donation_amount: float,
        campaign_title: str,
        use_contact_info: bool = True
    ) -> Dict[str, Any]:
        """
        Send a personalized thank you email using MCP.
        
        This tool:
        1. Gets contact information from MCP (if available)
        2. Generates personalized thank you message using LLM
        3. Sends email via MCP
        """
        mcp = get_mcp_server()
        
        # Get contact info for personalization
        contact_info = {}
        if use_contact_info:
            contact_result = await self._tool_get_donor_contact_info(donor_email)
            contact_info = contact_result.get("contact", {})
        
        # Get donor profile for context
        profile = self._donor_profiles.get(donor_id, {})
        engagement = self._donor_engagement.get(donor_id)
        
        # Generate personalized thank you message
        donor_name = contact_info.get("name") or profile.get("display_name") or "Valued Donor"
        
        prompt = f"""Write a heartfelt, personalized thank you email for a donor.

Donor Information:
- Name: {donor_name}
- Email: {donor_email}
- Donation Amount: ${donation_amount:.2f}
- Campaign: {campaign_title}
- Contact Notes: {contact_info.get('notes', 'None')}
- Tags: {', '.join(contact_info.get('tags', []))}

Donor Profile:
- Interests: {profile.get('cause_affinities', [])}
- Giving Pattern: {profile.get('primary_pattern', 'N/A')}
- Last Engagement: {engagement.last_interaction if engagement else 'N/A'}

Write:
1. A warm, personal subject line (max 60 characters)
2. A heartfelt message (3-4 sentences) that:
   - Thanks them by name
   - Acknowledges their specific donation amount
   - Mentions the campaign they supported
   - Shows the impact of their gift
   - Feels genuine and personal (use contact notes/tags if available)
3. A brief closing

Respond in JSON:
{{
    "subject": "...",
    "body": "...",
    "closing": "..."
}}"""

        try:
            response = await self._llm._provider.complete(
                prompt=prompt,
                system_prompt="You write warm, personal thank you messages that make donors feel truly appreciated.",
                temperature=0.7,
            )
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                content = json.loads(json_match.group())
                subject = content.get("subject", f"Thank you for your generous donation")
                body = content.get("body", f"Dear {donor_name},\n\nThank you for your generous donation of ${donation_amount:.2f} to {campaign_title}.")
                closing = content.get("closing", "With gratitude,")
            else:
                # Fallback
                subject = f"Thank you for your generous donation"
                body = f"Dear {donor_name},\n\nThank you for your generous donation of ${donation_amount:.2f} to {campaign_title}. Your support makes a real difference."
                closing = "With gratitude,"
        except Exception as e:
            self._logger.warning("llm_thank_you_generation_failed", error=str(e))
            # Fallback message
            subject = f"Thank you for your generous donation"
            body = f"Dear {donor_name},\n\nThank you for your generous donation of ${donation_amount:.2f} to {campaign_title}. Your support makes a real difference."
            closing = "With gratitude,"
        
        # Format full email body
        full_body = f"{body}\n\n{closing}\nThe Giving Intelligence Team"
        
        # Send email via MCP
        try:
            email_result = await mcp.call_tool(
                "send_email",
                {
                    "to": donor_email,
                    "subject": subject,
                    "body": full_body,
                    "body_type": "text",
                }
            )
            
            self._logger.info(
                "personalized_thank_you_sent",
                donor_id=donor_id,
                donor_email=donor_email,
                donation_amount=donation_amount,
                used_contact_info=use_contact_info and contact_info.get("found", False),
            )
            
            return {
                "success": True,
                "message_id": email_result.get("message_id"),
                "donor_email": donor_email,
                "subject": subject,
                "personalized_with_contact": use_contact_info and contact_info.get("found", False),
                "contact_name": contact_info.get("name"),
            }
        except Exception as e:
            self._logger.error("mcp_email_send_failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "donor_email": donor_email,
            }
    
    async def send_nudge(
        self,
        donor_id: str,
        nudge_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EngagementNudge:
        """
        Generate and schedule a personalized nudge.
        """
        # Generate content
        content = await self._tool_generate_nudge(donor_id, nudge_type, context)
        
        # Select channel
        channel = await self._tool_select_channel(donor_id, nudge_type)
        
        # Create nudge
        nudge_data = {
            "donor_id": donor_id,
            "nudge_type": nudge_type,
            "channel": channel["selected_channel"],
            **content,
        }
        
        # Schedule
        result = await self._tool_schedule_nudge(nudge_data)
        
        return self._nudge_queue[-1] if self._nudge_queue else EngagementNudge(
            nudge_id="unknown",
            donor_id=donor_id,
            nudge_type=NudgeType(nudge_type),
            channel=ChannelType.EMAIL,
        )
    
    async def get_at_risk_donors(self, threshold: float = 0.4) -> List[Dict[str, Any]]:
        """Get list of at-risk donors."""
        result = await self._tool_identify_at_risk(threshold)
        return result.get("donors", [])
    
    async def create_engagement_plan(
        self,
        donor_profile: Dict[str, Any],
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a comprehensive engagement plan for a donor.
        
        Returns a plan with recommended actions, timing, and channels.
        """
        donor_id = donor_profile.get("donor_id", "unknown")
        
        # Use autonomous planning
        goal = f"Create engagement plan for donor {donor_id}"
        context = {
            "donor_profile": donor_profile,
            "additional_context": additional_context or {},
        }
        
        await self.run(goal, context)
        
        # Build engagement plan from agent's work
        engagement = self._donor_engagement.get(donor_id)
        
        plan = {
            "donor_id": donor_id,
            "recommended_actions": [],
            "channels": [],
            "timing": {},
        }
        
        if engagement:
            # Generate recommended actions
            if engagement.engagement_summary:
                plan["recommended_actions"].append({
                    "action": "Send personalized update",
                    "reasoning": engagement.engagement_summary,
                })
            
            if engagement.recommended_actions:
                for action in engagement.recommended_actions:
                    plan["recommended_actions"].append({
                        "action": action,
                        "reasoning": f"Based on engagement level: {engagement.level.value}",
                    })
        
        # Add default actions if none
        if not plan["recommended_actions"]:
            plan["recommended_actions"] = [
                {
                    "action": "Send impact update",
                    "reasoning": "Maintain donor engagement with campaign progress",
                },
                {
                    "action": "Recommend similar campaigns",
                    "reasoning": "Based on donor's cause affinities",
                },
            ]
        
        return plan
    
    async def create_reengagement_campaign(
        self,
        donor_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Create a re-engagement campaign for multiple lapsed donors.
        """
        plans = []
        
        for donor_id in donor_ids:
            plan = await self._tool_create_reengagement(donor_id)
            plans.append(plan)
        
        return {
            "campaign_name": f"Re-engagement Campaign {datetime.utcnow().strftime('%Y-%m-%d')}",
            "donor_count": len(donor_ids),
            "plans": plans,
        }
    
    async def get_engagement_trends(self) -> Dict[str, Any]:
        """Get overall engagement trends."""
        return await self._tool_analyze_trends()
