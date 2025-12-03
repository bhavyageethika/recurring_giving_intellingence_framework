"""
Agent 4: Recurring Opportunity Curator (Autonomous Agent)

An LLM-based autonomous agent that curates recurring giving opportunities.
Uses planning, reasoning, and tool-use to:
- Identify campaigns suitable for recurring support
- Design personalized recurring giving plans
- Optimize giving schedules for donor preferences
- Track and adjust recurring commitments
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
from src.core.agent_collaboration import get_collaborator

logger = structlog.get_logger()


class RecurringFrequency(str, Enum):
    """Frequency options for recurring giving."""
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"


class CampaignSuitability(str, Enum):
    """Suitability of campaigns for recurring giving."""
    HIGHLY_SUITABLE = "highly_suitable"
    SUITABLE = "suitable"
    MARGINALLY_SUITABLE = "marginally_suitable"
    NOT_SUITABLE = "not_suitable"


@dataclass
class RecurringOpportunity:
    """A curated recurring giving opportunity."""
    opportunity_id: str
    campaign_id: str
    campaign_title: str
    campaign_url: str = ""
    
    # Suitability assessment
    suitability: CampaignSuitability = CampaignSuitability.SUITABLE
    suitability_score: float = 0.5
    suitability_reasons: List[str] = field(default_factory=list)
    
    # Recommended plan
    recommended_frequency: RecurringFrequency = RecurringFrequency.MONTHLY
    recommended_amount: float = 0.0
    impact_per_period: str = ""
    
    # LLM-generated content
    pitch: str = ""
    impact_story: str = ""
    
    # Metrics
    current_recurring_donors: int = 0
    monthly_recurring_total: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "opportunity_id": self.opportunity_id,
            "campaign_id": self.campaign_id,
            "campaign_title": self.campaign_title,
            "campaign_url": self.campaign_url,
            "suitability": self.suitability.value,
            "suitability_score": self.suitability_score,
            "suitability_reasons": self.suitability_reasons,
            "recommended_frequency": self.recommended_frequency.value,
            "recommended_amount": self.recommended_amount,
            "impact_per_period": self.impact_per_period,
            "pitch": self.pitch,
            "impact_story": self.impact_story,
        }


@dataclass
class RecurringPlan:
    """A personalized recurring giving plan for a donor."""
    plan_id: str
    donor_id: str
    
    # Plan details
    opportunities: List[RecurringOpportunity] = field(default_factory=list)
    total_monthly_commitment: float = 0.0
    
    # Schedule
    frequency: RecurringFrequency = RecurringFrequency.MONTHLY
    preferred_day: int = 1  # Day of month
    start_date: Optional[datetime] = None
    
    # LLM-generated
    plan_summary: str = ""
    diversification_analysis: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "donor_id": self.donor_id,
            "opportunities": [o.to_dict() for o in self.opportunities],
            "total_monthly_commitment": self.total_monthly_commitment,
            "frequency": self.frequency.value,
            "preferred_day": self.preferred_day,
            "plan_summary": self.plan_summary,
            "diversification_analysis": self.diversification_analysis,
        }


class RecurringCuratorAgent(AutonomousAgent[RecurringOpportunity]):
    """
    Autonomous agent for curating recurring giving opportunities.
    
    This agent:
    1. Analyzes campaigns for recurring suitability
    2. Designs personalized recurring giving plans
    3. Optimizes giving schedules
    4. Generates compelling recurring giving pitches
    5. Tracks and adjusts commitments over time
    """
    
    SYSTEM_PROMPT = """You are an expert in recurring giving and donor engagement.
Your expertise includes:
- Identifying campaigns suitable for sustained support
- Understanding donor capacity and preferences
- Designing impactful recurring giving programs
- Communicating the power of sustained giving
- Optimizing giving schedules for maximum impact

You understand that recurring giving creates:
- Predictable funding for causes
- Deeper donor engagement
- Compounding impact over time
- Stronger donor-cause relationships

You help donors find the right recurring commitments that match their values and capacity."""

    def __init__(self):
        super().__init__(
            agent_id="recurring_curator",
            name="Recurring Opportunity Curator",
            description="Autonomous agent for curating recurring giving opportunities",
            system_prompt=self.SYSTEM_PROMPT,
        )
        
        # Data stores
        self._campaigns: Dict[str, Dict[str, Any]] = {}
        self._opportunities: Dict[str, RecurringOpportunity] = {}
        self._donor_plans: Dict[str, RecurringPlan] = {}
        
        # Register domain-specific tools
        for tool in self._get_domain_tools():
            self.register_tool(tool)
        
        # Initialize collaborator for A2A communication
        self._collaborator = get_collaborator(self.agent_id)
    
    def _get_domain_system_prompt(self) -> str:
        return """
## Domain Expertise: Recurring Giving

You specialize in:

1. **Suitability Assessment**: Evaluating campaigns for recurring support
   - Ongoing needs vs. one-time goals
   - Organizational stability
   - Track record of updates and transparency
   - Clear impact metrics

2. **Plan Design**: Creating personalized recurring plans
   - Matching donor capacity
   - Diversifying across causes
   - Optimizing frequency and amounts
   - Balancing impact and sustainability

3. **Impact Communication**: Showing the power of recurring gifts
   - "Your $25/month provides meals for a family for a year"
   - Cumulative impact projections
   - Comparison to one-time giving

4. **Schedule Optimization**: Timing recurring gifts
   - Aligning with donor pay cycles
   - Coordinating multiple commitments
   - Seasonal adjustments
"""
    
    def _get_domain_tools(self) -> List[Tool]:
        return [
            Tool(
                name="assess_recurring_suitability",
                description="Assess if a campaign is suitable for recurring giving. Campaign data is available in context if not provided.",
                parameters={
                    "type": "object",
                    "properties": {
                        "campaign_id": {
                            "type": "string",
                            "description": "Campaign ID (optional, will use context if not provided)"
                        },
                        "campaign_data": {
                            "type": "object",
                            "description": "Campaign data object (optional, will use context if not provided)"
                        },
                    },
                    "required": [],
                },
                function=self._tool_assess_suitability,
            ),
            Tool(
                name="calculate_recommended_amount",
                description="Calculate recommended recurring amount based on donor profile",
                parameters={
                    "type": "object",
                    "properties": {
                        "donor_profile": {"type": "object"},
                        "campaign_data": {"type": "object"},
                    },
                    "required": ["donor_profile", "campaign_data"],
                },
                function=self._tool_calculate_amount,
            ),
            Tool(
                name="generate_impact_projection",
                description="Generate impact projection for recurring giving",
                parameters={
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number"},
                        "frequency": {"type": "string"},
                        "campaign_data": {"type": "object"},
                    },
                    "required": ["amount", "frequency", "campaign_data"],
                },
                function=self._tool_project_impact,
            ),
            Tool(
                name="generate_recurring_pitch",
                description="Generate a compelling pitch for recurring giving",
                parameters={
                    "type": "object",
                    "properties": {
                        "opportunity": {"type": "object"},
                        "donor_profile": {"type": "object"},
                    },
                    "required": ["opportunity"],
                },
                function=self._tool_generate_pitch,
            ),
            Tool(
                name="design_giving_plan",
                description="Design a personalized recurring giving plan",
                parameters={
                    "type": "object",
                    "properties": {
                        "donor_id": {"type": "string"},
                        "donor_profile": {"type": "object"},
                        "budget": {"type": "number"},
                        "preferences": {"type": "object"},
                    },
                    "required": ["donor_id", "donor_profile"],
                },
                function=self._tool_design_plan,
            ),
            Tool(
                name="optimize_schedule",
                description="Optimize the timing of recurring gifts",
                parameters={
                    "type": "object",
                    "properties": {
                        "donor_id": {"type": "string"},
                        "commitments": {"type": "array"},
                        "preferences": {"type": "object"},
                    },
                    "required": ["donor_id", "commitments"],
                },
                function=self._tool_optimize_schedule,
            ),
            Tool(
                name="analyze_diversification",
                description="Analyze cause diversification in a giving plan",
                parameters={
                    "type": "object",
                    "properties": {
                        "plan": {"type": "object"},
                    },
                    "required": ["plan"],
                },
                function=self._tool_analyze_diversification,
            ),
            Tool(
                name="create_recurring_opportunity",
                description="Create a RecurringOpportunity object from assessment results. Use this after assessing suitability, calculating amount, and generating pitch.",
                parameters={
                    "type": "object",
                    "properties": {
                        "campaign_id": {"type": "string"},
                        "campaign_title": {"type": "string"},
                        "suitability": {"type": "string"},
                        "suitability_score": {"type": "number"},
                        "suitability_reasons": {"type": "array", "items": {"type": "string"}},
                        "recommended_amount": {"type": "number"},
                        "impact_per_period": {"type": "string"},
                        "pitch": {"type": "string"},
                    },
                    "required": ["campaign_id", "campaign_title", "suitability_score", "recommended_amount"],
                },
                function=self._tool_create_opportunity,
            ),
        ]
    
    # ==================== Tool Implementations ====================
    
    async def _tool_assess_suitability(
        self,
        campaign_id: Optional[str] = None,
        campaign_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Assess campaign suitability for recurring giving."""
        # Get from context if not provided
        if not campaign_data:
            context = self._memory.current_context or {}
            if "campaigns" in context and context["campaigns"]:
                # Use first campaign if campaign_id not specified
                if campaign_id:
                    for camp in context["campaigns"]:
                        if camp.get("campaign_id") == campaign_id:
                            campaign_data = camp
                            break
                else:
                    campaign_data = context["campaigns"][0]
                    campaign_id = campaign_data.get("campaign_id", "")
            elif "campaign_data" in context:
                campaign_data = context["campaign_data"]
                campaign_id = campaign_data.get("campaign_id", campaign_id or "")
        
        if not campaign_data:
            return {
                "campaign_id": campaign_id or "unknown",
                "suitability": "not_suitable",
                "suitability_score": 0.0,
                "reasons": ["No campaign data provided"],
            }
        
        if not campaign_id:
            campaign_id = campaign_data.get("campaign_id", str(hash(campaign_data.get("title", ""))))
        
        # AGENT COLLABORATION: Ask Campaign Matcher for legitimacy check
        legitimacy_result = await self._collaborator.ask_matcher_for_legitimacy(campaign_data)
        is_legitimate = legitimacy_result.get("is_legitimate", True)
        legitimacy_score = legitimacy_result.get("legitimacy_score", 0.5)
        
        if not is_legitimate or legitimacy_score < 0.3:
            return {
                "campaign_id": campaign_id,
                "suitability": "not_suitable",
                "suitability_score": 0.0,
                "reasons": ["Campaign legitimacy concerns identified by Campaign Matching Engine"],
            }
        
        score = 0.0
        reasons = []
        
        description = campaign_data.get("description", "").lower()
        goal = campaign_data.get("goal_amount", 0)
        raised = campaign_data.get("raised_amount", 0)
        donor_count = campaign_data.get("donor_count", 0)
        category = campaign_data.get("category", "").lower()
        
        # Base score - most campaigns can benefit from recurring support
        score += 0.25
        # Don't add generic reason - will use LLM-specific reasons instead
        
        # Boost score based on legitimacy assessment from Matcher
        if legitimacy_score > 0.7:
            score += 0.1
            reasons.append(f"High legitimacy score ({legitimacy_score:.2f}) from Campaign Matching Engine")
        
        # Check for ongoing need keywords
        ongoing_keywords = ["ongoing", "monthly", "recurring", "sustained", "continuous", "long-term", "chronic", "treatment", "therapy"]
        if any(kw in description for kw in ongoing_keywords):
            score += 0.2
            # Find the specific keyword to make reason more specific
            found_keyword = next((kw for kw in ongoing_keywords if kw in description), None)
            if found_keyword:
                reasons.append(f"Campaign mentions '{found_keyword}' indicating ongoing need")
        
        # Category-based scoring - some categories naturally need ongoing support
        recurring_categories = ["medical", "education", "community", "animals", "environment"]
        if any(cat in category for cat in recurring_categories):
            score += 0.15
            reasons.append(f"{category.title()} campaigns often benefit from recurring support")
        
        # Campaign legitimacy indicators
        if raised > 0:
            score += 0.1
            reasons.append("Campaign has received support (shows legitimacy)")
        
        if donor_count > 0:
            score += 0.1
            reasons.append(f"Has {donor_count} donors (shows engagement)")
        
        # Check for organization vs individual
        if campaign_data.get("beneficiary_type") == "organization":
            score += 0.15
            reasons.append("Organizational beneficiary suggests sustained need")
        
        # Check update frequency
        updates = campaign_data.get("updates", [])
        if len(updates) >= 3:
            score += 0.15
            reasons.append("Regular updates show ongoing engagement")
        elif len(updates) > 0:
            score += 0.05
            reasons.append("Has updates (shows activity)")
        
        # Check for recurring donors already
        recurring_donors = campaign_data.get("recurring_donors", 0)
        if recurring_donors > 0:
            score += 0.15
            reasons.append(f"Already has {recurring_donors} recurring donors")
        
        # Check campaign age and activity
        created_at = campaign_data.get("created_at")
        if created_at:
            try:
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                age_days = (datetime.utcnow() - created_at.replace(tzinfo=None)).days
                if age_days > 30:
                    score += 0.1
                    reasons.append("Established campaign with history")
                elif age_days > 7:
                    score += 0.05
                    reasons.append("Active campaign")
            except:
                pass
        
        # Funding progress - if not fully funded, recurring support makes sense
        if goal > 0:
            funding_pct = (raised / goal) * 100
            if funding_pct < 100:
                score += 0.1
                reasons.append(f"Still needs funding ({funding_pct:.0f}% of goal)")
            elif funding_pct >= 100:
                score += 0.05
                reasons.append("Goal reached, but ongoing support still valuable")
        
        # Use LLM for deeper, campaign-specific assessment
        prompt = f"""Assess if this campaign is suitable for recurring giving:

Title: {campaign_data.get('title', '')}
Description: {description[:500]}
Category: {category}
Goal: ${campaign_data.get('goal_amount', 0):,.0f}
Raised: ${campaign_data.get('raised_amount', 0):,.0f}
Donors: {donor_count}
Updates: {len(updates)}

Is this campaign suitable for recurring monthly support? Consider:
1. Is the need ongoing or one-time? (e.g., chronic illness = ongoing, one-time surgery = one-time)
2. Would recurring support make sense for THIS specific campaign?
3. What makes THIS campaign unique for recurring giving?

Respond with JSON:
{{
    "suitable": true/false,
    "confidence": 0-1,
    "specific_reasons": ["reason 1 specific to this campaign", "reason 2 specific to this campaign"],
    "reasoning": "brief explanation of why this PARTICULAR campaign is suitable",
    "suggested_monthly_impact": "what $25/month could do for THIS specific cause"
}}

IMPORTANT: Make the reasons SPECIFIC to this campaign. Don't use generic phrases like "most campaigns can benefit". Instead, reference specific details from the title, description, or category."""

        response = await self._llm._provider.complete(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.4,  # Slightly higher for more creative, specific responses
        )
        
        llm_reasons = []
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                llm_assessment = json.loads(json_match.group())
                if llm_assessment.get("suitable"):
                    score += 0.2
                    # Use specific reasons from LLM instead of generic ones
                    if llm_assessment.get("specific_reasons"):
                        llm_reasons = llm_assessment["specific_reasons"]
                    elif llm_assessment.get("reasoning"):
                        llm_reasons = [llm_assessment["reasoning"]]
        except Exception as e:
            self._logger.warning("llm_assessment_parse_failed", error=str(e))
            # Continue with rule-based score
        
        # Ensure minimum score for legitimate campaigns
        if score < 0.3 and (raised > 0 or donor_count > 0):
            score = 0.35  # Boost to threshold for campaigns with any support
            if not reasons:  # Only add if we don't have specific reasons yet
                reasons.append(f"Campaign has {donor_count} donors and ${raised:,.0f} raised, showing community support")
        
        # Prioritize LLM-specific reasons over generic rule-based ones
        if llm_reasons:
            # Use LLM reasons as primary, supplement with top rule-based reasons
            final_reasons = llm_reasons[:2]  # Top 2 LLM reasons
            # Add 1-2 most specific rule-based reasons if they add value
            specific_rule_reasons = [r for r in reasons if not any(generic in r.lower() for generic in ["most campaigns", "can benefit", "indicates ongoing"])]
            if specific_rule_reasons:
                final_reasons.extend(specific_rule_reasons[:2])
            reasons = final_reasons[:3]  # Max 3 reasons total
        else:
            # Fallback: use rule-based reasons but filter out generic ones
            reasons = [r for r in reasons if not any(generic in r.lower() for generic in ["most campaigns", "can benefit"])][:3]
        
        # Determine suitability level
        if score >= 0.7:
            suitability = CampaignSuitability.HIGHLY_SUITABLE
        elif score >= 0.5:
            suitability = CampaignSuitability.SUITABLE
        elif score >= 0.3:
            suitability = CampaignSuitability.MARGINALLY_SUITABLE
        else:
            suitability = CampaignSuitability.NOT_SUITABLE
        
        final_score = round(min(1.0, score), 3)
        
        self._logger.debug(
            "suitability_assessment",
            campaign_id=campaign_id,
            score=final_score,
            suitability=suitability.value,
            reasons_count=len(reasons),
        )
        
        return {
            "campaign_id": campaign_id,
            "suitability": suitability.value,
            "suitability_score": final_score,
            "reasons": reasons,
        }
    
    async def _tool_calculate_amount(
        self,
        donor_profile: Dict[str, Any],
        campaign_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate recommended recurring amount."""
        # Get donor's typical giving
        avg_donation = donor_profile.get("average_donation", 50)
        total_giving = donor_profile.get("total_lifetime_giving", 0)
        donation_count = donor_profile.get("donation_count", 1)
        
        # Calculate monthly capacity (rough estimate)
        if donation_count > 0 and total_giving > 0:
            # Estimate annual giving
            annual_estimate = avg_donation * 12  # Assume monthly capacity
            monthly_capacity = annual_estimate / 12
        else:
            monthly_capacity = 25  # Default
        
        # Suggest amount based on tiers
        if monthly_capacity >= 100:
            suggested = 50
        elif monthly_capacity >= 50:
            suggested = 25
        elif monthly_capacity >= 25:
            suggested = 15
        else:
            suggested = 10
        
        # Provide tier options
        tiers = [
            {"amount": 10, "label": "Supporter"},
            {"amount": 25, "label": "Champion"},
            {"amount": 50, "label": "Hero"},
            {"amount": 100, "label": "Legend"},
        ]
        
        return {
            "recommended_amount": suggested,
            "estimated_monthly_capacity": round(monthly_capacity, 2),
            "tiers": tiers,
        }
    
    async def _tool_project_impact(
        self,
        amount: float,
        frequency: str,
        campaign_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate impact projection for recurring giving."""
        # Calculate totals over time
        monthly_amount = amount
        if frequency == "weekly":
            monthly_amount = amount * 4
        elif frequency == "biweekly":
            monthly_amount = amount * 2
        elif frequency == "quarterly":
            monthly_amount = amount / 3
        elif frequency == "annually":
            monthly_amount = amount / 12
        
        yearly_total = monthly_amount * 12
        three_year_total = yearly_total * 3
        
        # Use LLM to generate impact story
        prompt = f"""Create a compelling impact projection for recurring giving:

Campaign: {campaign_data.get('title', '')}
Category: {campaign_data.get('category', '')}
Monthly Amount: ${monthly_amount:.0f}
Yearly Total: ${yearly_total:.0f}

Create specific, tangible impact statements. Be creative but realistic.
Example: "$25/month = 300 meals for hungry children over a year"

Respond with JSON:
{{
    "monthly_impact": "what one month provides",
    "yearly_impact": "cumulative yearly impact",
    "three_year_impact": "long-term impact vision"
}}"""

        response = await self._llm._provider.complete(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
        )
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                impact = json.loads(json_match.group())
                return {
                    "amount": amount,
                    "frequency": frequency,
                    "monthly_equivalent": monthly_amount,
                    "yearly_total": yearly_total,
                    "three_year_total": three_year_total,
                    **impact,
                }
        except:
            pass
        
        return {
            "amount": amount,
            "frequency": frequency,
            "monthly_equivalent": monthly_amount,
            "yearly_total": yearly_total,
            "monthly_impact": f"${monthly_amount:.0f} in monthly support",
            "yearly_impact": f"${yearly_total:.0f} in annual support",
        }
    
    async def _tool_generate_pitch(
        self,
        opportunity: Dict[str, Any],
        donor_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a compelling recurring giving pitch."""
        donor_profile = donor_profile or {}
        
        prompt = f"""Create a compelling pitch for recurring giving:

Campaign: {opportunity.get('campaign_title', '')}
Recommended Amount: ${opportunity.get('recommended_amount', 25)}/month
Impact: {opportunity.get('impact_per_period', '')}

Donor Interests: {donor_profile.get('cause_affinities', [])}

Write a 2-3 sentence pitch that:
1. Connects with the cause and resonates with supporters
2. Shows the power of sustained giving
3. Makes the commitment feel achievable

Be warm, not pushy. Focus on partnership, not charity."""

        pitch = await self._llm._provider.complete(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
        )
        
        return {
            "pitch": pitch.strip(),
            "opportunity_id": opportunity.get("opportunity_id"),
        }
    
    async def _tool_design_plan(
        self,
        donor_id: str,
        donor_profile: Dict[str, Any],
        budget: Optional[float] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Design a personalized recurring giving plan."""
        preferences = preferences or {}
        
        # Estimate budget if not provided
        if not budget:
            avg_donation = donor_profile.get("average_donation", 50)
            budget = min(avg_donation * 2, 100)  # Cap at $100/month default
        
        # Get suitable opportunities
        suitable_opportunities = [
            opp for opp in self._opportunities.values()
            if opp.suitability in [CampaignSuitability.HIGHLY_SUITABLE, CampaignSuitability.SUITABLE]
        ]
        
        # Match to donor interests
        donor_affinities = {
            a.get("category"): a.get("score", 0)
            for a in donor_profile.get("cause_affinities", [])
        }
        
        scored_opportunities = []
        for opp in suitable_opportunities:
            campaign = self._campaigns.get(opp.campaign_id, {})
            category = campaign.get("category", "")
            affinity_score = donor_affinities.get(category, 0.3)
            total_score = (opp.suitability_score * 0.6) + (affinity_score * 0.4)
            scored_opportunities.append((opp, total_score))
        
        # Sort and select top opportunities
        scored_opportunities.sort(key=lambda x: x[1], reverse=True)
        selected = []
        remaining_budget = budget
        
        for opp, score in scored_opportunities[:5]:
            if remaining_budget >= 10:
                allocation = min(remaining_budget * 0.4, opp.recommended_amount)
                if allocation >= 10:
                    opp.recommended_amount = allocation
                    selected.append(opp)
                    remaining_budget -= allocation
        
        # Use LLM to generate plan summary
        prompt = f"""Create a summary for this recurring giving plan:

Donor Budget: ${budget:.0f}/month
Selected Causes: {[o.campaign_title for o in selected]}
Total Commitment: ${sum(o.recommended_amount for o in selected):.0f}/month

Write a 2-3 sentence summary that:
1. Highlights the diversity of impact
2. Shows the cumulative power of the plan
3. Affirms the donor's generosity"""

        summary = await self._llm._provider.complete(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
        )
        
        plan = RecurringPlan(
            plan_id=f"plan_{donor_id}_{datetime.utcnow().strftime('%Y%m%d')}",
            donor_id=donor_id,
            opportunities=selected,
            total_monthly_commitment=sum(o.recommended_amount for o in selected),
            frequency=RecurringFrequency(preferences.get("frequency", "monthly")),
            preferred_day=preferences.get("preferred_day", 1),
            plan_summary=summary.strip(),
        )
        
        self._donor_plans[donor_id] = plan
        
        return plan.to_dict()
    
    async def _tool_optimize_schedule(
        self,
        donor_id: str,
        commitments: List[Dict[str, Any]],
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Optimize timing of recurring gifts."""
        preferences = preferences or {}
        
        # Get preferred timing
        pay_frequency = preferences.get("pay_frequency", "monthly")
        preferred_day = preferences.get("preferred_day", 1)
        
        optimized = []
        
        if pay_frequency == "biweekly":
            # Split commitments across two pay periods
            for i, commitment in enumerate(commitments):
                day = 1 if i % 2 == 0 else 15
                optimized.append({
                    **commitment,
                    "scheduled_day": day,
                    "reason": f"Aligned with {'first' if day == 1 else 'second'} pay period",
                })
        else:
            # All on preferred day
            for commitment in commitments:
                optimized.append({
                    **commitment,
                    "scheduled_day": preferred_day,
                    "reason": f"Scheduled for day {preferred_day} of each month",
                })
        
        return {
            "donor_id": donor_id,
            "optimized_schedule": optimized,
            "total_monthly": sum(c.get("amount", 0) for c in commitments),
        }
    
    async def _tool_analyze_diversification(
        self,
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze cause diversification in a giving plan."""
        opportunities = plan.get("opportunities", [])
        
        # Categorize by cause
        by_category = defaultdict(float)
        total = 0
        
        for opp in opportunities:
            campaign_id = opp.get("campaign_id")
            campaign = self._campaigns.get(campaign_id, {})
            category = campaign.get("category", "other")
            amount = opp.get("recommended_amount", 0)
            by_category[category] += amount
            total += amount
        
        # Calculate percentages
        distribution = {
            cat: round(amt / total * 100, 1) if total > 0 else 0
            for cat, amt in by_category.items()
        }
        
        # Use LLM to analyze
        prompt = f"""Analyze this giving plan diversification:

Distribution: {json.dumps(distribution)}
Total Monthly: ${total:.0f}

Provide a brief analysis (2-3 sentences) of:
1. How well diversified is this plan?
2. Any recommendations for better balance?
3. The overall impact story"""

        analysis = await self._llm._provider.complete(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.5,
        )
        
        return {
            "distribution": distribution,
            "category_count": len(by_category),
            "total_monthly": total,
            "analysis": analysis.strip(),
        }
    
    async def _tool_create_opportunity(
        self,
        campaign_id: str,
        campaign_title: str,
        suitability_score: float,
        recommended_amount: float,
        suitability: str = "suitable",
        suitability_reasons: List[str] = None,
        impact_per_period: str = "",
        pitch: str = "",
        campaign_url: str = "",
    ) -> Dict[str, Any]:
        """Create a RecurringOpportunity object."""
        try:
            suitability_enum = CampaignSuitability(suitability)
        except:
            if suitability_score >= 0.7:
                suitability_enum = CampaignSuitability.HIGHLY_SUITABLE
            elif suitability_score >= 0.5:
                suitability_enum = CampaignSuitability.SUITABLE
            elif suitability_score >= 0.3:
                suitability_enum = CampaignSuitability.MARGINALLY_SUITABLE
            else:
                suitability_enum = CampaignSuitability.NOT_SUITABLE
        
        opportunity = RecurringOpportunity(
            opportunity_id=f"opp_{campaign_id}",
            campaign_id=campaign_id,
            campaign_title=campaign_title[:100],
            campaign_url=campaign_url,
            suitability=suitability_enum,
            suitability_score=suitability_score,
            suitability_reasons=suitability_reasons or [],
            recommended_frequency=RecurringFrequency.MONTHLY,
            recommended_amount=recommended_amount,
            impact_per_period=impact_per_period,
            pitch=pitch,
        )
        
        self._opportunities[opportunity.opportunity_id] = opportunity
        
        return {
            "opportunity_id": opportunity.opportunity_id,
            "status": "created",
            "campaign_title": campaign_title,
            "suitability_score": suitability_score,
        }
    
    # ==================== Public API ====================
    
    def register_campaign(self, campaign: Dict[str, Any]) -> None:
        """Register a campaign for recurring curation."""
        campaign_id = campaign.get("campaign_id", str(hash(campaign.get("title", ""))))
        self._campaigns[campaign_id] = campaign
    
    async def curate_opportunities(
        self,
        campaigns: Optional[List[Dict[str, Any]]] = None
    ) -> List[RecurringOpportunity]:
        """
        Autonomously curate recurring giving opportunities using planning.
        
        The agent will autonomously:
        1. Assess each campaign for recurring suitability
        2. Generate impact projections
        3. Create compelling pitches
        4. Rank opportunities
        """
        if campaigns:
            for campaign in campaigns:
                self.register_campaign(campaign)
        
        if not self._campaigns:
            self._logger.warning("no_campaigns_registered")
            return []
        
        self._logger.info("curating_opportunities", campaign_count=len(self._campaigns))
        
        # Use autonomous planning for curation
        campaign_summaries = []
        for campaign_id, campaign in self._campaigns.items():
            campaign_summaries.append({
                "campaign_id": campaign_id,
                "title": campaign.get("title", "")[:50],
                "category": campaign.get("category", ""),
                "goal": campaign.get("goal_amount", 0),
                "raised": campaign.get("raised_amount", 0),
                "donors": campaign.get("donor_count", 0),
            })
        
        goal = f"""Curate recurring giving opportunities from {len(self._campaigns)} registered campaigns.

Campaigns to analyze:
{json.dumps(campaign_summaries, indent=2, default=str)[:1000]}

For each campaign:
1. Assess if it's suitable for recurring giving (ongoing need, organizational support, etc.)
2. If suitable (score >= 0.3), calculate recommended monthly amount
3. Project the impact of recurring giving
4. Generate a compelling pitch for recurring support
5. Create a RecurringOpportunity object

Focus on campaigns that show:
- Ongoing or sustained needs
- Regular updates/engagement
- Organizational backing
- Clear impact potential

Skip campaigns that are clearly one-time needs or already fully funded."""
        
        context = {
            "campaigns": list(self._campaigns.values()),
            "campaign_ids": list(self._campaigns.keys()),
        }
        
        # Run autonomous curation
        result = await self.run(goal, context, max_iterations=20)
        
        # Return opportunities from cache (created by tools)
        opportunities = list(self._opportunities.values())
        
        # If autonomous run didn't create opportunities, fall back to direct execution
        if not opportunities:
            self._logger.warning("autonomous_curation_did_not_create_opportunities", falling_back_to_direct=True)
            # Fallback to direct execution
            for campaign_id, campaign in self._campaigns.items():
                try:
                    # Assess suitability
                    assessment = await self._tool_assess_suitability(campaign_id, campaign)
                    
                    self._logger.debug(
                        "campaign_assessed",
                        campaign_id=campaign_id,
                        score=assessment.get("suitability_score", 0),
                    )
                    
                    if assessment.get("suitability_score", 0) >= 0.3:
                        # Calculate recommended amount (using default profile)
                        amount_result = await self._tool_calculate_amount(
                            {"average_donation": 50}, campaign
                        )
                        
                        # Project impact
                        impact = await self._tool_project_impact(
                            amount_result.get("recommended_amount", 25),
                            "monthly",
                            campaign
                        )
                        
                        # Get title, fallback to URL if title is missing or is a URL
                        title = campaign.get("title", "")
                        if not title or title.startswith("http") or "/f/" in title:
                            # Title is missing or is actually a URL, try to extract from URL or use a default
                            url = campaign.get("url", "")
                            if url:
                                # Try to extract campaign name from URL
                                import re
                                match = re.search(r'/f/([^/?]+)', url)
                                if match:
                                    title = match.group(1).replace('-', ' ').title()
                                else:
                                    title = "Campaign"
                            else:
                                title = "Campaign"
                        
                        opportunity = RecurringOpportunity(
                            opportunity_id=f"opp_{campaign_id}",
                            campaign_id=campaign_id,
                            campaign_title=title[:100],  # Limit length
                            campaign_url=campaign.get("url", ""),
                            suitability=CampaignSuitability(assessment.get("suitability", "suitable")),
                            suitability_score=assessment.get("suitability_score", 0.5),
                            suitability_reasons=assessment.get("reasons", []),
                            recommended_frequency=RecurringFrequency.MONTHLY,
                            recommended_amount=amount_result.get("recommended_amount", 25),
                            impact_per_period=impact.get("monthly_impact", ""),
                        )
                        
                        # Generate pitch
                        try:
                            pitch_result = await self._tool_generate_pitch(opportunity.to_dict())
                            opportunity.pitch = pitch_result.get("pitch", "")
                        except Exception as e:
                            self._logger.warning("pitch_generation_failed", error=str(e))
                            opportunity.pitch = f"Support {campaign.get('title', 'this campaign')} with monthly giving."
                        
                        self._opportunities[opportunity.opportunity_id] = opportunity
                        opportunities.append(opportunity)
                    else:
                        self._logger.debug(
                            "campaign_below_threshold",
                            campaign_id=campaign_id,
                            score=assessment.get("suitability_score", 0),
                        )
                except Exception as e:
                    self._logger.error(
                        "opportunity_creation_failed",
                        campaign_id=campaign_id,
                        error=str(e),
                    )
                    # Continue with next campaign
                    continue
        
        # Sort by suitability
        opportunities.sort(key=lambda x: x.suitability_score, reverse=True)
        
        self._logger.info("opportunities_curated", count=len(opportunities))
        
        return opportunities
    
    async def create_donor_plan(
        self,
        donor_id: str,
        donor_profile: Dict[str, Any],
        monthly_budget: Optional[float] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> RecurringPlan:
        """
        Create a personalized recurring giving plan for a donor.
        """
        goal = f"Design optimal recurring giving plan for donor {donor_id}"
        
        context = {
            "donor_id": donor_id,
            "donor_profile": donor_profile,
            "budget": monthly_budget,
            "preferences": preferences,
        }
        
        # Run autonomous planning
        await self.run(goal, context)
        
        return self._donor_plans.get(donor_id, RecurringPlan(
            plan_id=f"plan_{donor_id}",
            donor_id=donor_id,
        ))
    
    async def get_opportunity(self, opportunity_id: str) -> Optional[RecurringOpportunity]:
        """Get a specific recurring opportunity."""
        return self._opportunities.get(opportunity_id)
    
    async def get_donor_plan(self, donor_id: str) -> Optional[RecurringPlan]:
        """Get a donor's recurring plan."""
        return self._donor_plans.get(donor_id)
