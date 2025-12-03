"""
Agent 1: Donor Affinity Profiler (Autonomous Agent)

An LLM-based autonomous agent that builds giving identities for donors.
Uses planning, reasoning, tool-use, and long-term task decomposition to:
- Analyze donation patterns and behaviors
- Identify causes that inspire donors
- Build comprehensive giving personas
- Predict future giving behavior
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from collections import defaultdict
import math
import json

import structlog

from src.core.autonomous_agent import AutonomousAgent, Tool, Task, ReasoningStep, ReasoningType
from src.core.llm_service import get_llm_service

logger = structlog.get_logger()


class CauseCategory(str, Enum):
    """Primary cause categories for donor affinities."""
    MEDICAL = "medical"
    EDUCATION = "education"
    DISASTER_RELIEF = "disaster_relief"
    ANIMALS = "animals"
    COMMUNITY = "community"
    ENVIRONMENT = "environment"
    ARTS_CULTURE = "arts_culture"
    SPORTS = "sports"
    MEMORIAL = "memorial"
    EMERGENCY = "emergency"
    CHILDREN = "children"
    VETERANS = "veterans"
    HOUSING = "housing"
    FOOD_SECURITY = "food_security"


class GivingMotivator(str, Enum):
    """Inspirational causes and motivators that drive giving."""
    PERSONAL_CONNECTION = "personal_connection"
    GEOGRAPHIC_PROXIMITY = "geographic_proximity"
    COMPELLING_STORY = "compelling_story"
    TIME_SENSITIVITY = "time_sensitivity"
    MATCHING_OPPORTUNITY = "matching_opportunity"
    COMMUNITY_MOMENTUM = "community_momentum"
    MILESTONE_CELEBRATION = "milestone_celebration"
    ANNIVERSARY_GIVING = "anniversary_giving"
    SEASONAL_GENEROSITY = "seasonal_generosity"
    TAX_BENEFIT_AWARENESS = "tax_benefit_awareness"


class GivingPattern(str, Enum):
    """Patterns of giving behavior."""
    IMPULSE_GIVER = "impulse_giver"
    PLANNED_GIVER = "planned_giver"
    RECURRING_GIVER = "recurring_giver"
    SEASONAL_GIVER = "seasonal_giver"
    MAJOR_GIFT_GIVER = "major_gift_giver"
    MICRO_GIVER = "micro_giver"
    SOCIAL_GIVER = "social_giver"
    ANONYMOUS_GIVER = "anonymous_giver"


@dataclass
class CauseAffinity:
    """Represents affinity for a specific cause."""
    category: CauseCategory
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    donation_count: int = 0
    total_amount: float = 0.0
    last_donation: Optional[datetime] = None
    subcategories: List[str] = field(default_factory=list)
    llm_insights: str = ""  # LLM-generated insights about this affinity


@dataclass
class DonorProfile:
    """Complete giving persona for a donor."""
    donor_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Cause affinities ranked by score
    cause_affinities: List[CauseAffinity] = field(default_factory=list)
    
    # Inspirational causes and motivators
    giving_motivators: Dict[GivingMotivator, float] = field(default_factory=dict)
    
    # Giving patterns
    primary_pattern: Optional[GivingPattern] = None
    secondary_patterns: List[GivingPattern] = field(default_factory=list)
    
    # Financial profile
    average_donation: float = 0.0
    median_donation: float = 0.0
    total_lifetime_giving: float = 0.0
    donation_count: int = 0
    
    # Timing patterns
    preferred_day_of_week: Optional[int] = None
    preferred_time_of_day: Optional[str] = None
    giving_frequency_days: float = 0.0
    
    # Engagement metrics
    engagement_score: float = 0.0
    recency_score: float = 0.0
    loyalty_score: float = 0.0
    
    # LLM-generated insights
    personality_summary: str = ""
    giving_philosophy: str = ""
    predicted_interests: List[str] = field(default_factory=list)
    engagement_recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    profile_completeness: float = 0.0
    last_activity: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "donor_id": self.donor_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "cause_affinities": [
                {
                    "category": ca.category.value,
                    "score": ca.score,
                    "confidence": ca.confidence,
                    "donation_count": ca.donation_count,
                    "total_amount": ca.total_amount,
                    "subcategories": ca.subcategories,
                    "llm_insights": ca.llm_insights,
                }
                for ca in self.cause_affinities
            ],
            "giving_motivators": {k.value: v for k, v in self.giving_motivators.items()},
            "primary_pattern": self.primary_pattern.value if self.primary_pattern else None,
            "secondary_patterns": [p.value for p in self.secondary_patterns],
            "average_donation": self.average_donation,
            "median_donation": self.median_donation,
            "total_lifetime_giving": self.total_lifetime_giving,
            "donation_count": self.donation_count,
            "preferred_day_of_week": self.preferred_day_of_week,
            "preferred_time_of_day": self.preferred_time_of_day,
            "giving_frequency_days": self.giving_frequency_days,
            "engagement_score": self.engagement_score,
            "recency_score": self.recency_score,
            "loyalty_score": self.loyalty_score,
            "personality_summary": self.personality_summary,
            "giving_philosophy": self.giving_philosophy,
            "predicted_interests": self.predicted_interests,
            "engagement_recommendations": self.engagement_recommendations,
            "profile_completeness": self.profile_completeness,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
        }


class DonorAffinityProfiler(AutonomousAgent[DonorProfile]):
    """
    Autonomous agent that builds rich donor profiles through reasoning and analysis.
    
    This agent:
    1. Plans how to analyze a donor's giving history
    2. Reasons about patterns and motivations
    3. Uses tools to compute metrics and generate insights
    4. Synthesizes findings into a comprehensive profile
    5. Reflects on confidence and suggests improvements
    """
    
    SYSTEM_PROMPT = """You are an expert donor behavior analyst with deep expertise in:
- Philanthropic psychology and giving motivations
- Pattern recognition in charitable behavior
- Donor segmentation and persona development
- Predictive modeling for donor engagement

Your goal is to understand WHY donors give, not just WHAT they give to. You look for:
- Underlying values and beliefs that drive giving
- Life circumstances that influence donation patterns
- Social and community connections to causes
- Inspirational stories and causes that resonate

You are thorough, analytical, and empathetic. You understand that behind every donation
is a human being with their own story, values, and motivations."""

    def __init__(self):
        super().__init__(
            agent_id="donor_affinity_profiler",
            name="Donor Affinity Profiler",
            description="Autonomous agent that builds comprehensive donor giving personas",
            system_prompt=self.SYSTEM_PROMPT,
        )
        
        # Profile cache
        self._profile_cache: Dict[str, DonorProfile] = {}
        
        # Category keywords for initial classification
        self._category_keywords = self._build_category_keywords()
        
        # Register domain-specific tools
        for tool in self._get_domain_tools():
            self.register_tool(tool)
        
        # Register A2A message handlers
        self.register_handler("get_donor_insights", self._handle_get_donor_insights)
        self.register_handler("build_profile", self._handle_build_profile)
    
    def _get_domain_system_prompt(self) -> str:
        """Get domain-specific system prompt additions."""
        return """
## Domain Expertise: Donor Profiling

You specialize in understanding donor behavior through:

1. **Quantitative Analysis**: Computing metrics like frequency, recency, monetary value
2. **Qualitative Analysis**: Understanding the stories and causes behind donations
3. **Pattern Recognition**: Identifying giving patterns (seasonal, recurring, impulse)
4. **Motivator Identification**: Understanding what inspires donors to give
5. **Predictive Insights**: Forecasting future giving behavior

When analyzing a donor:
- Look beyond the numbers to understand the person
- Consider life events that might influence giving
- Identify causes that create resonance and connection
- Recognize social influences on giving behavior
"""
    
    def _get_domain_tools(self) -> List[Tool]:
        """Get domain-specific tools for donor profiling."""
        return [
            Tool(
                name="analyze_donation_history",
                description="Analyze a donor's complete donation history to extract patterns and metrics",
                parameters={
                    "type": "object",
                    "properties": {
                        "donor_id": {"type": "string", "description": "The donor's ID"},
                        "donations": {"type": "array", "description": "List of donation records"},
                    },
                    "required": ["donor_id", "donations"],
                },
                function=self._tool_analyze_history,
            ),
            Tool(
                name="identify_cause_affinities",
                description="Identify which causes the donor is most drawn to and why",
                parameters={
                    "type": "object",
                    "properties": {
                        "donations": {"type": "array", "description": "Donation records to analyze"},
                    },
                    "required": ["donations"],
                },
                function=self._tool_identify_affinities,
            ),
            Tool(
                name="detect_giving_motivators",
                description="Detect what inspires and motivates the donor to give",
                parameters={
                    "type": "object",
                    "properties": {
                        "donations": {"type": "array", "description": "Donation records"},
                        "metadata": {"type": "object", "description": "Donor metadata"},
                    },
                    "required": ["donations"],
                },
                function=self._tool_detect_motivators,
            ),
            Tool(
                name="classify_giving_pattern",
                description="Classify the donor's giving pattern (recurring, impulse, seasonal, etc.)",
                parameters={
                    "type": "object",
                    "properties": {
                        "donations": {"type": "array", "description": "Donation records"},
                    },
                    "required": ["donations"],
                },
                function=self._tool_classify_pattern,
            ),
            Tool(
                name="calculate_engagement_scores",
                description="Calculate engagement, recency, and loyalty scores",
                parameters={
                    "type": "object",
                    "properties": {
                        "donations": {"type": "array", "description": "Donation records"},
                    },
                    "required": ["donations"],
                },
                function=self._tool_calculate_scores,
            ),
            Tool(
                name="generate_donor_insights",
                description="Use LLM to generate deep insights about the donor's giving personality",
                parameters={
                    "type": "object",
                    "properties": {
                        "profile_data": {"type": "object", "description": "Collected profile data so far"},
                        "donations": {"type": "array", "description": "Donation records"},
                    },
                    "required": ["profile_data", "donations"],
                },
                function=self._tool_generate_insights,
            ),
            Tool(
                name="predict_future_interests",
                description="Predict what causes and campaigns might interest the donor in the future",
                parameters={
                    "type": "object",
                    "properties": {
                        "profile": {"type": "object", "description": "Current donor profile"},
                    },
                    "required": ["profile"],
                },
                function=self._tool_predict_interests,
            ),
            Tool(
                name="save_profile",
                description="Save the completed donor profile. If profile is not provided, it will be constructed from previous tool results.",
                parameters={
                    "type": "object",
                    "properties": {
                        "profile": {"type": "object", "description": "Complete donor profile (optional - will be constructed from context if not provided)"},
                    },
                    "required": [],
                },
                function=self._tool_save_profile,
            ),
        ]
    
    def _build_category_keywords(self) -> Dict[CauseCategory, List[str]]:
        """Build keyword mappings for cause categories."""
        return {
            CauseCategory.MEDICAL: [
                "cancer", "surgery", "treatment", "hospital", "medical", "health",
                "illness", "disease", "therapy", "medication", "transplant", "chemo",
                "diagnosis", "recovery", "rehabilitation", "wheelchair", "prosthetic"
            ],
            CauseCategory.EDUCATION: [
                "school", "college", "university", "tuition", "scholarship", "education",
                "student", "learning", "books", "supplies", "degree", "graduation"
            ],
            CauseCategory.DISASTER_RELIEF: [
                "hurricane", "earthquake", "flood", "fire", "tornado", "disaster",
                "relief", "emergency", "rebuild", "evacuate", "storm", "wildfire"
            ],
            CauseCategory.ANIMALS: [
                "pet", "dog", "cat", "animal", "rescue", "shelter", "veterinary",
                "wildlife", "adoption", "spay", "neuter", "sanctuary"
            ],
            CauseCategory.COMMUNITY: [
                "community", "neighborhood", "local", "town", "city", "park",
                "library", "center", "program", "youth", "senior"
            ],
            CauseCategory.CHILDREN: [
                "child", "children", "kid", "baby", "infant", "pediatric", "orphan",
                "foster", "adoption", "youth", "playground", "daycare"
            ],
            CauseCategory.VETERANS: [
                "veteran", "military", "soldier", "army", "navy", "marine",
                "service", "deployment", "ptsd", "wounded", "hero"
            ],
            CauseCategory.MEMORIAL: [
                "memorial", "funeral", "burial", "tribute", "remembrance",
                "celebration of life", "passed away", "memory of"
            ],
            CauseCategory.EMERGENCY: [
                "urgent", "emergency", "critical", "immediate", "desperate",
                "life-saving", "time-sensitive", "crisis"
            ],
        }
    
    # ==================== Tool Implementations ====================
    
    async def _tool_analyze_history(
        self,
        donor_id: str,
        donations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze donation history for patterns and metrics."""
        if not donations:
            return {"error": "No donations to analyze", "metrics": {}}
        
        amounts = [d.get("amount", 0) for d in donations]
        
        # Calculate basic metrics
        metrics = {
            "total_donations": len(donations),
            "total_amount": sum(amounts),
            "average_amount": sum(amounts) / len(amounts),
            "median_amount": self._calculate_median(amounts),
            "min_amount": min(amounts),
            "max_amount": max(amounts),
        }
        
        # Analyze timing
        timestamps = []
        for d in donations:
            ts = d.get("timestamp")
            if ts:
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                timestamps.append(ts)
        
        if timestamps:
            timestamps.sort()
            metrics["first_donation"] = timestamps[0].isoformat()
            metrics["last_donation"] = timestamps[-1].isoformat()
            metrics["giving_span_days"] = (timestamps[-1] - timestamps[0]).days
            
            if len(timestamps) > 1:
                intervals = [(timestamps[i+1] - timestamps[i]).days for i in range(len(timestamps)-1)]
                metrics["avg_days_between_donations"] = sum(intervals) / len(intervals)
        
        # Analyze sources
        sources = [d.get("source", "direct") for d in donations]
        metrics["source_distribution"] = dict(defaultdict(int, {s: sources.count(s) for s in set(sources)}))
        
        # Analyze recurring vs one-time
        recurring = sum(1 for d in donations if d.get("is_recurring"))
        metrics["recurring_percentage"] = recurring / len(donations) * 100
        
        # Analyze anonymity
        anonymous = sum(1 for d in donations if d.get("is_anonymous"))
        metrics["anonymous_percentage"] = anonymous / len(donations) * 100
        
        return {
            "donor_id": donor_id,
            "metrics": metrics,
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }
    
    async def _tool_identify_affinities(
        self,
        donations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Identify cause affinities from donations."""
        category_stats = defaultdict(lambda: {
            "count": 0, "amount": 0, "campaigns": [], "last_donation": None
        })
        
        for donation in donations:
            category = self._categorize_donation(donation)
            if category:
                stats = category_stats[category]
                stats["count"] += 1
                stats["amount"] += donation.get("amount", 0)
                stats["campaigns"].append(donation.get("campaign_title", "")[:50])
                
                ts = donation.get("timestamp")
                if ts:
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if not stats["last_donation"] or ts > stats["last_donation"]:
                        stats["last_donation"] = ts
        
        # Calculate scores
        total_donations = len(donations)
        total_amount = sum(d.get("amount", 0) for d in donations)
        
        affinities = []
        for category, stats in category_stats.items():
            frequency_score = stats["count"] / total_donations if total_donations > 0 else 0
            amount_score = stats["amount"] / total_amount if total_amount > 0 else 0
            score = (frequency_score * 0.6) + (amount_score * 0.4)
            confidence = min(1.0, stats["count"] / 5)
            
            affinities.append({
                "category": category.value,
                "score": round(score, 3),
                "confidence": round(confidence, 3),
                "donation_count": stats["count"],
                "total_amount": stats["amount"],
                "sample_campaigns": stats["campaigns"][:3],
            })
        
        # Sort by score
        affinities.sort(key=lambda x: x["score"], reverse=True)
        
        # Use LLM to add insights for top affinities
        if affinities:
            top_affinity = affinities[0]
            insight_prompt = f"""Based on this donor's top cause affinity:
Category: {top_affinity['category']}
Donation count: {top_affinity['donation_count']}
Sample campaigns: {top_affinity['sample_campaigns']}

In 1-2 sentences, what does this suggest about the donor's values and motivations?"""
            
            insight = await self._llm._provider.complete(
                prompt=insight_prompt,
                system_prompt="You are an expert in donor psychology. Be concise and insightful.",
                temperature=0.7,
            )
            affinities[0]["llm_insight"] = insight
        
        return {
            "affinities": affinities,
            "dominant_category": affinities[0]["category"] if affinities else None,
        }
    
    async def _tool_detect_motivators(
        self,
        donations: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Detect giving motivators."""
        metadata = metadata or {}
        motivators = {}
        
        if not donations:
            return {"motivators": {}}
        
        sources = [d.get("source", "").lower() for d in donations]
        
        # Community momentum (social giving)
        shared_count = sum(1 for s in sources if "shared" in s or "social" in s)
        if shared_count > 0:
            motivators["community_momentum"] = min(1.0, shared_count / len(donations))
        
        # Geographic proximity
        donor_location = metadata.get("location", "")
        if donor_location:
            local_count = sum(
                1 for d in donations
                if donor_location.lower() in d.get("campaign_location", "").lower()
            )
            if local_count > 0:
                motivators["geographic_proximity"] = min(1.0, local_count / len(donations))
        
        # Time sensitivity
        urgent_keywords = ["urgent", "emergency", "critical", "immediate"]
        urgent_count = sum(
            1 for d in donations
            if any(kw in d.get("campaign_title", "").lower() for kw in urgent_keywords)
        )
        if urgent_count > 0:
            motivators["time_sensitivity"] = min(1.0, urgent_count / len(donations))
        
        # Personal connection
        personal_keywords = ["friend", "family", "colleague", "neighbor", "classmate"]
        personal_count = sum(
            1 for d in donations
            if any(kw in d.get("source", "").lower() for kw in personal_keywords)
        )
        if personal_count > 0:
            motivators["personal_connection"] = min(1.0, personal_count / len(donations))
        
        # Analyze timing for seasonal patterns
        months = []
        for d in donations:
            ts = d.get("timestamp")
            if ts:
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                months.append(ts.month)
        
        if months:
            # Seasonal generosity (November-December)
            holiday_count = sum(1 for m in months if m in [11, 12])
            if holiday_count > len(months) * 0.3:
                motivators["seasonal_generosity"] = min(1.0, holiday_count / len(months))
            
            # Tax benefit awareness (March-April)
            tax_count = sum(1 for m in months if m in [3, 4])
            if tax_count > len(months) * 0.2:
                motivators["tax_benefit_awareness"] = min(1.0, tax_count / len(months))
        
        # Use LLM to identify additional motivators
        if donations:
            sample_donations = donations[:5]
            llm_prompt = f"""Analyze these donations and identify what might motivate this donor:

{json.dumps(sample_donations, indent=2, default=str)[:1500]}

What deeper motivations might be driving this giving behavior? 
Consider: personal experiences, values, social influences, life stage.
Respond with a brief JSON: {{"additional_motivators": ["motivator1", "motivator2"], "reasoning": "brief explanation"}}"""
            
            try:
                llm_response = await self._llm._provider.complete(
                    prompt=llm_prompt,
                    system_prompt="You are an expert in philanthropic psychology.",
                    temperature=0.5,
                )
                import re
                json_match = re.search(r'\{[\s\S]*\}', llm_response)
                if json_match:
                    llm_insights = json.loads(json_match.group())
                    motivators["_llm_insights"] = llm_insights
            except:
                pass
        
        return {"motivators": motivators}
    
    async def _tool_classify_pattern(
        self,
        donations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Classify giving patterns."""
        patterns = []
        
        if not donations:
            return {"patterns": [], "primary": None}
        
        amounts = [d.get("amount", 0) for d in donations]
        avg_amount = sum(amounts) / len(amounts)
        
        # Check for recurring giver
        recurring_count = sum(1 for d in donations if d.get("is_recurring"))
        if recurring_count > len(donations) * 0.3:
            patterns.append(("recurring_giver", recurring_count / len(donations)))
        
        # Check for anonymous giver
        anonymous_count = sum(1 for d in donations if d.get("is_anonymous"))
        if anonymous_count > len(donations) * 0.5:
            patterns.append(("anonymous_giver", anonymous_count / len(donations)))
        
        # Check for major gift giver
        major_count = sum(1 for a in amounts if a > 500)
        if major_count > 0 and avg_amount > 200:
            patterns.append(("major_gift_giver", major_count / len(donations)))
        
        # Check for micro giver
        micro_count = sum(1 for a in amounts if a < 25)
        if micro_count > len(donations) * 0.7:
            patterns.append(("micro_giver", micro_count / len(donations)))
        
        # Check for social giver
        social_count = sum(1 for d in donations if "shared" in d.get("source", "").lower())
        if social_count > len(donations) * 0.4:
            patterns.append(("social_giver", social_count / len(donations)))
        
        # Analyze timing patterns
        timestamps = []
        for d in donations:
            ts = d.get("timestamp")
            if ts:
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                timestamps.append(ts)
        
        if len(timestamps) >= 3:
            timestamps.sort()
            intervals = [(timestamps[i+1] - timestamps[i]).days for i in range(len(timestamps)-1)]
            avg_interval = sum(intervals) / len(intervals)
            std_interval = (sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)) ** 0.5
            
            if avg_interval > 0:
                cv = std_interval / avg_interval  # Coefficient of variation
                if cv < 0.3:
                    patterns.append(("planned_giver", 1 - cv))
                elif cv > 0.7:
                    patterns.append(("impulse_giver", cv))
        
        # Check for seasonal giver
        if timestamps:
            months = [ts.month for ts in timestamps]
            month_counts = defaultdict(int)
            for m in months:
                month_counts[m] += 1
            
            for start_month in range(1, 13):
                quarter_months = [(start_month + i - 1) % 12 + 1 for i in range(3)]
                quarter_count = sum(month_counts[m] for m in quarter_months)
                if quarter_count > len(months) * 0.6:
                    patterns.append(("seasonal_giver", quarter_count / len(months)))
                    break
        
        # Sort by score
        patterns.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "patterns": [{"pattern": p[0], "score": round(p[1], 3)} for p in patterns],
            "primary": patterns[0][0] if patterns else "general_giver",
        }
    
    async def _tool_calculate_scores(
        self,
        donations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate engagement, recency, and loyalty scores."""
        if not donations:
            return {"engagement": 0, "recency": 0, "loyalty": 0}
        
        # Parse timestamps
        timestamps = []
        for d in donations:
            ts = d.get("timestamp")
            if ts:
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                timestamps.append(ts)
        
        # Recency score
        recency_score = 0.0
        if timestamps:
            last_donation = max(timestamps)
            days_since = (datetime.utcnow() - last_donation.replace(tzinfo=None)).days
            recency_score = round(math.exp(-days_since / 180), 3)
        
        # Loyalty score
        loyalty_score = 0.0
        if timestamps:
            timestamps.sort()
            tenure_days = (datetime.utcnow() - timestamps[0].replace(tzinfo=None)).days
            tenure_score = min(1.0, tenure_days / 730)
            repeat_score = min(1.0, len(donations) / 10)
            loyalty_score = round((tenure_score * 0.5 + repeat_score * 0.5), 3)
        
        # Engagement score
        frequency_score = min(1.0, len(donations) / 12)
        
        consistency_score = 0.5
        if len(timestamps) >= 3:
            timestamps.sort()
            intervals = [(timestamps[i+1] - timestamps[i]).days for i in range(len(timestamps)-1)]
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                if avg_interval > 0:
                    variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)
                    cv = (variance ** 0.5) / avg_interval
                    consistency_score = max(0, 1 - cv)
        
        engagement_score = round(
            frequency_score * 0.3 + recency_score * 0.4 + consistency_score * 0.3,
            3
        )
        
        return {
            "engagement_score": engagement_score,
            "recency_score": recency_score,
            "loyalty_score": loyalty_score,
            "components": {
                "frequency": round(frequency_score, 3),
                "consistency": round(consistency_score, 3),
            }
        }
    
    async def _tool_generate_insights(
        self,
        profile_data: Dict[str, Any],
        donations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate deep LLM insights about the donor."""
        # Prepare context for LLM
        donation_summary = []
        for d in donations[:10]:
            donation_summary.append({
                "campaign": d.get("campaign_title", "")[:50],
                "category": d.get("campaign_category", ""),
                "amount": d.get("amount", 0),
                "source": d.get("source", ""),
            })
        
        prompt = f"""Based on this donor's profile and donation history, provide deep insights:

Profile Data:
{json.dumps(profile_data, indent=2, default=str)[:1500]}

Recent Donations:
{json.dumps(donation_summary, indent=2)}

Please provide:
1. A 2-3 sentence personality summary describing this donor's giving identity
2. A brief statement of their apparent giving philosophy
3. 3-5 specific recommendations for engaging this donor effectively

Respond in JSON format:
{{
    "personality_summary": "...",
    "giving_philosophy": "...",
    "engagement_recommendations": ["rec1", "rec2", "rec3"]
}}"""

        response = await self._llm._provider.complete(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
        )
        
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "personality_summary": "Analysis in progress",
            "giving_philosophy": "",
            "engagement_recommendations": [],
        }
    
    async def _tool_predict_interests(
        self,
        profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict future interests using LLM."""
        prompt = f"""Based on this donor profile, predict what causes and campaigns they might be interested in:

Profile:
{json.dumps(profile, indent=2, default=str)[:2000]}

Consider:
- Their demonstrated cause affinities
- Their giving motivators
- Their giving patterns
- Logical extensions of their interests

Provide 5-7 specific predictions about what might interest them, with brief reasoning.
Respond in JSON: {{"predictions": [{{"interest": "...", "reasoning": "...", "confidence": 0.0-1.0}}]}}"""

        response = await self._llm._provider.complete(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.6,
        )
        
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {"predictions": []}
    
    async def _tool_save_profile(self, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Save the donor profile. If profile is not provided, construct it from context/task results."""
        # If profile not provided, construct it from context and task results
        if not profile:
            donor_id = self._memory.current_context.get("donor_id", "unknown")
            
            # Gather results from completed tasks
            profile = {
                "donor_id": donor_id,
                "total_lifetime_giving": 0,
                "donation_count": 0,
                "average_donation": 0,
                "engagement_score": 0,
                "recency_score": 0,
                "loyalty_score": 0,
                "affinities": [],
                "motivators": {},
                "patterns": [],
                "primary_pattern": None,
                "personality_summary": "",
                "giving_philosophy": "",
                "predictions": [],
                "engagement_recommendations": [],
            }
            
            # Extract data from completed tasks
            for task in self._memory.tasks.values():
                if task.status.value == "completed" and task.result:
                    result = task.result
                    
                    # From analyze_history
                    if "metrics" in result:
                        metrics = result["metrics"]
                        profile["total_lifetime_giving"] = metrics.get("total_amount", 0)
                        profile["donation_count"] = metrics.get("total_donations", 0)
                        profile["average_donation"] = metrics.get("average_amount", 0)
                    
                    # From identify_affinities
                    if "affinities" in result:
                        profile["affinities"] = result["affinities"]
                    
                    # From identify_motivators
                    if "motivators" in result:
                        profile["motivators"] = result["motivators"]
                    
                    # From classify_pattern
                    if "patterns" in result:
                        profile["patterns"] = result["patterns"]
                    if "primary" in result:
                        profile["primary_pattern"] = result["primary"]
                    
                    # From calculate_scores
                    if "engagement" in result:
                        profile["engagement_score"] = result.get("engagement", 0)
                        profile["recency_score"] = result.get("recency", 0)
                        profile["loyalty_score"] = result.get("loyalty", 0)
                    
                    # From generate_insights
                    if "personality_summary" in result:
                        profile["personality_summary"] = result.get("personality_summary", "")
                    if "giving_philosophy" in result:
                        profile["giving_philosophy"] = result.get("giving_philosophy", "")
                    if "recommendations" in result:
                        profile["engagement_recommendations"] = result.get("recommendations", [])
                    
                    # From predict_interests
                    if "predictions" in result:
                        profile["predictions"] = result["predictions"]
            
            # Also check current_context for donations if needed
            donations = self._memory.current_context.get("donations", [])
            if donations and profile["donation_count"] == 0:
                # Fallback: calculate basic metrics from donations
                profile["total_lifetime_giving"] = sum(d.get("amount", 0) for d in donations)
                profile["donation_count"] = len(donations)
                profile["average_donation"] = profile["total_lifetime_giving"] / len(donations) if donations else 0
        
        donor_id = profile.get("donor_id", "unknown")
        
        # Convert to DonorProfile object
        donor_profile = DonorProfile(donor_id=donor_id)
        
        # Populate from profile dict
        donor_profile.total_lifetime_giving = profile.get("total_lifetime_giving", 0)
        donor_profile.donation_count = profile.get("donation_count", 0)
        donor_profile.average_donation = profile.get("average_donation", 0)
        donor_profile.engagement_score = profile.get("engagement_score", 0)
        donor_profile.recency_score = profile.get("recency_score", 0)
        donor_profile.loyalty_score = profile.get("loyalty_score", 0)
        donor_profile.personality_summary = profile.get("personality_summary", "")
        donor_profile.giving_philosophy = profile.get("giving_philosophy", "")
        donor_profile.engagement_recommendations = profile.get("engagement_recommendations", [])
        donor_profile.predicted_interests = [
            p.get("interest", "") for p in profile.get("predictions", [])
        ]
        
        # Handle affinities
        for aff in profile.get("affinities", []):
            try:
                category = CauseCategory(aff.get("category", "community"))
                donor_profile.cause_affinities.append(CauseAffinity(
                    category=category,
                    score=aff.get("score", 0),
                    confidence=aff.get("confidence", 0),
                    donation_count=aff.get("donation_count", 0),
                    total_amount=aff.get("total_amount", 0),
                    llm_insights=aff.get("llm_insight", ""),
                ))
            except:
                pass
        
        # Handle motivators
        for key, value in profile.get("motivators", {}).items():
            if not key.startswith("_"):
                try:
                    motivator = GivingMotivator(key)
                    donor_profile.giving_motivators[motivator] = value
                except:
                    pass
        
        # Handle patterns
        if profile.get("primary_pattern"):
            try:
                donor_profile.primary_pattern = GivingPattern(profile["primary_pattern"])
            except:
                pass
        
        donor_profile.profile_completeness = self._calculate_completeness(donor_profile)
        donor_profile.updated_at = datetime.utcnow()
        
        # Cache the profile
        self._profile_cache[donor_id] = donor_profile
        
        return {
            "status": "saved",
            "donor_id": donor_id,
            "completeness": donor_profile.profile_completeness,
        }
    
    # ==================== Helper Methods ====================
    
    def _categorize_donation(self, donation: Dict[str, Any]) -> Optional[CauseCategory]:
        """Categorize a donation based on campaign info."""
        explicit_category = donation.get("campaign_category", "").lower()
        for category in CauseCategory:
            if category.value in explicit_category:
                return category
        
        text = f"{donation.get('campaign_title', '')} {donation.get('campaign_description', '')}".lower()
        
        category_scores = {}
        for category, keywords in self._category_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return CauseCategory.COMMUNITY
    
    @staticmethod
    def _calculate_median(values: List[float]) -> float:
        """Calculate median of a list."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_values[mid - 1] + sorted_values[mid]) / 2
        return sorted_values[mid]
    
    def _calculate_completeness(self, profile: DonorProfile) -> float:
        """Calculate profile completeness."""
        scores = [
            1.0 if profile.cause_affinities else 0.0,
            1.0 if profile.giving_motivators else 0.0,
            1.0 if profile.primary_pattern else 0.0,
            1.0 if profile.donation_count > 0 else 0.0,
            1.0 if profile.engagement_score > 0 else 0.0,
            1.0 if profile.personality_summary else 0.0,
        ]
        return round(sum(scores) / len(scores), 3)
    
    # ==================== Public API ====================
    
    async def build_profile(
        self,
        donor_id: str,
        donations: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> DonorProfile:
        """
        Build a comprehensive donor profile using autonomous planning.
        
        The agent will autonomously:
        1. Analyze the data and decide what analysis is needed
        2. Plan which tools to use and in what order
        3. Adapt the plan based on data quality and results
        4. Generate insights and predictions
        5. Save the complete profile
        """
        self._logger.info("building_profile", donor_id=donor_id, donation_count=len(donations))
        
        # Validate and normalize donations
        if not donations:
            donations = []
        
        # Ensure all donations are dicts with required fields
        normalized_donations = []
        for d in donations:
            if isinstance(d, dict):
                normalized_donations.append({
                    "amount": float(d.get("amount", 0)),
                    "campaign_category": d.get("category", d.get("campaign_category", "other")),
                    "campaign_title": d.get("campaign_title", d.get("description", "")),
                    "timestamp": d.get("timestamp", ""),
                })
        
        donations = normalized_donations
        
        # Prepare context for autonomous planning
        donation_summary = {
            "count": len(donations),
            "total_amount": sum(d.get("amount", 0) for d in donations if isinstance(d, dict)),
            "categories": list(set(d.get("campaign_category", "other") for d in donations if isinstance(d, dict))),
            "date_range": {
                "earliest": min((d.get("timestamp", "") for d in donations if isinstance(d, dict) and d.get("timestamp")), default=""),
                "latest": max((d.get("timestamp", "") for d in donations if isinstance(d, dict) and d.get("timestamp")), default=""),
            }
        }
        
        goal = f"""Build a comprehensive donor profile for {donor_id}.

Available data:
- {donation_summary['count']} donations totaling ${donation_summary['total_amount']:.2f}
- Categories: {', '.join(donation_summary['categories'][:5])}
- Metadata: {json.dumps(metadata or {}, default=str)[:200]}

Consider:
- If there are few donations (<3), focus on basic patterns and use LLM for insights
- If there are many donations, do detailed statistical analysis
- Always identify cause affinities and giving motivators
- Generate personality insights using LLM
- Predict future interests based on patterns
- Save the complete profile when done"""

        context = {
            "donor_id": donor_id,
            "donations": donations,
            "donation_summary": donation_summary,
            "metadata": metadata or {},
        }
        
        # Run autonomous agent - it will plan and execute
        result = await self.run(goal, context, max_iterations=15)
        
        # Extract profile from cache (saved by save_profile tool)
        profile = self._profile_cache.get(donor_id)
        
        if not profile:
            # Fallback: create basic profile from direct tool execution if autonomous run didn't save it
            self._logger.warning("profile_not_saved_by_agent", donor_id=donor_id)
            try:
                history_result = await self._tool_analyze_history(donor_id, donations)
                affinities_result = await self._tool_identify_affinities(donations)
                
                basic_profile = DonorProfile(donor_id=donor_id)
                basic_profile.total_lifetime_giving = history_result.get("metrics", {}).get("total_amount", 0)
                basic_profile.donation_count = history_result.get("metrics", {}).get("total_donations", 0)
                basic_profile.average_donation = history_result.get("metrics", {}).get("average_amount", 0)
                
                for aff in affinities_result.get("affinities", []):
                    try:
                        category = CauseCategory(aff.get("category", "community"))
                        basic_profile.cause_affinities.append(CauseAffinity(
                            category=category,
                            score=aff.get("score", 0),
                            confidence=aff.get("confidence", 0),
                        ))
                    except:
                        pass
                
                self._profile_cache[donor_id] = basic_profile
                profile = basic_profile
            except Exception as e:
                self._logger.error("fallback_profile_creation_failed", error=str(e))
                profile = DonorProfile(donor_id=donor_id)
        
        return profile
    
    async def get_profile(self, donor_id: str) -> Optional[DonorProfile]:
        """Get a cached donor profile."""
        return self._profile_cache.get(donor_id)
    
    async def analyze_donor(
        self,
        donor_id: str,
        question: str,
        donations: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Answer a specific question about a donor using reasoning.
        
        This allows for ad-hoc analysis beyond standard profiling.
        """
        profile = self._profile_cache.get(donor_id)
        
        context = {
            "donor_id": donor_id,
            "profile": profile.to_dict() if profile else None,
            "donations": donations,
        }
        
        return await self.reason(question, context)
    
    async def _handle_build_profile(self, message) -> Dict[str, Any]:
        """A2A handler for build_profile requests."""
        payload = message.payload if hasattr(message, "payload") else message.get("payload", {})
        donations = payload.get("donations", [])
        donor_info = payload.get("donor_info", {})
        user_message = payload.get("user_message", "")
        
        # Parse donations from user_message if donations list is empty
        if not donations and user_message:
            try:
                parsed_donations = await self._parse_donations_from_message(user_message)
                donations = parsed_donations
            except Exception as e:
                self._logger.warning("Failed to parse donations from message", error=str(e))
        
        # Generate or use donor_id
        donor_id = donor_info.get("donor_id") or donor_info.get("email") or "default_donor"
        
        # Build profile
        try:
            profile = await self.build_profile(
                donor_id=donor_id,
                donations=donations,
                metadata=donor_info
            )
            return profile.to_dict()
        except Exception as e:
            self._logger.error("Failed to build profile", error=str(e), exc_info=True)
            return {"error": str(e)}
    
    async def _parse_donations_from_message(self, message: str) -> List[Dict[str, Any]]:
        """Parse donation information from natural language message using LLM."""
        prompt = f"""Parse donation information from this user message and return as JSON array:

User message: {message}

Extract all donations mentioned. For each donation, identify:
- amount (number)
- category/cause (string, e.g., "medical", "education", "emergency")
- campaign_title or description (string, optional)
- timestamp (string, optional, format: YYYY-MM-DD)

Return JSON array format:
[
  {{"amount": 50, "campaign_category": "medical", "campaign_title": "Help Sarah", "timestamp": "2024-01-15"}},
  {{"amount": 100, "campaign_category": "education", "timestamp": "2024-02-20"}}
]

If no donations found, return empty array []. Return ONLY valid JSON, no markdown."""
        
        response = await self._llm._provider.complete(
            prompt=prompt,
            system_prompt="You are a helpful assistant that extracts structured donation data from natural language. Always return valid JSON.",
            temperature=0.3,
            max_tokens=500,
        )
        
        import re
        import json
        
        # Extract JSON from response
        json_match = re.search(r'\[[\s\S]*?\]', response)
        if json_match:
            try:
                donations = json.loads(json_match.group(0))
                return donations if isinstance(donations, list) else []
            except json.JSONDecodeError:
                pass
        
        return []
    
    async def _handle_get_donor_insights(self, message) -> Dict[str, Any]:
        """A2A handler for get_donor_insights requests from other agents."""
        from src.core.base_agent import AgentMessage
        
        donor_id = message.payload.get("donor_id", "")
        profile = await self.get_profile(donor_id)
        
        if not profile:
            return {
                "insights": {},
                "error": f"No profile found for donor {donor_id}",
            }
        
        # Return key insights that other agents might need
        return {
            "insights": {
                "donor_id": donor_id,
                "primary_pattern": profile.primary_pattern.value if profile.primary_pattern else None,
                "top_causes": [
                    {
                        "category": aff.category.value,
                        "score": aff.score,
                    }
                    for aff in sorted(profile.cause_affinities, key=lambda x: x.score, reverse=True)[:3]
                ],
                "giving_motivators": {
                    k.value: v for k, v in profile.giving_motivators.items()
                },
                "personality_summary": profile.personality_summary,
                "predicted_interests": profile.predicted_interests,
            },
        }
