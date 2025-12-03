"""
Agent 2: Campaign Taxonomy & Matching Engine (Autonomous Agent)

An LLM-based autonomous agent for deep semantic understanding of campaigns.
Uses planning, reasoning, and tool-use to:
- Classify campaigns with rich taxonomy
- Assess urgency and legitimacy
- Build semantic embeddings
- Match campaigns to donor profiles
- Generate personalized match explanations
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import hashlib
import re
import json

import structlog

from src.core.autonomous_agent import AutonomousAgent, Tool, Task, ReasoningStep, ReasoningType
from src.core.llm_service import get_llm_service
from src.core.agent_collaboration import get_collaborator

logger = structlog.get_logger()


class UrgencyLevel(str, Enum):
    """Campaign urgency levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LegitimacyIndicator(str, Enum):
    """Indicators of campaign legitimacy."""
    VERIFIED_IDENTITY = "verified_identity"
    MEDICAL_DOCUMENTATION = "medical_documentation"
    MEDIA_COVERAGE = "media_coverage"
    INSTITUTIONAL_BACKING = "institutional_backing"
    ORGANIZER_HISTORY = "organizer_history"
    UPDATE_FREQUENCY = "update_frequency"
    DONOR_TESTIMONIALS = "donor_testimonials"
    TRANSPARENT_SPENDING = "transparent_spending"


@dataclass
class CampaignTaxonomy:
    """Rich taxonomy classification for a campaign."""
    primary_category: str
    subcategories: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    beneficiary_type: str = "individual"
    beneficiary_age_group: Optional[str] = None
    geographic_scope: str = "local"
    is_recurring_need: bool = False
    estimated_duration: Optional[str] = None
    semantic_tags: List[str] = field(default_factory=list)
    llm_classification: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CampaignAnalysis:
    """Complete analysis of a campaign."""
    campaign_id: str
    title: str
    taxonomy: CampaignTaxonomy = field(default_factory=lambda: CampaignTaxonomy(primary_category="general"))
    
    # Urgency assessment
    urgency_level: UrgencyLevel = UrgencyLevel.MEDIUM
    urgency_score: float = 0.5
    urgency_factors: List[str] = field(default_factory=list)
    
    # Legitimacy assessment
    legitimacy_score: float = 0.5
    legitimacy_indicators: List[LegitimacyIndicator] = field(default_factory=list)
    
    # Community backing
    community_score: float = 0.0
    donor_count: int = 0
    share_count: int = 0
    comment_count: int = 0
    
    # Financial metrics
    goal_amount: float = 0.0
    raised_amount: float = 0.0
    funding_percentage: float = 0.0
    average_donation: float = 0.0
    
    # LLM-generated content
    summary: str = ""
    key_themes: List[str] = field(default_factory=list)
    sentiment: str = ""
    match_keywords: List[str] = field(default_factory=list)
    
    # Embedding for similarity search
    embedding: Optional[List[float]] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "title": self.title,
            "taxonomy": {
                "primary_category": self.taxonomy.primary_category,
                "subcategories": self.taxonomy.subcategories,
                "attributes": self.taxonomy.attributes,
                "beneficiary_type": self.taxonomy.beneficiary_type,
                "geographic_scope": self.taxonomy.geographic_scope,
                "is_recurring_need": self.taxonomy.is_recurring_need,
                "semantic_tags": self.taxonomy.semantic_tags,
                "llm_classification": self.taxonomy.llm_classification,
            },
            "urgency_level": self.urgency_level.value,
            "urgency_score": self.urgency_score,
            "urgency_factors": self.urgency_factors,
            "legitimacy_score": self.legitimacy_score,
            "legitimacy_indicators": [i.value for i in self.legitimacy_indicators],
            "community_score": self.community_score,
            "donor_count": self.donor_count,
            "goal_amount": self.goal_amount,
            "raised_amount": self.raised_amount,
            "funding_percentage": self.funding_percentage,
            "summary": self.summary,
            "key_themes": self.key_themes,
            "sentiment": self.sentiment,
            "analyzed_at": self.analyzed_at.isoformat(),
        }


class CampaignMatchingEngine(AutonomousAgent[CampaignAnalysis]):
    """
    Autonomous agent for deep semantic understanding and matching of campaigns.
    
    This agent:
    1. Uses LLM to deeply understand campaign content and context
    2. Builds rich taxonomies beyond simple categorization
    3. Assesses urgency and legitimacy through reasoning
    4. Creates semantic embeddings for similarity matching
    5. Generates personalized match explanations for donors
    """
    
    SYSTEM_PROMPT = """You are an expert campaign analyst specializing in crowdfunding and charitable giving.
Your expertise includes:
- Deep semantic understanding of campaign narratives
- Assessing urgency and legitimacy of fundraising campaigns
- Understanding the taxonomy of charitable causes
- Matching campaigns to donor interests and values

You read between the lines to understand:
- The real story behind each campaign
- The urgency and time-sensitivity of needs
- Signs of legitimacy and trustworthiness
- The practical appeal and resonance of campaigns

You are thorough, fair, and focused on helping donors find campaigns that truly resonate with them."""

    def __init__(self):
        super().__init__(
            agent_id="campaign_matching_engine",
            name="Campaign Taxonomy & Matching Engine",
            description="Autonomous agent for deep semantic campaign analysis and matching",
            system_prompt=self.SYSTEM_PROMPT,
        )
        
        # Campaign catalog
        self._campaign_catalog: Dict[str, CampaignAnalysis] = {}
        
        # Category taxonomy for rule-based fallback
        self._category_taxonomy = self._build_category_taxonomy()
        
        # Register domain-specific tools
        for tool in self._get_domain_tools():
            self.register_tool(tool)
        
        # Register A2A message handlers
        self.register_handler("evaluate_legitimacy", self._handle_evaluate_legitimacy)
        self.register_handler("analyze_campaign", self._handle_analyze_campaign)
        self.register_handler("match_campaign", self._handle_match_campaign)
        
        # Initialize collaborator for A2A communication
        self._collaborator = get_collaborator(self.agent_id)
    
    def _get_domain_system_prompt(self) -> str:
        return """
## Domain Expertise: Campaign Analysis & Matching

You specialize in:

1. **Semantic Classification**: Understanding campaigns beyond surface categories
   - "Help Sarah fight cancer" → medical > oncology > pediatric > treatment-phase
   
2. **Urgency Assessment**: Evaluating time-sensitivity
   - Critical: Life-threatening, hours/days matter
   - High: Time-sensitive, weeks matter
   - Medium: Important but not urgent
   - Low: Ongoing needs, no time pressure

3. **Legitimacy Evaluation**: Assessing trustworthiness
   - Verified organizer identity
   - Documentation and evidence
   - Update frequency and transparency
   - Community backing and testimonials

4. **Donor Matching**: Finding resonance between donors and campaigns
   - Cause alignment
   - Value alignment
   - Geographic/social proximity
   - Giving capacity fit
"""
    
    def _get_domain_tools(self) -> List[Tool]:
        return [
            Tool(
                name="analyze_campaign_semantics",
                description="Use LLM to deeply analyze campaign content and extract semantic meaning. Campaign data is available in context if not provided.",
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Campaign title (optional, will use context if not provided)"},
                        "description": {"type": "string", "description": "Campaign description (optional, will use context if not provided)"},
                        "category": {"type": "string", "description": "Campaign category (optional)"},
                        "campaign_data": {
                            "type": "object",
                            "description": "Campaign data object (optional, will use context if not provided)"
                        },
                    },
                    "required": [],
                },
                function=self._tool_analyze_semantics,
            ),
            Tool(
                name="build_taxonomy",
                description="Build a rich taxonomy classification for a campaign. Campaign data is available in context if not provided.",
                parameters={
                    "type": "object",
                    "properties": {
                        "campaign_data": {
                            "type": "object",
                            "description": "Campaign data object (optional, will use context if not provided)"
                        },
                        "llm_analysis": {"type": "object"},
                    },
                    "required": [],
                },
                function=self._tool_build_taxonomy,
            ),
            Tool(
                name="assess_urgency",
                description="Assess the urgency level of a campaign. Campaign data is available in context if not provided.",
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Campaign title (optional, will use context if not provided)"
                        },
                        "description": {
                            "type": "string",
                            "description": "Campaign description (optional, will use context if not provided)"
                        },
                        "campaign_data": {
                            "type": "object",
                            "description": "Full campaign data object (optional, will use context if not provided)"
                        },
                    },
                    "required": [],
                },
                function=self._tool_assess_urgency,
            ),
            Tool(
                name="evaluate_legitimacy",
                description="Evaluate the legitimacy and trustworthiness of a campaign. Campaign data is available in context if not provided.",
                parameters={
                    "type": "object",
                    "properties": {
                        "campaign_data": {
                            "type": "object",
                            "description": "Campaign data object (optional, will use context if not provided)"
                        },
                    },
                    "required": [],
                },
                function=self._tool_evaluate_legitimacy,
            ),
            Tool(
                name="generate_embedding",
                description="Generate a semantic embedding for similarity matching",
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["title"],
                },
                function=self._tool_generate_embedding,
            ),
            Tool(
                name="match_to_donor",
                description="Match a campaign to a donor profile and generate explanation",
                parameters={
                    "type": "object",
                    "properties": {
                        "campaign": {"type": "object"},
                        "donor_profile": {"type": "object"},
                    },
                    "required": ["campaign", "donor_profile"],
                },
                function=self._tool_match_to_donor,
            ),
            Tool(
                name="find_similar_campaigns",
                description="Find campaigns similar to a given campaign",
                parameters={
                    "type": "object",
                    "properties": {
                        "campaign_id": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["campaign_id"],
                },
                function=self._tool_find_similar,
            ),
            Tool(
                name="save_analysis",
                description="Save the campaign analysis to the catalog",
                parameters={
                    "type": "object",
                    "properties": {
                        "analysis": {"type": "object"},
                    },
                    "required": ["analysis"],
                },
                function=self._tool_save_analysis,
            ),
        ]
    
    def _build_category_taxonomy(self) -> Dict[str, Dict[str, List[str]]]:
        return {
            "medical": {
                "conditions": ["cancer", "heart disease", "diabetes", "rare disease", "mental health",
                              "chronic illness", "accident injury", "surgery", "transplant"],
                "stages": ["diagnosis", "treatment", "recovery", "ongoing care", "palliative"],
                "modifiers": ["pediatric", "elderly", "veteran", "uninsured"],
            },
            "education": {
                "levels": ["elementary", "high school", "college", "graduate", "vocational"],
                "needs": ["tuition", "books", "housing", "study abroad", "special education"],
            },
            "emergency": {
                "types": ["natural disaster", "fire", "accident", "crime victim", "sudden loss"],
                "needs": ["immediate relief", "rebuilding", "relocation", "funeral"],
            },
            "community": {
                "focus": ["local project", "neighborhood", "civic", "cultural", "religious"],
                "goals": ["infrastructure", "programs", "events", "preservation"],
            },
            "animals": {
                "types": ["pet", "rescue", "wildlife", "sanctuary"],
                "needs": ["medical", "shelter", "food", "adoption"],
            },
        }
    
    # ==================== Tool Implementations ====================
    
    async def _tool_analyze_semantics(
        self,
        title: Optional[str] = None,
        description: Optional[str] = None,
        category: str = "",
        campaign_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Use LLM for deep semantic analysis of campaign."""
        # Get campaign data from context if not provided
        if campaign_data is None:
            context = self._memory.current_context or {}
            campaign_data = context.get("campaign_data", {})
        
        # Extract title, description, category from campaign_data if not provided
        if title is None:
            title = campaign_data.get("title", "") if campaign_data else ""
        if description is None:
            description = campaign_data.get("description", "") if campaign_data else ""
        if not category and campaign_data:
            category = campaign_data.get("category", "")
        
        # Ensure we have at least title or description
        if not title and not description:
            return {
                "primary_category": category or "general",
                "subcategories": [],
                "key_themes": [],
                "summary": "No campaign data available for analysis",
            }
        prompt = f"""Analyze this crowdfunding campaign deeply:

Title: {title}
Category: {category or 'Not specified'}
Description: {description[:3000]}

Provide a comprehensive analysis in JSON format:
{{
    "primary_category": "main cause category",
    "subcategories": ["specific subcategory1", "subcategory2"],
    "beneficiary": {{
        "type": "individual/family/organization/community",
        "age_group": "child/adult/senior/mixed",
        "situation": "brief description of their situation"
    }},
    "urgency": {{
        "level": "critical/high/medium/low",
        "factors": ["factor1", "factor2"],
        "time_sensitivity": "description of time constraints"
    }},
    "key_themes": ["theme1", "theme2", "theme3"],
    "sentiment": "overall tone",
    "legitimacy_signals": ["signal1", "signal2"],
    "is_recurring_need": true/false,
    "summary": "2-3 sentence summary of the campaign",
    "match_keywords": ["keyword1", "keyword2", "keyword3"]
}}"""

        response = await self._llm._provider.complete(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.3,
        )
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "primary_category": category or "general",
            "subcategories": [],
            "key_themes": [],
            "summary": "",
        }
    
    async def _tool_build_taxonomy(
        self,
        campaign_data: Optional[Dict[str, Any]] = None,
        llm_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build rich taxonomy from campaign data and LLM analysis."""
        # Get campaign_data from context if not provided
        if campaign_data is None:
            context = self._memory.current_context or {}
            campaign_data = context.get("campaign_data", {})
        
        if not campaign_data:
            return {
                "taxonomy": {
                    "primary_category": "general",
                    "subcategories": [],
                    "attributes": {},
                },
                "error": "No campaign data provided"
            }
        
        llm_analysis = llm_analysis or {}
        
        taxonomy = CampaignTaxonomy(
            primary_category=llm_analysis.get("primary_category", campaign_data.get("category", "general")),
            subcategories=llm_analysis.get("subcategories", []),
            beneficiary_type=llm_analysis.get("beneficiary", {}).get("type", "individual"),
            beneficiary_age_group=llm_analysis.get("beneficiary", {}).get("age_group"),
            is_recurring_need=llm_analysis.get("is_recurring_need", False),
            semantic_tags=llm_analysis.get("match_keywords", []),
            llm_classification=llm_analysis,
        )
        
        # Extract attributes from LLM analysis
        if "beneficiary" in llm_analysis:
            taxonomy.attributes["beneficiary_situation"] = llm_analysis["beneficiary"].get("situation", "")
        
        # Determine geographic scope
        description = campaign_data.get("description", "").lower()
        if any(word in description for word in ["international", "global", "worldwide"]):
            taxonomy.geographic_scope = "international"
        elif any(word in description for word in ["national", "nationwide"]):
            taxonomy.geographic_scope = "national"
        elif any(word in description for word in ["regional", "state"]):
            taxonomy.geographic_scope = "regional"
        else:
            taxonomy.geographic_scope = "local"
        
        # Estimate duration
        if taxonomy.is_recurring_need:
            taxonomy.estimated_duration = "ongoing"
        elif any(word in description for word in ["emergency", "urgent", "immediate"]):
            taxonomy.estimated_duration = "short-term"
        else:
            taxonomy.estimated_duration = "medium-term"
        
        return {
            "taxonomy": {
                "primary_category": taxonomy.primary_category,
                "subcategories": taxonomy.subcategories,
                "attributes": taxonomy.attributes,
                "beneficiary_type": taxonomy.beneficiary_type,
                "beneficiary_age_group": taxonomy.beneficiary_age_group,
                "geographic_scope": taxonomy.geographic_scope,
                "is_recurring_need": taxonomy.is_recurring_need,
                "estimated_duration": taxonomy.estimated_duration,
                "semantic_tags": taxonomy.semantic_tags,
            }
        }
    
    async def _tool_assess_urgency(
        self,
        title: Optional[str] = None,
        description: Optional[str] = None,
        campaign_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Assess campaign urgency using LLM reasoning."""
        # Get campaign data from context if not provided
        if campaign_data is None:
            context = self._memory.current_context or {}
            campaign_data = context.get("campaign_data", {})
        
        # Get title and description from campaign_data if not provided
        if not title:
            title = campaign_data.get("title", "")
        if not description:
            description = campaign_data.get("description", "")
        
        if not title and not description:
            return {
                "urgency_score": 0.5,
                "urgency_level": "medium",
                "factors": [],
                "error": "No campaign data provided"
            }
        
        # First, use LLM to assess urgency
        prompt = f"""Assess the urgency of this campaign:

Title: {title}
Description: {description[:1500]}

Funding: ${campaign_data.get('raised_amount', 0):,.0f} of ${campaign_data.get('goal_amount', 0):,.0f} goal

On a scale of 1-10, how urgent is this campaign? Consider:
- Is there a life-threatening situation?
- Are there time constraints mentioned?
- How critical is the funding gap?

Respond in JSON:
{{
    "urgency_score": 1-10,
    "urgency_level": "critical/high/medium/low",
    "factors": ["factor1", "factor2"],
    "reasoning": "brief explanation"
}}"""

        response = await self._llm._provider.complete(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.2,
        )
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                # Normalize score to 0-1
                result["urgency_score"] = result.get("urgency_score", 5) / 10
                return result
        except:
            pass
        
        # Fallback to rule-based assessment
        text = f"{title} {description}".lower()
        urgency_keywords = {
            "critical": ["dying", "life-threatening", "hours left", "critical condition"],
            "high": ["urgent", "time-sensitive", "deadline", "running out"],
            "medium": ["needed", "important", "help"],
            "low": ["ongoing", "long-term"],
        }
        
        for level, keywords in urgency_keywords.items():
            if any(kw in text for kw in keywords):
                scores = {"critical": 0.9, "high": 0.7, "medium": 0.5, "low": 0.3}
                return {
                    "urgency_score": scores[level],
                    "urgency_level": level,
                    "factors": [kw for kw in keywords if kw in text][:3],
                }
        
        return {"urgency_score": 0.5, "urgency_level": "medium", "factors": []}
    
    async def _tool_evaluate_legitimacy(
        self,
        campaign_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate campaign legitimacy."""
        # Get campaign_data from context if not provided
        if campaign_data is None:
            context = self._memory.current_context or {}
            campaign_data = context.get("campaign_data", {})
        
        if not campaign_data:
            return {
                "legitimacy_score": 0.5,
                "indicators": [],
                "indicator_count": 0,
                "error": "No campaign data provided"
            }
        
        indicators = []
        scores = []
        
        description = campaign_data.get("description", "").lower()
        organizer = campaign_data.get("organizer", {})
        if not organizer or not isinstance(organizer, dict):
            # If organizer is a string or missing, create a dict
            organizer = {
                "name": campaign_data.get("organizer_name", ""),
                "verified": False,
                "campaign_count": 0,
            }
        updates = campaign_data.get("updates", [])
        
        # Check verified identity
        if organizer.get("verified"):
            indicators.append(LegitimacyIndicator.VERIFIED_IDENTITY)
            scores.append(0.2)
        
        # Check organizer history
        if organizer.get("campaign_count", 0) > 0:
            indicators.append(LegitimacyIndicator.ORGANIZER_HISTORY)
            scores.append(0.15)
        
        # Check for medical documentation mentions
        if re.search(r"doctor|hospital|medical center|diagnosis|treatment plan", description):
            indicators.append(LegitimacyIndicator.MEDICAL_DOCUMENTATION)
            scores.append(0.2)
        
        # Check for media coverage
        if re.search(r"news|article|coverage|reported|featured", description):
            indicators.append(LegitimacyIndicator.MEDIA_COVERAGE)
            scores.append(0.15)
        
        # Check for institutional backing
        if re.search(r"university|school|church|organization|foundation", description):
            indicators.append(LegitimacyIndicator.INSTITUTIONAL_BACKING)
            scores.append(0.15)
        
        # Check update frequency
        if len(updates) >= 3:
            indicators.append(LegitimacyIndicator.UPDATE_FREQUENCY)
            scores.append(0.15)
        
        # Base score + indicator scores
        total_score = min(1.0, 0.3 + sum(scores))
        
        return {
            "legitimacy_score": round(total_score, 3),
            "indicators": [i.value for i in indicators],
            "indicator_count": len(indicators),
            "is_legitimate": total_score >= 0.5,
        }
    
    async def _handle_evaluate_legitimacy(self, message) -> Dict[str, Any]:
        """A2A handler for evaluate_legitimacy requests from other agents."""
        campaign_data = message.payload.get("campaign_data", {})
        result = await self._tool_evaluate_legitimacy(campaign_data=campaign_data)
        return result
    
    async def _handle_analyze_campaign(self, message) -> Dict[str, Any]:
        """A2A handler for analyze_campaign requests from other agents."""
        campaign_data = message.payload.get("campaign_data", {})
        
        analysis = await self.analyze_campaign(
            campaign_data=campaign_data,
        )
        return analysis.to_dict() if hasattr(analysis, "to_dict") else analysis
    
    async def _handle_match_campaign(self, message) -> Dict[str, Any]:
        """A2A handler for match_campaign requests from other agents."""
        campaign_data = message.payload.get("campaign_data", {})
        donor_profile = message.payload.get("donor_profile", {})
        
        # AGENT COLLABORATION: If donor profile is incomplete, ask Profiler
        if not donor_profile.get("cause_affinities"):
            try:
                donor_id = message.payload.get("donor_id", "")
                if donor_id:
                    insights = await self._collaborator.ask_profiler_for_insights(donor_id)
                    donor_profile = {**donor_profile, **insights}
            except Exception as e:
                self._logger.warning("profiler_request_failed", error=str(e))
        
        match_result = await self._tool_match_to_donor(
            campaign=campaign_data,
            donor_profile=donor_profile,
        )
        return match_result
    
    async def _tool_generate_embedding(
        self,
        title: str,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate semantic embedding for the campaign."""
        tags = tags or []
        
        # Combine text for embedding
        text_parts = [title]
        text_parts.extend(tags)
        if description:
            text_parts.append(description[:1000])
        
        combined_text = " | ".join(filter(None, text_parts))
        
        embedding = await self._llm.generate_embedding(combined_text)
        
        return {
            "embedding": embedding,
            "embedding_dim": len(embedding) if embedding else 0,
        }
    
    async def _tool_match_to_donor(
        self,
        campaign: Dict[str, Any],
        donor_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Match campaign to donor and generate explanation."""
        # Calculate base match score
        score = 0.0
        reasons = []
        
        # Cause affinity matching
        campaign_category = campaign.get("taxonomy", {}).get("primary_category", "")
        donor_affinities = {
            a.get("category"): a.get("score", 0)
            for a in donor_profile.get("cause_affinities", [])
        }
        
        if campaign_category in donor_affinities:
            affinity_score = donor_affinities[campaign_category]
            score += affinity_score * 0.4
            reasons.append(f"Matches your interest in {campaign_category}")
        
        # Motivator matching
        motivators = donor_profile.get("giving_motivators", {})
        
        if "time_sensitivity" in motivators and campaign.get("urgency_score", 0) > 0.6:
            score += motivators["time_sensitivity"] * 0.2
            reasons.append("Time-sensitive cause that resonates with you")
        
        if "community_momentum" in motivators and campaign.get("community_score", 0) > 0.5:
            score += motivators["community_momentum"] * 0.2
            reasons.append("Strong community backing")
        
        # Legitimacy bonus
        if campaign.get("legitimacy_score", 0) > 0.7:
            score += 0.1
            reasons.append("Verified and trustworthy campaign")
        
        # Generate personalized explanation using LLM
        explanation = ""
        if score > 0.3:
            prompt = f"""Write a brief, personalized explanation (2-3 sentences) for why this donor might connect with this campaign.

Campaign: {campaign.get('title', '')}
Campaign Summary: {campaign.get('summary', '')[:300]}
Campaign Category: {campaign_category}

Donor Interests: {list(donor_affinities.keys())[:5]}
Donor Giving Style: {donor_profile.get('personality_summary', '')[:200]}

Be specific and personal, focusing on the connection. Don't be pushy."""

            explanation = await self._llm._provider.complete(
                prompt=prompt,
                system_prompt="You are a thoughtful matchmaker connecting donors with causes they care about.",
                temperature=0.7,
            )
        
        return {
            "match_score": round(score, 3),
            "reasons": reasons,
            "explanation": explanation,
        }
    
    async def _tool_find_similar(
        self,
        campaign_id: str,
        limit: int = 5
    ) -> Dict[str, Any]:
        """Find similar campaigns using embeddings."""
        reference = self._campaign_catalog.get(campaign_id)
        if not reference or not reference.embedding:
            return {"similar": [], "error": "Campaign not found or no embedding"}
        
        similarities = []
        
        for cid, campaign in self._campaign_catalog.items():
            if cid == campaign_id or not campaign.embedding:
                continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(reference.embedding, campaign.embedding)
            similarities.append({
                "campaign_id": cid,
                "title": campaign.title,
                "similarity": round(similarity, 3),
                "category": campaign.taxonomy.primary_category,
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return {"similar": similarities[:limit]}
    
    async def _tool_save_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Save campaign analysis to catalog."""
        campaign_id = analysis.get("campaign_id", str(hash(analysis.get("title", ""))))
        
        # Build CampaignAnalysis object
        taxonomy_data = analysis.get("taxonomy", {})
        taxonomy = CampaignTaxonomy(
            primary_category=taxonomy_data.get("primary_category", "general"),
            subcategories=taxonomy_data.get("subcategories", []),
            attributes=taxonomy_data.get("attributes", {}),
            beneficiary_type=taxonomy_data.get("beneficiary_type", "individual"),
            beneficiary_age_group=taxonomy_data.get("beneficiary_age_group"),
            geographic_scope=taxonomy_data.get("geographic_scope", "local"),
            is_recurring_need=taxonomy_data.get("is_recurring_need", False),
            estimated_duration=taxonomy_data.get("estimated_duration"),
            semantic_tags=taxonomy_data.get("semantic_tags", []),
            llm_classification=taxonomy_data.get("llm_classification", {}),
        )
        
        campaign_analysis = CampaignAnalysis(
            campaign_id=campaign_id,
            title=analysis.get("title", ""),
            taxonomy=taxonomy,
            urgency_level=UrgencyLevel(analysis.get("urgency_level", "medium")),
            urgency_score=analysis.get("urgency_score", 0.5),
            urgency_factors=analysis.get("urgency_factors", []),
            legitimacy_score=analysis.get("legitimacy_score", 0.5),
            legitimacy_indicators=[
                LegitimacyIndicator(i) for i in analysis.get("legitimacy_indicators", [])
                if i in [e.value for e in LegitimacyIndicator]
            ],
            community_score=analysis.get("community_score", 0),
            donor_count=analysis.get("donor_count", 0),
            share_count=analysis.get("share_count", 0),
            goal_amount=analysis.get("goal_amount", 0),
            raised_amount=analysis.get("raised_amount", 0),
            funding_percentage=analysis.get("funding_percentage", 0),
            summary=analysis.get("summary", ""),
            key_themes=analysis.get("key_themes", []),
            sentiment=analysis.get("sentiment", ""),
            embedding=analysis.get("embedding"),
        )
        
        self._campaign_catalog[campaign_id] = campaign_analysis
        
        return {
            "status": "saved",
            "campaign_id": campaign_id,
            "category": taxonomy.primary_category,
        }
    
    # ==================== Helper Methods ====================
    
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    # ==================== Public API ====================
    
    async def analyze_campaign(
        self,
        campaign_data: Dict[str, Any]
    ) -> CampaignAnalysis:
        """
        Autonomously analyze a campaign.
        
        The agent will:
        1. Plan the analysis approach
        2. Use LLM for semantic understanding
        3. Build taxonomy and assess urgency/legitimacy
        4. Generate embeddings
        5. Save to catalog
        """
        title = campaign_data.get("title", "")
        description = campaign_data.get("description", "")[:200]
        category = campaign_data.get("category", "")
        campaign_id = campaign_data.get("campaign_id", str(hash(title)))
        
        goal = f"""Deeply analyze the campaign '{title[:50]}' and build a comprehensive understanding.

Campaign details:
- Title: {title}
- Category: {category or 'Unknown'}
- Description: {description}...
- Goal: ${campaign_data.get('goal_amount', 0):,.0f}
- Raised: ${campaign_data.get('raised_amount', 0):,.0f}
- Donors: {campaign_data.get('donor_count', 0)}

Required analysis:
1. Extract semantic understanding using LLM
2. Build rich taxonomy (category, beneficiary type, geographic scope)
3. Assess urgency level and factors
4. Evaluate legitimacy indicators
5. Generate embedding for similarity matching
6. Save complete analysis to catalog

Adapt your approach based on available data. If description is short, focus on title and category. If it's detailed, do deep semantic analysis."""
        
        context = {
            "campaign_id": campaign_id,
            "campaign_data": campaign_data,
        }
        
        # Run autonomous analysis
        await self.run(goal, context, max_iterations=12)
        
        return self._campaign_catalog.get(campaign_id, CampaignAnalysis(
            campaign_id=campaign_id,
            title=title,
        ))
    
    async def match_campaigns_to_donor(
        self,
        donor_profile: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find and rank campaigns that match a donor's profile.
        """
        matches = []
        
        for campaign_id, campaign in self._campaign_catalog.items():
            result = await self._tool_match_to_donor(
                campaign=campaign.to_dict(),
                donor_profile=donor_profile,
            )
            
            if result.get("match_score", 0) > 0.2:
                matches.append({
                    "campaign": campaign.to_dict(),
                    "match_score": result["match_score"],
                    "reasons": result["reasons"],
                    "explanation": result["explanation"],
                })
        
        # Sort by match score
        matches.sort(key=lambda x: x["match_score"], reverse=True)
        
        return matches[:limit]
    
    async def get_campaign(self, campaign_id: str) -> Optional[CampaignAnalysis]:
        """Get a campaign from the catalog."""
        return self._campaign_catalog.get(campaign_id)
    
    async def search_campaigns(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[CampaignAnalysis]:
        """
        Search campaigns using natural language query.
        """
        filters = filters or {}
        
        # Use LLM to understand the query
        prompt = f"""Interpret this campaign search query: "{query}"

What is the user looking for? Extract:
- Categories they might want
- Urgency preference (if any)
- Geographic preference (if any)
- Other relevant filters

Respond in JSON:
{{
    "categories": ["category1"],
    "urgency_min": 0-1 or null,
    "keywords": ["keyword1", "keyword2"]
}}"""

        response = await self._llm._provider.complete(
            prompt=prompt,
            system_prompt="You interpret search queries for a crowdfunding platform.",
            temperature=0.2,
        )
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                query_intent = json.loads(json_match.group())
        except:
            query_intent = {"categories": [], "keywords": query.split()}
        
        # Filter campaigns
        results = []
        for campaign in self._campaign_catalog.values():
            score = 0
            
            # Category match
            if query_intent.get("categories"):
                if campaign.taxonomy.primary_category in query_intent["categories"]:
                    score += 0.5
            
            # Keyword match
            campaign_text = f"{campaign.title} {' '.join(campaign.key_themes)}".lower()
            for keyword in query_intent.get("keywords", []):
                if keyword.lower() in campaign_text:
                    score += 0.2
            
            # Urgency filter
            if query_intent.get("urgency_min"):
                if campaign.urgency_score >= query_intent["urgency_min"]:
                    score += 0.2
            
            if score > 0:
                results.append((campaign, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return [r[0] for r in results[:20]]
