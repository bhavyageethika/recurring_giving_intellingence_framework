"""
LLM Service for semantic understanding and intelligent analysis.
Provides LLM-powered capabilities for all agents.
"""

import json
import random
import re
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

import structlog

from src.config import settings

logger = structlog.get_logger()


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a completion from the LLM."""
        pass
    
    @abstractmethod
    async def complete_with_tools(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        """
        Generate a completion with function calling support.
        
        Returns:
            Dict with 'content' (str) and 'tool_calls' (List[Dict]) if any
        """
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.openai_api_key
        self._client = None
        self._logger = logger.bind(provider="openai")
    
    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                if not self.api_key:
                    self._logger.debug("OpenAI API key not configured, skipping client initialization")
                    return None
                self._client = AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                # Only warn if we actually need OpenAI (have API key configured)
                if self.api_key:
                    self._logger.warning("OpenAI package not installed. Install with: pip install openai")
                # Don't log warning if API key is not set (user is using a different provider)
                return None
        return self._client
    
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a completion using OpenAI."""
        client = self._get_client()
        if not client:
            return ""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            kwargs = {
                "model": "gpt-4o-mini",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            response = await client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            self._logger.error("openai_completion_failed", error=str(e))
            return ""
    
    async def complete_with_tools(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        """Generate a completion with function calling support."""
        client = self._get_client()
        if not client:
            return {"content": "", "tool_calls": []}
        
        message_list = []
        if system_prompt:
            message_list.append({"role": "system", "content": system_prompt})
        message_list.extend(messages)
        
        try:
            kwargs = {
                "model": "gpt-4o-mini",
                "messages": message_list,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            response = await client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    })
            
            return {
                "content": message.content or "",
                "tool_calls": tool_calls,
            }
        except Exception as e:
            self._logger.error("openai_completion_with_tools_failed", error=str(e))
            return {"content": "", "tool_calls": []}
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI."""
        client = self._get_client()
        if not client:
            return []
        
        try:
            response = await client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            self._logger.error("openai_embedding_failed", error=str(e))
            return []


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.anthropic_api_key
        self._client = None
        self._logger = logger.bind(provider="anthropic")
    
    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                self._logger.warning("Anthropic package not installed. Install with: pip install anthropic")
                return None
        return self._client
    
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a completion using Claude."""
        client = self._get_client()
        if not client:
            return ""
        
        try:
            kwargs = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": max_tokens,
                "system": system_prompt or "You are a helpful assistant.",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }
            if tools:
                kwargs["tools"] = tools
            
            # Add explicit timeout to prevent hanging
            import asyncio
            message = await asyncio.wait_for(
                client.messages.create(**kwargs),
                timeout=10.0  # 10 second timeout
            )
            # Extract text content
            text_content = ""
            for content_block in message.content:
                if content_block.type == "text":
                    text_content += content_block.text
            return text_content
        except asyncio.TimeoutError:
            self._logger.error("anthropic_completion_timed_out")
            return ""
        except Exception as e:
            self._logger.error("anthropic_completion_failed", error=str(e))
            return ""
    
    async def complete_with_tools(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        """Generate a completion with function calling support."""
        client = self._get_client()
        if not client:
            return {"content": "", "tool_calls": []}
        
        try:
            # Convert messages to Anthropic format
            anthropic_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    anthropic_messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    content = []
                    if "content" in msg and msg["content"]:
                        content.append({"type": "text", "text": msg["content"]})
                    if "tool_calls" in msg:
                        for tc in msg["tool_calls"]:
                            content.append({
                                "type": "tool_use",
                                "id": tc.get("id", ""),
                                "name": tc["name"],
                                "input": tc.get("arguments", {}),
                            })
                    anthropic_messages.append({"role": "assistant", "content": content})
                elif msg["role"] == "tool":
                    anthropic_messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": str(msg.get("content", "")),
                        }]
                    })
            
            kwargs = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": max_tokens,
                "system": system_prompt or "You are a helpful assistant.",
                "messages": anthropic_messages,
                "temperature": temperature,
            }
            if tools:
                kwargs["tools"] = tools
            
            # Add explicit timeout to prevent hanging
            import asyncio
            message = await asyncio.wait_for(
                client.messages.create(**kwargs),
                timeout=10.0  # 10 second timeout
            )
            
            # Extract content and tool calls
            text_content = ""
            tool_calls = []
            
            for content_block in message.content:
                if content_block.type == "text":
                    text_content += content_block.text
                elif content_block.type == "tool_use":
                    tool_calls.append({
                        "id": content_block.id,
                        "name": content_block.name,
                        "arguments": content_block.input,
                    })
            
            return {
                "content": text_content,
                "tool_calls": tool_calls,
            }
        except asyncio.TimeoutError:
            self._logger.error("anthropic_completion_with_tools_timed_out")
            return {"content": "", "tool_calls": []}
        except Exception as e:
            self._logger.error("anthropic_completion_with_tools_failed", error=str(e))
            return {"content": "", "tool_calls": []}
    
    async def embed(self, text: str) -> List[float]:
        """
        Generate embeddings. 
        Note: Anthropic doesn't have an embedding API, so we use a hash-based approach.
        For production, consider using a dedicated embedding service.
        """
        import hashlib
        
        # Create a deterministic pseudo-embedding from text
        hash_bytes = hashlib.sha256(text.encode()).digest()
        embedding = []
        for i in range(384):
            byte_idx = i % len(hash_bytes)
            embedding.append((hash_bytes[byte_idx] - 128) / 128.0)
        
        return embedding


class IntelligentMockProvider(LLMProvider):
    """
    Intelligent mock LLM provider that generates realistic responses.
    Uses pattern matching and templates to simulate LLM behavior.
    """
    
    def __init__(self):
        self._logger = logger.bind(provider="intelligent_mock")
        self._call_count = 0
        
        # Response templates for different scenarios
        self._personality_summaries = [
            "This donor demonstrates a deep commitment to medical causes, particularly those involving children. Their consistent giving pattern suggests they respond to personal stories of struggle and hope, preferring to support individuals rather than organizations.",
            "A community-focused giver who prioritizes local impact. They show strong seasonal giving patterns around the holidays and respond well to matching opportunities. Their giving reflects values of neighborhood solidarity and grassroots change.",
            "An analytical donor who carefully researches before giving. They favor educational causes and show loyalty to campaigns they've previously supported. Their methodical approach suggests they value transparency and measurable outcomes.",
            "A compassionate giver motivated by urgent needs. They respond quickly to disaster relief and emergency medical campaigns. Their giving pattern shows impulse generosity balanced with sustained support for causes they discover.",
            "A socially-influenced donor who often gives after seeing friends' shares. They engage most with campaigns that have strong community momentum and visible progress toward goals.",
        ]
        
        self._giving_philosophies = [
            "Believes in the power of small, consistent contributions to create lasting change.",
            "Focuses on supporting individuals in crisis rather than large organizations.",
            "Values transparency and direct impact, preferring to see exactly where funds go.",
            "Motivated by community connection and the ripple effects of local giving.",
            "Sees giving as a way to express gratitude and pay forward past help received.",
        ]
        
        self._engagement_recommendations = [
            "Share personal updates from beneficiaries to maintain connection and engagement",
            "Highlight matching opportunities to leverage their responsiveness to multiplied impact",
            "Provide quarterly impact reports with specific metrics and stories",
            "Invite them to join a giving circle aligned with their cause interests",
            "Send anniversary acknowledgments celebrating their giving journey",
            "Feature them in donor spotlight stories with their permission",
            "Offer recurring giving options at their typical donation amount",
            "Connect them with other donors who share their cause passions",
        ]
        
        self._campaign_categories = [
            "medical", "education", "emergency", "community", "animals", 
            "children", "veterans", "environment", "arts_culture", "housing"
        ]
        
        self._urgency_factors = [
            "Time-sensitive medical treatment deadline",
            "Immediate housing need due to eviction",
            "School enrollment deadline approaching",
            "Critical surgery scheduled within weeks",
            "Emergency relief needed after disaster",
            "Matching funds expire soon",
        ]
        
        self._impact_statements = [
            "Your ${amount}/month provides {count} meals for families in need",
            "This support covers {weeks} weeks of medical treatment",
            "Your generosity helps {count} students access educational materials",
            "Each contribution provides shelter for {count} nights",
            "Your recurring gift sustains ongoing care for {count} individuals",
        ]
    
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generate an intelligent mock completion."""
        self._call_count += 1
        self._logger.info("mock_completion", call_number=self._call_count, prompt_preview=prompt[:100])
        
        prompt_lower = prompt.lower()
        
        # Detect what kind of response is needed
        if "json" in prompt_lower:
            return self._generate_json_response(prompt_lower, prompt)
        elif "personality" in prompt_lower or "summary" in prompt_lower:
            return self._generate_personality_response(prompt_lower)
        elif "impact" in prompt_lower:
            return self._generate_impact_response(prompt_lower)
        elif "pitch" in prompt_lower or "compelling" in prompt_lower:
            return self._generate_pitch_response(prompt_lower, prompt)
        elif "narrative" in prompt_lower or "connection" in prompt_lower:
            return self._generate_narrative_response(prompt)
        elif "mission" in prompt_lower:
            return self._generate_mission_response(prompt)
        elif "plan" in prompt_lower or "strategy" in prompt_lower:
            return self._generate_plan_response(prompt_lower)
        elif "recommend" in prompt_lower or "suggest" in prompt_lower:
            return self._generate_recommendation_response(prompt)
        elif "analyze" in prompt_lower or "assess" in prompt_lower:
            return self._generate_analysis_response(prompt_lower, prompt)
        else:
            return self._generate_general_response(prompt)
    
    def _generate_json_response(self, prompt_lower: str, prompt: str) -> str:
        """Generate a JSON response based on context."""
        
        if "campaign" in prompt_lower and ("analyze" in prompt_lower or "assess" in prompt_lower):
            # Campaign analysis
            category = random.choice(self._campaign_categories)
            return json.dumps({
                "primary_category": category,
                "subcategories": [f"{category}_specific", "treatment", "support"],
                "beneficiary": {
                    "type": random.choice(["individual", "family", "organization"]),
                    "age_group": random.choice(["child", "adult", "senior", "mixed"]),
                    "situation": "Facing challenging circumstances requiring community support"
                },
                "urgency": {
                    "level": random.choice(["critical", "high", "medium"]),
                    "factors": random.sample(self._urgency_factors, 2),
                    "time_sensitivity": "Funds needed within the next 30 days"
                },
                "key_themes": ["hope", "community support", "resilience", "family"],
                "sentiment": random.choice(["hopeful", "urgent", "grateful", "determined"]),
                "legitimacy_signals": ["verified organizer", "regular updates", "transparent spending"],
                "is_recurring_need": random.choice([True, False]),
                "summary": "A compelling campaign that demonstrates genuine need and community engagement.",
                "match_keywords": [category, "support", "community", "help", "urgent"]
            }, indent=2)
        
        elif "suitab" in prompt_lower and "recurring" in prompt_lower:
            # Recurring suitability
            suitable = random.random() > 0.3
            return json.dumps({
                "suitable": suitable,
                "confidence": round(random.uniform(0.6, 0.95), 2),
                "reasoning": "The campaign shows ongoing need with regular updates and transparent fund usage" if suitable else "This appears to be a one-time need with a specific goal",
                "suggested_monthly_impact": f"${random.randint(15, 50)}/month could provide ongoing support for essential needs"
            }, indent=2)
        
        elif "motivat" in prompt_lower or "inspir" in prompt_lower:
            # Giving motivators
            return json.dumps({
                "additional_motivators": [
                    "Personal connection to the cause through life experience",
                    "Desire to make tangible, visible impact",
                    "Community belonging and shared purpose"
                ],
                "reasoning": "Based on donation patterns and campaign selections, this donor values personal stories and measurable outcomes."
            }, indent=2)
        
        elif "impact" in prompt_lower and "project" in prompt_lower:
            # Impact projection
            return json.dumps({
                "monthly_impact": f"Provides essential support for {random.randint(2, 10)} individuals each month",
                "yearly_impact": f"Over 12 months, your giving will touch {random.randint(20, 100)} lives",
                "three_year_impact": f"Your sustained generosity creates lasting change for an entire community"
            }, indent=2)
        
        elif "nudge" in prompt_lower or "message" in prompt_lower:
            # Engagement nudge
            return json.dumps({
                "subject": "You're making a difference - here's how",
                "message": "Your generous support continues to create ripples of positive change. We wanted to share how your contributions have directly impacted the lives of those you've helped.",
                "call_to_action": "See the stories of hope you've helped write"
            }, indent=2)
        
        elif "reengag" in prompt_lower or "lapsed" in prompt_lower:
            # Re-engagement plan
            return json.dumps({
                "analysis": "This donor was previously engaged but may have experienced life changes or competing priorities. Their past giving shows genuine care for the causes they supported.",
                "steps": [
                    {"timing": "day 0", "theme": "Warm reconnection", "message_idea": "We've missed you - here's what's happened since your last gift"},
                    {"timing": "day 7", "theme": "Impact reminder", "message_idea": "Your past support made this possible - see the ongoing impact"},
                    {"timing": "day 14", "theme": "Easy return", "message_idea": "A small gift today continues the change you started"}
                ]
            }, indent=2)
        
        elif "predict" in prompt_lower and "interest" in prompt_lower:
            # Future interests prediction
            return json.dumps({
                "predictions": [
                    {"interest": "Local community health initiatives", "reasoning": "Aligns with medical cause affinity and geographic preference", "confidence": 0.85},
                    {"interest": "Youth education programs", "reasoning": "Extension of children-focused giving pattern", "confidence": 0.78},
                    {"interest": "Emergency relief funds", "reasoning": "History of responding to urgent needs", "confidence": 0.72},
                    {"interest": "Recurring giving programs", "reasoning": "Consistent giving pattern suggests readiness for commitment", "confidence": 0.68},
                    {"interest": "Giving circles with like-minded donors", "reasoning": "Social giving tendencies indicate community interest", "confidence": 0.65}
                ]
            }, indent=2)
        
        elif "diversif" in prompt_lower:
            # Diversification analysis
            return json.dumps({
                "is_diversified": True,
                "recommendations": "Consider adding environmental causes to broaden impact",
                "analysis": "Your giving plan shows healthy diversification across medical and education causes. Adding one more category could further spread your impact."
            }, indent=2)
        
        else:
            # Generic JSON response
            return json.dumps({
                "status": "analyzed",
                "confidence": round(random.uniform(0.7, 0.95), 2),
                "insights": ["Pattern detected", "Recommendation generated", "Action suggested"],
                "next_steps": ["Review findings", "Implement suggestions", "Monitor outcomes"]
            }, indent=2)
    
    def _generate_personality_response(self, prompt_lower: str) -> str:
        """Generate personality/summary response."""
        return random.choice(self._personality_summaries)
    
    def _generate_impact_response(self, prompt_lower: str) -> str:
        """Generate impact statement."""
        template = random.choice(self._impact_statements)
        return template.format(
            amount=random.randint(15, 50),
            count=random.randint(5, 50),
            weeks=random.randint(2, 12)
        )
    
    def _generate_pitch_response(self, prompt_lower: str, prompt: str) -> str:
        """Generate a compelling pitch."""
        pitches = [
            "Every month, your support provides a lifeline to families facing their toughest moments. By joining as a recurring donor, you become part of a community of changemakers creating sustained hope. Your $25 monthly doesn't just help once—it builds a foundation of support that families can count on.",
            "Imagine knowing that each month, your generosity is actively changing lives. As a recurring supporter, you'll receive personal updates showing exactly how your contributions make a difference. Together, we're not just giving—we're building a movement of sustained compassion.",
            "Your consistent support means the world to those we serve. When you commit to monthly giving, you're telling families in need that they're not alone—not just today, but every day. Join our community of dedicated supporters making lasting change.",
        ]
        return random.choice(pitches)
    
    def _generate_narrative_response(self, prompt: str) -> str:
        """Generate a connection narrative."""
        narratives = [
            "Your colleague at work, Sarah, recently started this campaign to help her neighbor's family. Three other people from your company have already contributed, creating a wave of workplace generosity.",
            "Someone from your Stanford alumni network is facing a challenging medical situation. Fellow alumni have rallied together, and your support would join a community of Treehouse members helping one of their own.",
            "This campaign is right in your neighborhood—the family lives just a few blocks away. Local supporters have already raised significant funds, and your contribution would strengthen this community response.",
            "A friend of your connection Mike is organizing this fundraiser. The personal connection through your network makes this an opportunity to support someone within your extended community.",
        ]
        return random.choice(narratives)
    
    def _generate_mission_response(self, prompt: str) -> str:
        """Generate a mission statement."""
        missions = [
            "Together, we amplify our individual generosity into collective impact. Our circle believes that when community members unite around shared values, we create change that none of us could achieve alone.",
            "We are neighbors, colleagues, and friends united by the belief that giving together multiplies our impact. Our mission is to transform individual acts of kindness into a powerful force for community good.",
            "United by compassion, guided by purpose. Our giving circle exists to pool our resources and wisdom, supporting causes that reflect our shared commitment to making our community stronger.",
        ]
        return random.choice(missions)
    
    def _generate_plan_response(self, prompt_lower: str) -> str:
        """Generate a plan summary."""
        return "Your personalized giving plan balances impact across multiple causes you care about. By distributing your monthly commitment across medical, education, and community initiatives, you're creating a diversified portfolio of positive change. Each contribution builds on the others, creating a multiplier effect that extends your generosity further than any single gift could reach."
    
    def _generate_recommendation_response(self, prompt: str) -> str:
        """Generate recommendations."""
        recs = random.sample(self._engagement_recommendations, 3)
        return f"Based on this donor's profile, I recommend: 1) {recs[0]}. 2) {recs[1]}. 3) {recs[2]}."
    
    def _generate_analysis_response(self, prompt_lower: str, prompt: str) -> str:
        """Generate analysis response."""
        if "urgency" in prompt_lower:
            return f"Urgency Assessment: This campaign shows {random.choice(['critical', 'high', 'moderate'])} urgency. Key factors include {random.choice(self._urgency_factors).lower()} and the clear timeline for fund utilization. The organizer has provided transparent updates, increasing confidence in the assessment."
        elif "legitim" in prompt_lower:
            return "Legitimacy Analysis: This campaign demonstrates strong trust signals including verified organizer identity, consistent update history, and transparent fund allocation. The beneficiary's situation is well-documented with supporting evidence."
        else:
            return "Analysis complete. The data reveals consistent patterns suggesting genuine engagement and clear opportunities for enhanced connection. Recommended actions have been identified based on behavioral indicators."
    
    def _generate_general_response(self, prompt: str) -> str:
        """Generate a general response."""
        responses = [
            "Based on the available information, this situation presents a clear opportunity for meaningful engagement. The patterns observed suggest genuine need and potential for positive impact.",
            "The analysis reveals several key insights that can guide decision-making. The data points toward authentic engagement opportunities aligned with stated values and preferences.",
            "This assessment indicates strong alignment between the opportunity and the stated objectives. The evidence supports moving forward with the recommended approach.",
        ]
        return random.choice(responses)
    
    async def embed(self, text: str) -> List[float]:
        """Return a mock embedding (deterministic hash-based)."""
        import hashlib
        
        # Create a deterministic pseudo-embedding from text
        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Convert to 384-dimensional vector
        embedding = []
        for i in range(384):
            byte_idx = i % len(hash_bytes)
            embedding.append((hash_bytes[byte_idx] - 128) / 128.0)
        
        return embedding


# Alias for backward compatibility
MockLLMProvider = IntelligentMockProvider


class LLMService:
    """
    Central LLM service for the platform.
    Provides intelligent analysis capabilities to all agents.
    """
    
    def __init__(self, provider: Optional[LLMProvider] = None):
        if provider:
            self._provider = provider
        else:
            # Select provider based on configuration
            llm_provider = settings.llm_provider.lower()
            
            if llm_provider == "anthropic" and settings.anthropic_api_key:
                self._provider = AnthropicProvider()
                logger.info("Using Anthropic Claude as LLM provider")
            elif llm_provider == "openai" and settings.openai_api_key:
                self._provider = OpenAIProvider()
                logger.info("Using OpenAI as LLM provider")
            elif settings.anthropic_api_key:
                self._provider = AnthropicProvider()
                logger.info("Using Anthropic Claude as LLM provider (auto-detected)")
            elif settings.openai_api_key:
                self._provider = OpenAIProvider()
                logger.info("Using OpenAI as LLM provider (auto-detected)")
            else:
                self._provider = IntelligentMockProvider()
                logger.info("Using Mock LLM provider (no API keys configured)")
        
        self._logger = logger.bind(component="llm_service")
        
        # System prompts for different tasks
        self._system_prompts = {
            "campaign_analysis": """You are an expert at analyzing crowdfunding campaigns. 
Your task is to extract structured information about campaigns including:
- Primary and secondary categories
- Urgency level and factors
- Beneficiary information
- Key themes and sentiment
- Legitimacy indicators
Always respond with valid JSON.""",
            
            "donor_profiling": """You are an expert at understanding donor behavior and motivations.
Your task is to analyze donation patterns and create donor profiles including:
- Giving style and preferences
- Causes that inspire them
- Engagement patterns
- Recommended engagement approaches
Always respond with valid JSON.""",
            
            "campaign_matching": """You are an expert at matching donors with campaigns they would find inspiring.
Your task is to explain why specific campaigns would resonate with a donor based on:
- Their cause affinities
- Their giving motivators
- Their past donation patterns
- Geographic and social proximity
Provide clear, compelling reasons for each match.""",
            
            "engagement_personalization": """You are an expert at crafting personalized donor communications.
Your task is to create engaging, heartfelt messages that:
- Acknowledge the donor's past generosity
- Share meaningful updates
- Inspire continued engagement
- Maintain authentic, warm tone
Never use manipulative language or pressure tactics.""",
        }
    
    async def analyze_campaign_semantics(
        self,
        title: str,
        description: str,
        category: str = "",
    ) -> Dict[str, Any]:
        """Use LLM to perform deep semantic analysis of a campaign."""
        prompt = f"""Analyze this crowdfunding campaign:

Title: {title}
Category: {category or 'Not specified'}
Description: {description[:2000]}

Extract and return as JSON:
1. "primary_category": The main cause category
2. "subcategories": List of specific subcategories
3. "urgency_level": "critical", "high", "medium", or "low"
4. "urgency_factors": List of factors contributing to urgency
5. "beneficiary_type": "individual", "family", "organization", or "community"
6. "key_themes": List of inspirational/thematic elements
7. "sentiment": Overall tone (hopeful, urgent, grateful, etc.)
8. "legitimacy_signals": List of trust indicators found
9. "is_recurring_need": true/false
10. "suggested_tags": List of tags for matching"""

        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self._system_prompts["campaign_analysis"],
            temperature=0.3,
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self._logger.warning("failed_to_parse_llm_response", response=response[:200])
            return {}
    
    async def analyze_donor_behavior(
        self,
        donations: List[Dict[str, Any]],
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Use LLM to analyze donor behavior patterns."""
        donation_summary = []
        for d in donations[:20]:
            donation_summary.append({
                "campaign": d.get("campaign_title", "")[:50],
                "category": d.get("campaign_category", ""),
                "amount": d.get("amount", 0),
                "source": d.get("source", ""),
            })
        
        prompt = f"""Analyze this donor's giving behavior:

Donation History (most recent):
{json.dumps(donation_summary, indent=2)}

Donor Info:
{json.dumps(metadata or {}, indent=2)}

Provide analysis as JSON:
1. "giving_style": Description of their giving approach
2. "primary_motivators": What inspires them to give
3. "cause_preferences": Ranked list of cause preferences
4. "engagement_pattern": How they typically engage
5. "optimal_ask_amount": Suggested donation amount range
6. "best_channels": Preferred communication channels
7. "personalization_tips": How to best engage this donor
8. "churn_risk": "low", "medium", or "high"
9. "growth_potential": Potential for increased giving"""

        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self._system_prompts["donor_profiling"],
            temperature=0.4,
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {}
    
    async def generate_match_explanation(
        self,
        donor_profile: Dict[str, Any],
        campaign: Dict[str, Any],
    ) -> str:
        """Generate a personalized explanation for why a campaign matches a donor."""
        prompt = f"""Create a brief, compelling explanation for why this donor would connect with this campaign.

Donor Profile:
- Causes they care about: {donor_profile.get('cause_affinities', [])}
- What inspires them: {donor_profile.get('giving_motivators', {})}
- Average donation: ${donor_profile.get('average_donation', 50)}

Campaign:
- Title: {campaign.get('title', '')}
- Category: {campaign.get('category', '')}
- Description: {campaign.get('description', '')[:500]}

Write 2-3 sentences explaining why this campaign would resonate with them.
Be specific and personal, not generic. Focus on the connection, not pressure."""

        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self._system_prompts["campaign_matching"],
            temperature=0.7,
        )
        
        return response
    
    async def personalize_engagement_message(
        self,
        donor_name: str,
        message_type: str,
        context: Dict[str, Any],
    ) -> Dict[str, str]:
        """Generate personalized engagement message."""
        prompt = f"""Create a personalized {message_type} message for {donor_name}.

Context:
{json.dumps(context, indent=2)}

Generate a JSON response with:
1. "subject": Email subject line (compelling but not clickbait)
2. "body": Message body (warm, authentic, 2-3 paragraphs)
3. "call_to_action": Optional soft CTA if appropriate"""

        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self._system_prompts["engagement_personalization"],
            temperature=0.8,
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"subject": "", "body": response}
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        return await self._provider.embed(text)
    
    async def generate_campaign_embedding(
        self,
        title: str,
        description: str,
        category: str = "",
        tags: List[str] = None,
    ) -> List[float]:
        """Generate a rich embedding for a campaign."""
        text_parts = [title, category]
        if tags:
            text_parts.extend(tags)
        text_parts.append(description[:1000])
        
        combined_text = " | ".join(filter(None, text_parts))
        return await self._provider.embed(combined_text)
    
    async def calculate_semantic_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """Calculate semantic similarity between two texts."""
        emb1 = await self.generate_embedding(text1)
        emb2 = await self.generate_embedding(text2)
        
        if not emb1 or not emb2:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a * a for a in emb1) ** 0.5
        norm2 = sum(b * b for b in emb2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


# Global LLM service instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create the global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
