"""
Campaign Intelligence Agent

Provides comprehensive campaign analysis including:
- Quality scoring with improvement suggestions
- Success probability prediction
- Similar successful campaigns analysis
- Optimal sharing strategy generation
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime
import json
import re
import structlog

from src.core.autonomous_agent import AutonomousAgent, Tool
from src.core.llm_service import get_llm_service

logger = structlog.get_logger()


class QualityDimension(str, Enum):
    """Dimensions of campaign quality."""
    STORY_CLARITY = "story_clarity"
    URGENCY = "urgency"
    TRANSPARENCY = "transparency"
    SOCIAL_PROOF = "social_proof"
    SHAREABILITY = "shareability"
    GOAL_REALISM = "goal_realism"
    UPDATE_FREQUENCY = "update_frequency"


@dataclass
class QualityScore:
    """Campaign quality score breakdown."""
    overall_score: float
    dimension_scores: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]
    priority_improvements: List[str]  # Top 3 most impactful


@dataclass
class SuccessPrediction:
    """Predicted success probability and reasoning."""
    success_probability: float  # 0-1
    predicted_amount: float
    predicted_donors: int
    confidence_level: str  # "high", "medium", "low"
    key_factors: List[str]
    risk_factors: List[str]
    reasoning: str


@dataclass
class SimilarCampaign:
    """Similar successful campaign analysis."""
    campaign_id: str
    title: str
    success_metrics: Dict[str, Any]
    what_made_it_work: List[str]
    similarity_score: float
    lessons_applicable: List[str]


@dataclass
class CampaignIntelligence:
    """Complete campaign intelligence report."""
    campaign_id: str
    quality_score: QualityScore
    success_prediction: SuccessPrediction
    similar_campaigns: List[SimilarCampaign]
    messaging_variants: Dict[str, str]  # channel -> message
    timestamp: str


class CampaignIntelligenceAgent(AutonomousAgent[CampaignIntelligence]):
    """
    Autonomous agent for comprehensive campaign intelligence.
    
    Provides:
    - Quality scoring with actionable improvements
    - Success probability prediction
    - Similar successful campaigns analysis
    - Messaging variants for different channels
    """
    
    SYSTEM_PROMPT = """You are an expert campaign intelligence analyst specializing in crowdfunding success.
Your expertise includes:
- Evaluating campaign quality across multiple dimensions
- Predicting campaign success based on patterns and data
- Identifying what makes campaigns successful
- Generating actionable improvement suggestions
- Generating effective messaging variants for different channels

You provide comprehensive, actionable intelligence that helps organizers:
- Understand their campaign's strengths and weaknesses
- Predict likely outcomes
- Learn from similar successful campaigns
- Create effective messaging for different channels

You are thorough, data-driven, and focused on actionable insights.
IMPORTANT: Always return valid JSON. Do not include markdown code blocks or extra text. Return only the JSON object."""
    
    def _parse_json_response(self, response: str, retry_count: int = 0) -> Dict[str, Any]:
        """Robustly parse JSON from LLM response with multiple fallback strategies."""
        if not response or not response.strip():
            raise ValueError("Empty response from LLM")
        
        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON from markdown code blocks
        # Find the content between ```json and ``` (handle multiline)
        json_start = response.find('```json')
        if json_start != -1:
            json_start = response.find('\n', json_start) + 1
            json_end = response.find('```', json_start)
            if json_end != -1:
                json_str = response[json_start:json_end].strip()
                try:
                    # Clean up common issues
                    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass
        
        # Try without "json" label
        json_start = response.find('```')
        if json_start != -1:
            json_start = response.find('\n', json_start) + 1
            json_end = response.find('```', json_start)
            if json_end != -1:
                json_str = response[json_start:json_end].strip()
                # Check if it looks like JSON (starts with { or [)
                if json_str.startswith(('{', '[')):
                    try:
                        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        pass
        
        # Strategy 3: Find JSON object using regex
        json_object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_object_pattern, response, re.DOTALL)
        
        for match in matches:
            # Try to find the largest valid JSON object
            try:
                # Clean up common issues
                cleaned = match.strip()
                # Remove trailing commas before closing braces/brackets
                cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
                # Fix single quotes to double quotes (basic)
                cleaned = cleaned.replace("'", '"')
                
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and len(parsed) > 0:
                    return parsed
            except json.JSONDecodeError:
                continue
        
        # Strategy 4: Try to extract and fix common JSON issues
        # Find the first { and last } and try to parse
        first_brace = response.find('{')
        last_brace = response.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = response[first_brace:last_brace + 1]
            try:
                # Remove trailing commas
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Strategy 5: If all else fails, try to reconstruct from response
        # This is a last resort - try to extract key information
        logger.warning("Failed to parse JSON, attempting to extract key information", response=response[:200])
        
        # Try one more time with a cleaned version
        cleaned_response = response.strip()
        # Remove markdown formatting
        cleaned_response = re.sub(r'```[a-z]*\s*', '', cleaned_response)
        cleaned_response = re.sub(r'```', '', cleaned_response)
        # Remove leading/trailing non-JSON text
        cleaned_response = re.sub(r'^[^{]*', '', cleaned_response)
        cleaned_response = re.sub(r'[^}]*$', '', cleaned_response)
        
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.error("All JSON parsing strategies failed", error=str(e), response_preview=response[:500])
            raise ValueError(f"Could not parse JSON from LLM response: {str(e)}")
    
    def __init__(self):
        super().__init__(
            agent_id="campaign_intelligence_agent",
            name="Campaign Intelligence Agent",
            description="Comprehensive campaign analysis and intelligence",
            system_prompt=self.SYSTEM_PROMPT,
        )
        
        # Intelligence reports cache
        self._intelligence_reports: Dict[str, CampaignIntelligence] = {}
        
        # Register domain-specific tools
        for tool in self._get_domain_tools():
            self.register_tool(tool)
        
        # Register A2A message handlers
        self.register_handler("analyze_campaign", self._handle_analyze_campaign)
    
    def _get_domain_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT
    
    def _get_domain_tools(self) -> List[Tool]:
        """Get domain-specific tools for campaign intelligence."""
        return [
            Tool(
                name="assess_quality",
                description="Assess campaign quality across multiple dimensions and provide improvement suggestions",
                parameters={
                    "type": "object",
                    "properties": {
                        "campaign_data": {
                            "type": "object",
                            "description": "Campaign data to assess",
                        },
                    },
                    "required": ["campaign_data"],
                },
                function=self._tool_assess_quality,
            ),
            Tool(
                name="predict_success",
                description="Predict campaign success probability and key factors",
                parameters={
                    "type": "object",
                    "properties": {
                        "campaign_data": {
                            "type": "object",
                            "description": "Campaign data to analyze",
                        },
                        "quality_score": {
                            "type": "object",
                            "description": "Quality score from assessment",
                        },
                    },
                    "required": ["campaign_data"],
                },
                function=self._tool_predict_success,
            ),
            Tool(
                name="find_similar_successful",
                description="Find similar successful campaigns and analyze what made them work",
                parameters={
                    "type": "object",
                    "properties": {
                        "campaign_data": {
                            "type": "object",
                            "description": "Campaign to find similar ones for",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of similar campaigns to find",
                            "default": 5,
                        },
                    },
                    "required": ["campaign_data"],
                },
                function=self._tool_find_similar_successful,
            ),
            Tool(
                name="generate_messaging_variants",
                description="Generate messaging variants for different channels (Facebook, Twitter, Email, etc.)",
                parameters={
                    "type": "object",
                    "properties": {
                        "campaign_data": {
                            "type": "object",
                            "description": "Campaign data to generate messaging for",
                        },
                    },
                    "required": ["campaign_data"],
                },
                function=self._tool_generate_messaging_variants,
            ),
        ]
    
    async def _tool_assess_quality(
        self,
        campaign_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assess campaign quality across multiple dimensions."""
        title = campaign_data.get("title", "")
        description = campaign_data.get("description", "")
        goal = campaign_data.get("goal_amount", 0)
        raised = campaign_data.get("raised_amount", 0)
        donors = campaign_data.get("donor_count", 0)
        updates = campaign_data.get("updates", [])
        
        llm = get_llm_service()
        
        # Use LLM to assess quality dimensions
        prompt = f"""Assess the quality of this campaign across multiple dimensions:

Campaign:
- Title: {title}
- Description: {description[:500]}...
- Goal: ${goal:,.0f}
- Raised: ${raised:,.0f} ({raised/goal*100 if goal > 0 else 0:.1f}%)
- Donors: {donors}
- Updates: {len(updates)} updates

Assess these dimensions (0-1 scale):
1. Story Clarity: How clear and compelling is the narrative?
2. Urgency: How urgent is the need?
3. Transparency: How transparent about use of funds?
4. Social Proof: How much evidence of legitimacy (donors, updates, testimonials)?
5. Shareability: How easy to share and understand?
6. Goal Realism: How realistic is the goal?
7. Update Frequency: How actively updated?

For each dimension:
- Score (0-1)
- Brief explanation
- If score < 0.7, provide improvement suggestion

Return JSON:
{{
  "dimension_scores": {{
    "story_clarity": {{"score": 0.8, "explanation": "...", "suggestion": "..."}},
    ...
  }},
  "strengths": ["..."],
  "weaknesses": ["..."],
  "improvement_suggestions": ["..."],
  "priority_improvements": ["..."]  // Top 3 most impactful
}}"""
        
        response = await llm._provider.complete(
            prompt=prompt,
            system_prompt="You are an expert campaign quality assessor. Be specific and actionable. Return ONLY valid JSON, no markdown or extra text.",
            temperature=0.3,
        )
        
        try:
            assessment = self._parse_json_response(response)
        except (ValueError, json.JSONDecodeError) as e:
            logger.error("Failed to parse quality assessment JSON", error=str(e), response_preview=response[:200])
            # Retry with a more explicit prompt
            retry_prompt = prompt + "\n\nCRITICAL: Return ONLY valid JSON. No markdown, no code blocks, no explanations. Just the JSON object."
            retry_response = await llm._provider.complete(
                prompt=retry_prompt,
                system_prompt="You are an expert campaign quality assessor. Return ONLY valid JSON, no markdown or extra text.",
                temperature=0.2,
            )
            try:
                assessment = self._parse_json_response(retry_response)
            except (ValueError, json.JSONDecodeError) as e2:
                logger.error("Retry also failed for quality assessment", error=str(e2))
                raise ValueError(f"Could not parse quality assessment after retry: {str(e2)}")
        
        # Calculate overall score
        scores = assessment.get("dimension_scores", {})
        overall = sum(s.get("score", 0.5) for s in scores.values()) / max(len(scores), 1)
        
        return {
            "overall_score": round(overall, 3),
            "dimension_scores": {k: v.get("score", 0.5) for k, v in scores.items()},
            "strengths": assessment.get("strengths", []),
            "weaknesses": assessment.get("weaknesses", []),
            "improvement_suggestions": assessment.get("improvement_suggestions", []),
            "priority_improvements": assessment.get("priority_improvements", [])[:3],
        }
    
    def _fallback_quality_assessment(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback quality assessment using rule-based logic."""
        title = campaign_data.get("title", "")
        description = campaign_data.get("description", "")
        goal = campaign_data.get("goal_amount", 0)
        raised = campaign_data.get("raised_amount", 0)
        donors = campaign_data.get("donor_count", 0)
        
        # Simple rule-based scoring
        story_score = min(len(description) / 200, 1.0) if description else 0.5
        goal_score = 0.7 if 1000 <= goal <= 100000 else 0.5
        progress_score = min(raised / goal, 1.0) if goal > 0 else 0.5
        
        return {
            "dimension_scores": {
                "story_clarity": {"score": story_score},
                "goal_realism": {"score": goal_score},
                "social_proof": {"score": min(donors / 50, 1.0)},
            },
            "strengths": ["Clear title"] if title else [],
            "weaknesses": ["Needs more detail"] if len(description) < 100 else [],
            "improvement_suggestions": ["Add more detail to description"],
            "priority_improvements": ["Enhance story clarity"],
        }
    
    async def _tool_predict_success(
        self,
        campaign_data: Dict[str, Any],
        quality_score: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Predict campaign success probability."""
        title = campaign_data.get("title", "")
        description = campaign_data.get("description", "")
        goal = campaign_data.get("goal_amount", 0)
        raised = campaign_data.get("raised_amount", 0)
        donors = campaign_data.get("donor_count", 0)
        category = campaign_data.get("category", "")
        
        llm = get_llm_service()
        
        quality_info = ""
        if quality_score:
            quality_info = f"\nQuality Score: {quality_score.get('overall_score', 0.5):.2f}"
            quality_info += f"\nStrengths: {', '.join(quality_score.get('strengths', [])[:3])}"
        
        prompt = f"""Predict the success probability for this campaign:

Campaign:
- Title: {title}
- Category: {category}
- Goal: ${goal:,.0f}
- Current: ${raised:,.0f} ({raised/goal*100 if goal > 0 else 0:.1f}%)
- Donors: {donors}
{quality_info}

Based on patterns of successful campaigns, predict:
1. Success probability (0-1): Likelihood of reaching goal
2. Predicted final amount: Estimated total raised
3. Predicted donor count: Estimated total donors
4. Confidence level: "high", "medium", or "low"
5. Key factors: What will drive success
6. Risk factors: What could prevent success
7. Reasoning: Detailed explanation

Return JSON:
{{
  "success_probability": 0.75,
  "predicted_amount": 45000,
  "predicted_donors": 120,
  "confidence_level": "medium",
  "key_factors": ["..."],
  "risk_factors": ["..."],
  "reasoning": "..."
}}"""
        
        response = await llm._provider.complete(
            prompt=prompt,
            system_prompt="You are an expert at predicting crowdfunding success. Be realistic and data-driven.",
            temperature=0.3,
        )
        
        try:
            prediction = self._parse_json_response(response)
        except (ValueError, json.JSONDecodeError) as e:
            logger.error("Failed to parse success prediction JSON", error=str(e), response_preview=response[:200])
            # Retry with a more explicit prompt
            retry_prompt = prompt + "\n\nCRITICAL: Return ONLY valid JSON. No markdown, no code blocks, no explanations. Just the JSON object."
            retry_response = await llm._provider.complete(
                prompt=retry_prompt,
                system_prompt="You are an expert at predicting campaign success. Return ONLY valid JSON, no markdown or extra text.",
                temperature=0.2,
            )
            try:
                prediction = self._parse_json_response(retry_response)
            except (ValueError, json.JSONDecodeError) as e2:
                logger.error("Retry also failed for success prediction", error=str(e2))
                raise ValueError(f"Could not parse success prediction after retry: {str(e2)}")
        
        # Validate required fields
        if "success_probability" not in prediction:
            progress_ratio = raised / goal if goal > 0 else 0
            prediction["success_probability"] = min(progress_ratio + 0.3, 0.9) if progress_ratio > 0 else 0.4
        
        return prediction
    
    async def _tool_find_similar_successful(
        self,
        campaign_data: Dict[str, Any],
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Find similar successful campaigns and analyze what made them work."""
        title = campaign_data.get("title", "")
        description = campaign_data.get("description", "")
        category = campaign_data.get("category", "")
        
        llm = get_llm_service()
        
        prompt = f"""Find similar successful campaigns to this one:

Campaign:
- Title: {title}
- Category: {category}
- Description: {description[:300]}...

Based on successful campaigns in similar categories, identify {limit} similar campaigns that succeeded.
For each, provide:
1. Campaign title (similar theme/cause)
2. Success metrics (goal, raised, donors, timeframe)
3. What made it work (3-5 key factors)
4. Similarity score (0-1)
5. Lessons applicable to this campaign

Return JSON:
{{
  "similar_campaigns": [
    {{
      "campaign_id": "similar_1",
      "title": "...",
      "success_metrics": {{"goal": 50000, "raised": 60000, "donors": 200, "days": 45}},
      "what_made_it_work": ["..."],
      "similarity_score": 0.85,
      "lessons_applicable": ["..."]
    }},
    ...
  ]
}}"""
        
        response = await llm._provider.complete(
            prompt=prompt,
            system_prompt="You are an expert at analyzing successful campaigns. Be specific about what made them work. Return ONLY valid JSON, no markdown or extra text.",
            temperature=0.7,
        )
        
        try:
            result = self._parse_json_response(response)
            # Validate structure
            if "similar_campaigns" not in result:
                raise ValueError("Response missing 'similar_campaigns' key")
            if not isinstance(result["similar_campaigns"], list):
                raise ValueError("'similar_campaigns' must be a list")
            if len(result["similar_campaigns"]) == 0:
                raise ValueError("'similar_campaigns' list is empty")
            return result
        except (ValueError, json.JSONDecodeError) as e:
            logger.error("Failed to parse similar campaigns JSON", error=str(e), response_preview=response[:200])
            # Retry with a more explicit prompt
            retry_prompt = prompt + "\n\nCRITICAL: Return ONLY valid JSON. No markdown, no code blocks, no explanations. Just the JSON object with 'similar_campaigns' array."
            retry_response = await llm._provider.complete(
                prompt=retry_prompt,
                system_prompt="You are an expert at analyzing successful campaigns. Return ONLY valid JSON, no markdown or extra text.",
                temperature=0.5,
            )
            try:
                result = self._parse_json_response(retry_response)
                # Validate structure
                if "similar_campaigns" not in result:
                    raise ValueError("Retry response missing 'similar_campaigns' key")
                if not isinstance(result["similar_campaigns"], list):
                    raise ValueError("'similar_campaigns' must be a list")
                return result
            except (ValueError, json.JSONDecodeError) as e2:
                logger.error("Retry also failed for similar campaigns", error=str(e2))
                # Return empty list instead of raising - don't break the whole analysis
                logger.warning("Returning empty similar campaigns list due to parsing failure")
                return {"similar_campaigns": []}
    
    async def _tool_generate_messaging_variants(
        self,
        campaign_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate messaging variants for different channels."""
        title = campaign_data.get("title", "")
        description = campaign_data.get("description", "")
        category = campaign_data.get("category", "")
        
        llm = get_llm_service()
        
        prompt = f"""Generate messaging variants for this campaign across different channels:

Campaign:
- Title: {title}
- Category: {category}
- Description: {description[:300]}...

Create channel-specific messaging variants. Each message should be:
- Appropriate length for the channel (short for Twitter, longer for Facebook/Email)
- Tailored to the channel's audience and format
- Clear, compelling, and action-oriented

Return JSON:
{{
  "messaging_variants": {{
    "Facebook": "Full message for Facebook (can be longer, include story)",
    "Twitter": "Short message for Twitter (280 chars or less)",
    "Email": "Message for email (can be detailed)",
    "LinkedIn": "Professional message for LinkedIn",
    "Instagram": "Message for Instagram (engaging, visual-focused)"
  }}
}}"""
        
        response = await llm._provider.complete(
            prompt=prompt,
            system_prompt="You are an expert in creating effective campaign messaging. Be specific and actionable. Return ONLY valid JSON, no markdown or extra text.",
            temperature=0.7,
        )
        
        try:
            result = self._parse_json_response(response)
            return result
        except (ValueError, json.JSONDecodeError) as e:
            logger.error("Failed to parse messaging variants JSON", error=str(e), response_preview=response[:200])
            # Retry with a more explicit prompt
            retry_prompt = prompt + "\n\nCRITICAL: Return ONLY valid JSON. No markdown, no code blocks, no explanations. Just the JSON object with 'messaging_variants' key."
            retry_response = await llm._provider.complete(
                prompt=retry_prompt,
                system_prompt="You are an expert in creating effective campaign messaging. Return ONLY valid JSON, no markdown or extra text.",
                temperature=0.5,
            )
            try:
                result = self._parse_json_response(retry_response)
                return result
            except (ValueError, json.JSONDecodeError) as e2:
                logger.error("Retry also failed for messaging variants", error=str(e2))
                raise ValueError(f"Could not parse messaging variants after retry: {str(e2)}")
    
    async def _handle_analyze_campaign(self, message) -> Dict[str, Any]:
        """A2A handler for analyze_campaign requests."""
        payload = message.payload if hasattr(message, "payload") else message.get("payload", {})
        url = payload.get("url")
        campaign_data = payload.get("campaign_data", {})
        user_message = payload.get("user_message", "")
        
        # If URL provided, fetch campaign data first (with timeout)
        if url and not campaign_data:
            from src.agents.campaign_data_agent import CampaignDataAgent
            import asyncio
            data_agent = CampaignDataAgent()
            try:
                # Use direct tool call for speed (3 seconds max - very aggressive)
                result = await asyncio.wait_for(
                    data_agent._tool_process_url(url),  # Direct tool call is faster than process_url
                    timeout=3.0  # Very aggressive timeout
                )
                campaign_data = result.get("campaign", {})
                if not campaign_data:
                    # Fallback: create minimal campaign data from URL
                    self._logger.warning("No campaign data extracted, using minimal data", url=url[:50])
                    campaign_data = {
                        "url": url,
                        "title": "Campaign Analysis",
                        "description": f"Analysis for campaign at {url}",
                        "campaign_id": url.split('/')[-1].split('?')[0],
                    }
            except asyncio.TimeoutError:
                self._logger.warning("URL fetching timed out, using minimal data", url=url[:50])
                # Fallback: create minimal campaign data
                campaign_data = {
                    "url": url,
                    "title": "Campaign Analysis",
                    "description": f"Analysis for campaign at {url} (data fetch timed out)",
                    "campaign_id": url.split('/')[-1].split('?')[0],
                }
            except Exception as e:
                self._logger.warning("Failed to fetch campaign data, using minimal data", url=url, error=str(e))
                # Fallback: create minimal campaign data
                campaign_data = {
                    "url": url,
                    "title": "Campaign Analysis",
                    "description": f"Analysis for campaign at {url} (data fetch failed: {str(e)[:50]})",
                    "campaign_id": url.split('/')[-1].split('?')[0],
                }
        
        if not campaign_data:
            return {"error": "No campaign data or URL provided"}
        
        # Analyze campaign with aggressive timeout
        try:
            import asyncio
            intelligence = await asyncio.wait_for(
                self.analyze_campaign(campaign_data),
                timeout=12.0  # Very aggressive - 12 second timeout for analysis
            )
            return intelligence.to_dict() if hasattr(intelligence, "to_dict") else intelligence
        except asyncio.TimeoutError:
            self._logger.warning("Campaign analysis timed out, returning partial results")
            # Return a quick analysis with minimal data
            return {
                "campaign_id": campaign_data.get("campaign_id", ""),
                "quality_score": {
                    "overall_score": 0.5,
                    "message": "Analysis timed out - partial results only"
                },
                "success_prediction": {
                    "probability": 0.5,
                    "message": "Analysis incomplete due to timeout"
                },
                "similar_campaigns": [],
                "messaging_variants": {},
                "error": "Analysis timed out. Please try again with a simpler request."
            }
        except Exception as e:
            self._logger.error("Failed to analyze campaign", error=str(e), exc_info=True)
            return {"error": str(e)}
    
    async def analyze_campaign(
        self,
        campaign_data: Dict[str, Any],
    ) -> CampaignIntelligence:
        """
        Generate comprehensive campaign intelligence.
        
        Returns quality score, success prediction, similar campaigns, and sharing strategy.
        """
        campaign_id = campaign_data.get("campaign_id", str(hash(campaign_data.get("title", ""))))
        
        goal = f"""Generate comprehensive intelligence for campaign '{campaign_data.get('title', 'Unknown')}'.

Provide:
1. Quality assessment with improvement suggestions
2. Success probability prediction
3. Similar successful campaigns analysis
4. Optimal sharing strategy

Campaign details:
- Title: {campaign_data.get('title', '')}
- Category: {campaign_data.get('category', '')}
- Goal: ${campaign_data.get('goal_amount', 0):,.0f}
- Raised: ${campaign_data.get('raised_amount', 0):,.0f}
- Donors: {campaign_data.get('donor_count', 0)}
"""
        
        context = {"campaign_data": campaign_data}
        
        # Run tools directly with aggressive timeouts (skip autonomous run to avoid redundancy and speed up)
        # These tools are already optimized and don't need the full autonomous framework overhead
        import asyncio
        
        # Run quality assessment with timeout
        try:
            quality_result = await asyncio.wait_for(
                self._tool_assess_quality(campaign_data),
                timeout=8.0  # Reduced timeout
            )
            success_result = await asyncio.wait_for(
                self._tool_predict_success(campaign_data, quality_result),
                timeout=8.0  # Reduced timeout
            )
            
            # Run independent operations in parallel with timeouts
            similar_task = asyncio.wait_for(
                self._tool_find_similar_successful(campaign_data, limit=3),  # Reduced from 5
                timeout=8.0  # Reduced timeout
            )
            messaging_task = asyncio.wait_for(
                self._tool_generate_messaging_variants(campaign_data),
                timeout=8.0  # Reduced timeout
            )
            
            similar_result, messaging_result = await asyncio.gather(similar_task, messaging_task, return_exceptions=True)
            
            # Handle exceptions
            if isinstance(similar_result, Exception):
                self._logger.warning("Similar campaigns search failed", error=str(similar_result))
                similar_result = {"similar_campaigns": [], "error": str(similar_result)}
            if isinstance(messaging_result, Exception):
                self._logger.warning("Messaging variants generation failed", error=str(messaging_result))
                messaging_result = {"variants": [], "error": str(messaging_result)}
                
        except asyncio.TimeoutError:
            self._logger.error("Campaign analysis timed out")
            raise TimeoutError("Campaign analysis timed out. Please try again with a simpler request.")
        
        # Build intelligence report
        intelligence = CampaignIntelligence(
            campaign_id=campaign_id,
            quality_score=QualityScore(
                overall_score=quality_result["overall_score"],
                dimension_scores=quality_result["dimension_scores"],
                strengths=quality_result["strengths"],
                weaknesses=quality_result["weaknesses"],
                improvement_suggestions=quality_result["improvement_suggestions"],
                priority_improvements=quality_result["priority_improvements"],
            ),
            success_prediction=SuccessPrediction(
                success_probability=success_result.get("success_probability", 0.5),
                predicted_amount=success_result.get("predicted_amount", campaign_data.get("goal_amount", 0) * 0.5),
                predicted_donors=success_result.get("predicted_donors", campaign_data.get("donor_count", 0) * 2),
                confidence_level=success_result.get("confidence_level", "medium"),
                key_factors=success_result.get("key_factors", []),
                risk_factors=success_result.get("risk_factors", []),
                reasoning=success_result.get("reasoning", "Analysis completed"),
            ),
            similar_campaigns=[
                SimilarCampaign(
                    campaign_id=sc.get("campaign_id", f"similar_{i}"),
                    title=sc.get("title", "Similar Campaign"),
                    success_metrics=sc.get("success_metrics", {}),
                    what_made_it_work=sc.get("what_made_it_work", []),
                    similarity_score=sc.get("similarity_score", 0.5),
                    lessons_applicable=sc.get("lessons_applicable", []),
                )
                for i, sc in enumerate(similar_result.get("similar_campaigns", []))
            ],
            messaging_variants=messaging_result.get("messaging_variants", {}),
            timestamp=datetime.now().isoformat(),
        )
        
        self._intelligence_reports[campaign_id] = intelligence
        
        return intelligence

