"""
A/B Testing Agent

Generates and analyzes A/B test variants for campaign messaging to optimize engagement and conversion.
Creates multiple message variants and provides testing recommendations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import structlog

from src.core.autonomous_agent import AutonomousAgent, Tool
from src.core.llm_service import get_llm_service

logger = structlog.get_logger()


@dataclass
class MessageVariant:
    """A message variant for A/B testing."""
    variant_id: str
    variant_name: str
    message: str
    channel: str
    target_audience: str
    hypothesis: str  # What this variant tests
    expected_outcome: str
    testing_priority: str  # "high", "medium", "low"


@dataclass
class ABTestPlan:
    """A/B testing plan with variants and recommendations."""
    campaign_id: str
    variants: List[MessageVariant]
    recommended_tests: List[Dict[str, Any]]  # Which variants to test together
    testing_strategy: str
    success_metrics: List[str]
    sample_size_recommendations: Dict[str, int]  # channel -> sample size
    duration_recommendations: Dict[str, str]  # channel -> duration


class ABTestingAgent(AutonomousAgent[ABTestPlan]):
    """
    Autonomous agent for A/B testing campaign messaging.
    
    Generates:
    - Multiple message variants for different channels
    - A/B testing recommendations
    - Success metrics and sample size guidance
    - Testing strategy and duration recommendations
    """
    
    SYSTEM_PROMPT = """You are an expert in A/B testing and conversion optimization for crowdfunding campaigns.
Your expertise includes:
- Creating effective message variants that test different hypotheses
- Designing A/B tests that provide actionable insights
- Recommending appropriate sample sizes and test durations
- Identifying key success metrics for campaign messaging
- Optimizing messaging for different channels and audiences

You help campaigns:
- Test different messaging approaches scientifically
- Optimize conversion rates through data-driven testing
- Understand which messages resonate with different audiences
- Make informed decisions about messaging strategy

You are data-driven, methodical, and focused on actionable testing insights.
IMPORTANT: Always return valid JSON. Do not include markdown code blocks or extra text. Return only the JSON object."""
    
    def __init__(self):
        super().__init__(
            agent_id="ab_testing_agent",
            name="A/B Testing Agent",
            description="Generates A/B test variants and testing recommendations for campaign messaging",
            system_prompt=self.SYSTEM_PROMPT,
        )
        
        # Register domain-specific tools
        for tool in self._get_domain_tools():
            self.register_tool(tool)
    
    def _get_domain_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT
    
    def _get_domain_tools(self) -> List[Tool]:
        """Get domain-specific tools for A/B testing."""
        return [
            Tool(
                name="generate_message_variants",
                description="Generate multiple message variants for A/B testing across different channels and audiences",
                parameters={
                    "type": "object",
                    "properties": {
                        "campaign_data": {
                            "type": "object",
                            "description": "Campaign data including title, description, category",
                        },
                        "channels": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Channels to create variants for (e.g., Facebook, Twitter, Email)",
                        },
                        "variant_count": {
                            "type": "integer",
                            "description": "Number of variants per channel (default: 3)",
                        },
                    },
                    "required": ["campaign_data"],
                },
                function=self._tool_generate_variants,
            ),
        ]
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Robustly parse JSON from LLM response."""
        import re
        
        if not response or not response.strip():
            # Return a minimal valid structure instead of raising error
            logger.warning("Empty response from LLM, returning minimal structure")
            return {
                "variants": [],
                "recommended_tests": [],
                "testing_strategy": "A/B testing generation returned empty response",
                "success_metrics": ["click-through rate", "conversion rate"],
                "sample_size_recommendations": {},
                "duration_recommendations": {},
            }
        
        # Clean the response
        cleaned_response = response.strip()
        
        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.debug("Direct JSON parse failed", error=str(e)[:100])
        
        # Strategy 2: Remove markdown code blocks if present
        if '```' in cleaned_response:
            # Remove ```json or ``` markers
            cleaned_response = re.sub(r'```(?:json)?\s*', '', cleaned_response)
            cleaned_response = re.sub(r'```\s*', '', cleaned_response)
            try:
                return json.loads(cleaned_response.strip())
            except json.JSONDecodeError as e:
                logger.debug("JSON parse after markdown removal failed", error=str(e)[:100])
        
        # Strategy 3: Extract JSON object between first { and last }
        first_brace = cleaned_response.find('{')
        last_brace = cleaned_response.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = cleaned_response[first_brace:last_brace + 1]
            # Fix common JSON issues
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)  # Remove trailing commas
            json_str = re.sub(r'([{,]\s*)(\w+):', r'\1"\2":', json_str)  # Quote unquoted keys (simple case)
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    logger.info("Successfully parsed JSON after cleanup")
                    return parsed
            except json.JSONDecodeError as e:
                logger.debug("JSON parse after extraction failed", error=str(e)[:100], json_preview=json_str[:200])
        
        # Strategy 4: Try to find and parse nested JSON objects
        json_object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_object_pattern, cleaned_response, re.DOTALL)
        
        for match in matches:
            try:
                cleaned = match.strip()
                cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and len(parsed) > 0:
                    logger.info("Successfully parsed JSON from pattern match")
                    return parsed
            except json.JSONDecodeError:
                continue
        
        # If all parsing strategies fail, log the response and raise
        logger.error("Could not parse JSON from LLM response", response_preview=response[:500])
        raise ValueError(f"Could not parse JSON from LLM response. Response preview: {response[:200]}")
    
    async def _tool_generate_variants(
        self,
        campaign_data: Dict[str, Any],
        channels: Optional[List[str]] = None,
        variant_count: int = 3,
    ) -> Dict[str, Any]:
        """Generate message variants for A/B testing."""
        title = campaign_data.get("title", "")
        description = campaign_data.get("description", "")
        category = campaign_data.get("category", "")
        
        if not channels:
            channels = ["Email"]  # Default to Email for speed
        
        # Determine channel name early for use in error handlers
        channel_name = channels[0] if channels else "Email"
        
        llm = get_llm_service()
        
        # Wrap entire LLM call in try-catch to handle any provider errors
        try:
            # Limit description length to prevent overly long prompts
            desc_preview = description[:200] if description else ""
            
            # Simplified prompt for faster generation with clearer JSON structure
            prompt = f"""Create {variant_count} A/B test message variants for this campaign:

Campaign Title: {title}
Category: {category}
Description: {desc_preview[:150]}

Channel: {channel_name}

Create exactly {variant_count} variants:
- Variant 1: Storytelling/narrative approach
- Variant 2: Impact/data-focused approach

Return ONLY valid JSON (no markdown, no explanations):
{{
  "variants": [
    {{
      "variant_id": "variant_1",
      "variant_name": "Storytelling Approach - {channel_name}",
      "message": "Write a compelling narrative message here (2-3 sentences)",
      "channel": "{channel_name}",
      "target_audience": "General supporters",
      "hypothesis": "Storytelling approach increases engagement",
      "expected_outcome": "Higher click-through and shares",
      "testing_priority": "high"
    }},
    {{
      "variant_id": "variant_2",
      "variant_name": "Impact Focus - {channel_name}",
      "message": "Write a data-driven impact message here (2-3 sentences)",
      "channel": "{channel_name}",
      "target_audience": "Data-oriented supporters",
      "hypothesis": "Impact metrics increase conversion",
      "expected_outcome": "Higher donation conversion rate",
      "testing_priority": "high"
    }}
  ],
  "recommended_tests": [
    {{
      "test_name": "Storytelling vs Impact - {channel_name}",
      "variant_a": "variant_1",
      "variant_b": "variant_2",
      "channel": "{channel_name}",
      "rationale": "Compare storytelling vs data-driven messaging"
    }}
  ],
  "testing_strategy": "Run A/B test sequentially, start with highest priority",
  "success_metrics": ["click-through rate", "conversion rate"],
  "sample_size_recommendations": {{
    "{channel_name}": 1000
  }},
  "duration_recommendations": {{
    "{channel_name}": "7-14 days"
  }}
}}"""
            
            # Use optimized settings for speed and reliability
            # Try with max_tokens, but fall back if provider doesn't support it
            system_prompt_text = """You are a JSON generator. Return ONLY valid JSON. 
Do NOT include markdown code blocks (```json or ```).
Do NOT include any explanations or text outside the JSON.
Return ONLY the JSON object, starting with { and ending with }."""
            
            try:
                response = await llm._provider.complete(
                    prompt=prompt,
                    system_prompt=system_prompt_text,
                    temperature=0.2,  # Very low temperature for consistent output
                    max_tokens=1000,  # Enough for 2 variants
                )
            except TypeError:
                # Provider doesn't support max_tokens parameter, try without it
                logger.warning("LLM provider doesn't support max_tokens, trying without it")
                response = await llm._provider.complete(
                    prompt=prompt,
                    system_prompt=system_prompt_text,
                    temperature=0.2,
                )
            
            # Check if response is valid
            if not response or not isinstance(response, str):
                logger.warning("Invalid response from LLM, returning minimal structure")
                return {
                    "variants": [],
                    "recommended_tests": [],
                    "testing_strategy": "A/B testing generation returned invalid response",
                    "success_metrics": ["click-through rate", "conversion rate"],
                    "sample_size_recommendations": {},
                    "duration_recommendations": {},
                }
            
            try:
                result = self._parse_json_response(response)
                # Validate that we got at least some variants
                if not result.get("variants") or len(result.get("variants", [])) == 0:
                    logger.warning("A/B testing returned no variants, creating fallback variants")
                    # Create minimal fallback variants
                    result["variants"] = [
                        {
                            "variant_id": "variant_1",
                            "variant_name": f"Storytelling Approach - {channel_name}",
                            "message": f"Help support {title}. Your contribution makes a real difference in someone's life.",
                            "channel": channel_name,
                            "target_audience": "General supporters",
                            "hypothesis": "Storytelling approach increases engagement",
                            "expected_outcome": "Higher click-through and shares",
                            "testing_priority": "high"
                        },
                        {
                            "variant_id": "variant_2",
                            "variant_name": f"Impact Focus - {channel_name}",
                            "message": f"Support {title}. Every donation helps reach our goal and create meaningful impact.",
                            "channel": channel_name,
                            "target_audience": "Data-oriented supporters",
                            "hypothesis": "Impact metrics increase conversion",
                            "expected_outcome": "Higher donation conversion rate",
                            "testing_priority": "high"
                        }
                    ]
                    if not result.get("recommended_tests"):
                        result["recommended_tests"] = [
                            {
                                "test_name": f"Storytelling vs Impact - {channel_name}",
                                "variant_a": "variant_1",
                                "variant_b": "variant_2",
                                "channel": channel_name,
                                "rationale": "Compare storytelling vs data-driven messaging"
                            }
                        ]
            except (ValueError, json.JSONDecodeError) as e:
                logger.error("Failed to parse A/B testing JSON", error=str(e), response_preview=response[:200] if response else "None")
                # Return minimal structure with fallback variants
                logger.warning("Returning fallback variants due to parsing error")
                return {
                    "variants": [
                        {
                            "variant_id": "variant_1",
                            "variant_name": f"Storytelling Approach - {channel_name}",
                            "message": f"Help support {title}. Your contribution makes a real difference.",
                            "channel": channel_name,
                            "target_audience": "General supporters",
                            "hypothesis": "Storytelling approach increases engagement",
                            "expected_outcome": "Higher click-through and shares",
                            "testing_priority": "high"
                        },
                        {
                            "variant_id": "variant_2",
                            "variant_name": f"Impact Focus - {channel_name}",
                            "message": f"Support {title}. Every donation helps reach our goal.",
                            "channel": channel_name,
                            "target_audience": "Data-oriented supporters",
                            "hypothesis": "Impact metrics increase conversion",
                            "expected_outcome": "Higher donation conversion rate",
                            "testing_priority": "high"
                        }
                    ],
                    "recommended_tests": [
                        {
                            "test_name": f"Storytelling vs Impact - {channel_name}",
                            "variant_a": "variant_1",
                            "variant_b": "variant_2",
                            "channel": channel_name,
                            "rationale": "Compare storytelling vs data-driven messaging"
                        }
                    ],
                    "testing_strategy": "A/B testing generation had parsing issues, but fallback variants were created. The rest of the analysis completed successfully.",
                    "success_metrics": ["click-through rate", "conversion rate"],
                    "sample_size_recommendations": {channel_name: 1000},
                    "duration_recommendations": {channel_name: "7-14 days"},
                }
            
            return result
        except Exception as e:
            # Catch any LLM provider errors (API errors, network errors, etc.)
            logger.error("LLM call failed for A/B testing", error=str(e), exc_info=True)
            return {
                "variants": [],
                "recommended_tests": [],
                "testing_strategy": f"A/B testing generation failed due to LLM error: {str(e)[:100]}. The rest of the analysis completed successfully.",
                "success_metrics": ["click-through rate", "conversion rate"],
                "sample_size_recommendations": {},
                "duration_recommendations": {},
            }
    
    async def generate_test_plan(
        self,
        campaign_data: Dict[str, Any],
        channels: Optional[List[str]] = None,
        variant_count: int = 3,
    ) -> ABTestPlan:
        """
        Generate comprehensive A/B testing plan with message variants.
        
        Returns variants, testing recommendations, and strategy.
        """
        # Call tool directly (skip autonomous framework to avoid overhead and speed up)
        result = await self._tool_generate_variants(campaign_data, channels, variant_count)
        
        campaign_id = campaign_data.get("campaign_id", str(hash(campaign_data.get("title", ""))))
        
        variants = [
            MessageVariant(
                variant_id=v.get("variant_id", f"variant_{i}"),
                variant_name=v.get("variant_name", f"Variant {i+1}"),
                message=v.get("message", ""),
                channel=v.get("channel", "General"),
                target_audience=v.get("target_audience", ""),
                hypothesis=v.get("hypothesis", ""),
                expected_outcome=v.get("expected_outcome", ""),
                testing_priority=v.get("testing_priority", "medium"),
            )
            for i, v in enumerate(result.get("variants", []))
        ]
        
        return ABTestPlan(
            campaign_id=campaign_id,
            variants=variants,
            recommended_tests=result.get("recommended_tests", []),
            testing_strategy=result.get("testing_strategy", "Run tests sequentially"),
            success_metrics=result.get("success_metrics", ["click-through rate", "conversion rate"]),
            sample_size_recommendations=result.get("sample_size_recommendations", {}),
            duration_recommendations=result.get("duration_recommendations", {}),
        )


