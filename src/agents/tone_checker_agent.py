"""
Tone Checker Agent

Analyzes campaign messaging tone to ensure it's appropriate, empathetic, and effective.
Checks for:
- Empathy and sensitivity
- Authenticity
- Clarity and accessibility
- Appropriate urgency without manipulation
- Respectful language
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import structlog

from src.core.autonomous_agent import AutonomousAgent, Tool
from src.core.llm_service import get_llm_service

logger = structlog.get_logger()


@dataclass
class ToneAnalysis:
    """Tone analysis results."""
    overall_tone_score: float  # 0-1
    empathy_score: float
    authenticity_score: float
    clarity_score: float
    urgency_appropriateness: float
    respectful_language_score: float
    tone_issues: List[str]
    tone_strengths: List[str]
    improvement_suggestions: List[str]
    sensitive_phrases: List[str]  # Phrases that may be insensitive
    recommended_changes: List[str]


class ToneCheckerAgent(AutonomousAgent[ToneAnalysis]):
    """
    Autonomous agent for analyzing campaign messaging tone.
    
    Ensures campaigns use:
    - Empathetic and sensitive language
    - Authentic storytelling
    - Clear and accessible communication
    - Appropriate urgency without manipulation
    - Respectful language that honors the cause
    """
    
    SYSTEM_PROMPT = """You are an expert in empathetic communication and campaign messaging.
Your expertise includes:
- Identifying insensitive or inappropriate language
- Ensuring empathetic and respectful tone
- Detecting manipulative or exploitative language
- Recommending authentic and genuine messaging
- Ensuring clarity and accessibility

You help campaigns communicate with:
- Empathy and sensitivity
- Authenticity and genuineness
- Clarity and accessibility
- Appropriate urgency without manipulation
- Respectful language that honors the cause

You are thorough, empathetic, and focused on ensuring campaigns communicate respectfully and effectively."""
    
    def __init__(self):
        super().__init__(
            agent_id="tone_checker_agent",
            name="Tone Checker Agent",
            description="Analyzes campaign messaging tone for empathy, authenticity, and appropriateness",
            system_prompt=self.SYSTEM_PROMPT,
        )
        
        # Register domain-specific tools
        for tool in self._get_domain_tools():
            self.register_tool(tool)
    
    def _get_domain_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT
    
    def _get_domain_tools(self) -> List[Tool]:
        """Get domain-specific tools for tone checking."""
        return [
            Tool(
                name="check_tone",
                description="Analyze the tone of campaign messaging (title, description, updates) for empathy, authenticity, clarity, and appropriateness",
                parameters={
                    "type": "object",
                    "properties": {
                        "campaign_data": {
                            "type": "object",
                            "description": "Campaign data including title, description, and updates",
                        },
                    },
                    "required": ["campaign_data"],
                },
                function=self._tool_check_tone,
            ),
        ]
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Robustly parse JSON from LLM response."""
        if not response or not response.strip():
            raise ValueError("Empty response from LLM")
        
        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON from markdown code blocks
        import re
        json_block_patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'`(\{.*?\})`',
        ]
        
        for pattern in json_block_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
        
        # Strategy 3: Find JSON object using regex
        json_object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_object_pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                cleaned = match.strip()
                cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and len(parsed) > 0:
                    return parsed
            except json.JSONDecodeError:
                continue
        
        # Strategy 4: Extract between first { and last }
        first_brace = response.find('{')
        last_brace = response.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = response[first_brace:last_brace + 1]
            try:
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not parse JSON from LLM response")
    
    async def _tool_check_tone(
        self,
        campaign_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze campaign messaging tone for empathy, authenticity, and appropriateness."""
        title = campaign_data.get("title", "")
        description = campaign_data.get("description", "")
        updates = campaign_data.get("updates", [])
        
        # Combine all text for analysis
        all_text = f"{title}\n\n{description}"
        if updates:
            update_texts = [u.get("text", u.get("content", "")) for u in updates if isinstance(u, dict)]
            all_text += "\n\nUpdates:\n" + "\n".join(update_texts)
        
        llm = get_llm_service()
        
        prompt = f"""Analyze the tone and messaging of this campaign for empathy, authenticity, clarity, and appropriateness:

Campaign Text:
Title: {title}
Description: {description[:1000]}

Analyze across these dimensions (0-1 scale):
1. Empathy: How empathetic and sensitive is the language? Does it honor the person/cause?
2. Authenticity: How genuine and authentic does the messaging feel? Is it manipulative?
3. Clarity: How clear and accessible is the language? Can people easily understand?
4. Urgency Appropriateness: Is urgency communicated appropriately without manipulation?
5. Respectful Language: Is the language respectful and appropriate for the cause?

Also identify:
- Tone issues: Any insensitive, manipulative, or inappropriate language
- Tone strengths: What's working well in the messaging
- Sensitive phrases: Specific phrases that may be insensitive or inappropriate
- Improvement suggestions: How to improve the tone while maintaining authenticity

Return JSON:
{{
  "empathy_score": 0.8,
  "authenticity_score": 0.7,
  "clarity_score": 0.9,
  "urgency_appropriateness": 0.75,
  "respectful_language_score": 0.85,
  "tone_issues": ["..."],
  "tone_strengths": ["..."],
  "improvement_suggestions": ["..."],
  "sensitive_phrases": ["..."],
  "recommended_changes": ["..."]
}}"""
        
        response = await llm._provider.complete(
            prompt=prompt,
            system_prompt="You are an expert in empathetic communication. Be specific and actionable. Return ONLY valid JSON, no markdown or extra text.",
            temperature=0.3,
        )
        
        try:
            analysis = self._parse_json_response(response)
        except (ValueError, json.JSONDecodeError) as e:
            logger.error("Failed to parse tone analysis JSON", error=str(e), response_preview=response[:200])
            # Retry with a more explicit prompt
            retry_prompt = prompt + "\n\nCRITICAL: Return ONLY valid JSON. No markdown, no code blocks, no explanations. Just the JSON object."
            retry_response = await llm._provider.complete(
                prompt=retry_prompt,
                system_prompt="You are an expert in empathetic communication. Return ONLY valid JSON, no markdown or extra text.",
                temperature=0.2,
            )
            try:
                analysis = self._parse_json_response(retry_response)
            except (ValueError, json.JSONDecodeError) as e2:
                logger.error("Retry also failed for tone analysis", error=str(e2))
                raise ValueError(f"Could not parse tone analysis after retry: {str(e2)}")
        
        # Calculate overall tone score (weighted average)
        scores = [
            analysis.get("empathy_score", 0.5),
            analysis.get("authenticity_score", 0.5),
            analysis.get("clarity_score", 0.5),
            analysis.get("urgency_appropriateness", 0.5),
            analysis.get("respectful_language_score", 0.5),
        ]
        overall_score = sum(scores) / len(scores) if scores else 0.5
        
        return {
            "overall_tone_score": round(overall_score, 3),
            "empathy_score": analysis.get("empathy_score", 0.5),
            "authenticity_score": analysis.get("authenticity_score", 0.5),
            "clarity_score": analysis.get("clarity_score", 0.5),
            "urgency_appropriateness": analysis.get("urgency_appropriateness", 0.5),
            "respectful_language_score": analysis.get("respectful_language_score", 0.5),
            "tone_issues": analysis.get("tone_issues", []),
            "tone_strengths": analysis.get("tone_strengths", []),
            "improvement_suggestions": analysis.get("improvement_suggestions", []),
            "sensitive_phrases": analysis.get("sensitive_phrases", []),
            "recommended_changes": analysis.get("recommended_changes", []),
        }
    
    async def analyze_tone(
        self,
        campaign_data: Dict[str, Any],
    ) -> ToneAnalysis:
        """
        Analyze campaign tone and return comprehensive tone analysis.
        
        Returns tone scores, issues, strengths, and improvement suggestions.
        """
        result = await self._tool_check_tone(campaign_data)
        
        return ToneAnalysis(
            overall_tone_score=result["overall_tone_score"],
            empathy_score=result["empathy_score"],
            authenticity_score=result["authenticity_score"],
            clarity_score=result["clarity_score"],
            urgency_appropriateness=result["urgency_appropriateness"],
            respectful_language_score=result["respectful_language_score"],
            tone_issues=result["tone_issues"],
            tone_strengths=result["tone_strengths"],
            improvement_suggestions=result["improvement_suggestions"],
            sensitive_phrases=result["sensitive_phrases"],
            recommended_changes=result["recommended_changes"],
        )





