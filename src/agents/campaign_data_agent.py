"""
Campaign Data Acquisition Agent (Autonomous Agent)

An LLM-based autonomous agent that intelligently acquires and enriches campaign data.
Uses planning, reasoning, and tool-use to:
- Accept manual campaign data entry
- Import campaign data from JSON/CSV
- Enrich incomplete campaign data using LLM reasoning
- Discover similar campaigns using semantic understanding
- Generate realistic campaign data for demonstration
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import csv
import re

import structlog

from src.core.autonomous_agent import AutonomousAgent, Tool, Task, ReasoningStep, ReasoningType
from src.core.llm_service import get_llm_service
from src.data.synthetic import SyntheticDataGenerator

logger = structlog.get_logger()


@dataclass
class CampaignData:
    """Standardized campaign data structure."""
    campaign_id: str
    title: str
    description: str = ""
    category: str = ""
    goal_amount: float = 0.0
    raised_amount: float = 0.0
    donor_count: int = 0
    location: str = ""
    organizer_name: str = ""
    url: str = ""
    created_at: Optional[str] = None
    updates: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "goal_amount": self.goal_amount,
            "raised_amount": self.raised_amount,
            "donor_count": self.donor_count,
            "location": self.location,
            "organizer_name": self.organizer_name,
            "url": self.url,
            "created_at": self.created_at or datetime.now().isoformat(),
            "updates": self.updates,
        }


class CampaignDataAgent(AutonomousAgent[CampaignData]):
    """
    Autonomous agent for acquiring and enriching campaign data.
    
    This agent:
    1. Accepts campaign data from multiple sources (manual, JSON, CSV)
    2. Enriches incomplete data using LLM reasoning
    3. Discovers similar campaigns using semantic understanding
    4. Validates and normalizes campaign data
    5. Generates realistic campaign data for demonstration
    """
    
    SYSTEM_PROMPT = """You are an expert in campaign data acquisition and enrichment.
Your expertise includes:
- Understanding campaign data structures and requirements
- Enriching incomplete campaign information using reasoning
- Discovering similar campaigns through semantic understanding
- Validating and normalizing campaign data
- Generating realistic campaign scenarios for demonstration

You help acquire high-quality campaign data through multiple methods:
- Manual entry with intelligent validation
- Data import from structured formats (JSON, CSV)
- Semantic enrichment of incomplete data
- Discovery of similar campaigns based on themes and causes
- Generation of realistic demonstration data

You are thorough, accurate, and focused on acquiring complete, validated campaign data."""

    def __init__(self):
        super().__init__(
            agent_id="campaign_data_agent",
            name="Campaign Data Acquisition Agent",
            description="Autonomous agent for acquiring and enriching campaign data",
            system_prompt=self.SYSTEM_PROMPT,
        )
        
        # Data stores
        self._campaigns: Dict[str, CampaignData] = {}
        self._synthetic_generator = SyntheticDataGenerator(seed=42)
        
        # Register domain-specific tools
        for tool in self._get_domain_tools():
            self.register_tool(tool)
    
    def _get_domain_system_prompt(self) -> str:
        """Get domain-specific system prompt additions."""
        return """
## Domain Expertise: Campaign Data Acquisition

You specialize in:
1. **Data Validation**: Ensuring campaign data is complete, accurate, and properly structured
2. **Data Enrichment**: Using LLM reasoning to fill in missing information intelligently
3. **Semantic Discovery**: Finding similar campaigns through understanding themes and causes
4. **Data Import**: Handling multiple formats (JSON, CSV) and normalizing to standard structure
5. **Demo Generation**: Creating realistic campaign data for demonstration purposes

You are thorough, accurate, and focused on acquiring high-quality campaign data through ethical, agentic methods."""
    
    def _get_domain_tools(self) -> List[Tool]:
        return [
            Tool(
                name="validate_campaign_data",
                description="Validate and normalize campaign data structure",
                parameters={
                    "type": "object",
                    "properties": {
                        "campaign_data": {"type": "object", "description": "Campaign data to validate"},
                    },
                    "required": ["campaign_data"],
                },
                function=self._tool_validate_data,
            ),
            Tool(
                name="enrich_campaign_data",
                description="Enrich incomplete campaign data using LLM reasoning",
                parameters={
                    "type": "object",
                    "properties": {
                        "campaign_data": {"type": "object", "description": "Incomplete campaign data"},
                    },
                    "required": ["campaign_data"],
                },
                function=self._tool_enrich_data,
            ),
            Tool(
                name="discover_similar_campaigns",
                description="Discover similar campaigns using semantic understanding",
                parameters={
                    "type": "object",
                    "properties": {
                        "reference_campaign": {"type": "object", "description": "Campaign to find similar ones for"},
                        "limit": {"type": "integer", "description": "Maximum number of similar campaigns to find", "default": 5},
                    },
                    "required": ["reference_campaign"],
                },
                function=self._tool_discover_similar,
            ),
            Tool(
                name="import_from_json",
                description="Import campaign data from JSON format",
                parameters={
                    "type": "object",
                    "properties": {
                        "json_data": {"type": "string", "description": "JSON string or file path"},
                    },
                    "required": ["json_data"],
                },
                function=self._tool_import_json,
            ),
            Tool(
                name="import_from_csv",
                description="Import campaign data from CSV format",
                parameters={
                    "type": "object",
                    "properties": {
                        "csv_path": {"type": "string", "description": "Path to CSV file"},
                    },
                    "required": ["csv_path"],
                },
                function=self._tool_import_csv,
            ),
            Tool(
                name="generate_demo_campaigns",
                description="Generate realistic campaign data for demonstration",
                parameters={
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer", "description": "Number of campaigns to generate", "default": 5},
                        "categories": {"type": "array", "description": "Specific categories to generate", "items": {"type": "string"}},
                    },
                    "required": [],
                },
                function=self._tool_generate_demo,
            ),
            Tool(
                name="process_campaign_url",
                description="Process a campaign URL and extract/enrich campaign data using LLM reasoning",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Campaign URL (e.g., gofundme.com/f/campaign-name)"},
                    },
                    "required": ["url"],
                },
                function=self._tool_process_url,
            ),
        ]
    
    async def _tool_validate_data(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize campaign data."""
        validated = {
            "campaign_id": campaign_data.get("campaign_id") or f"camp_{hash(campaign_data.get('title', ''))}",
            "title": campaign_data.get("title", "").strip(),
            "description": campaign_data.get("description", "").strip(),
            "category": campaign_data.get("category", "").strip().lower(),
            "goal_amount": float(campaign_data.get("goal_amount", 0) or 0),
            "raised_amount": float(campaign_data.get("raised_amount", 0) or 0),
            "donor_count": int(campaign_data.get("donor_count", 0) or 0),
            "location": campaign_data.get("location", "").strip(),
            "organizer_name": campaign_data.get("organizer_name", "").strip(),
            "url": campaign_data.get("url", "").strip(),
            "created_at": campaign_data.get("created_at") or datetime.now().isoformat(),
            "updates": campaign_data.get("updates", []),
        }
        
        # Validation checks
        issues = []
        if not validated["title"]:
            issues.append("Missing title")
        if validated["goal_amount"] < 0:
            issues.append("Invalid goal amount")
        if validated["raised_amount"] < 0:
            issues.append("Invalid raised amount")
        if validated["raised_amount"] > validated["goal_amount"] * 1.1:  # Allow 10% over
            issues.append("Raised amount exceeds goal significantly")
        
        return {
            "validated_data": validated,
            "is_valid": len(issues) == 0,
            "issues": issues,
        }
    
    async def _tool_enrich_data(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich incomplete campaign data using LLM reasoning."""
        title = campaign_data.get("title", "")
        description = campaign_data.get("description", "")
        category = campaign_data.get("category", "")
        
        # Use LLM to enrich missing fields
        prompt = f"""Given this campaign information, enrich it with missing details:

Title: {title}
Description: {description[:500] if description else 'Not provided'}
Category: {category or 'Not specified'}

Provide enriched information in JSON format:
{{
    "suggested_category": "most appropriate category",
    "estimated_goal": "reasonable goal amount if not provided",
    "key_themes": ["theme1", "theme2"],
    "urgency_indicators": ["indicator1", "indicator2"],
    "suggested_description": "enhanced description if original is short",
    "beneficiary_type": "individual/family/organization/community"
}}"""
        
        llm = get_llm_service()
        response = await llm._provider.complete(
            prompt=prompt,
            system_prompt="You are an expert at understanding charitable campaigns and enriching incomplete data.",
            temperature=0.3,
        )
        
        enriched = campaign_data.copy()
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                llm_data = json.loads(json_match.group())
                
                if not enriched.get("category") and llm_data.get("suggested_category"):
                    enriched["category"] = llm_data["suggested_category"]
                
                if not enriched.get("description") or len(enriched.get("description", "")) < 50:
                    if llm_data.get("suggested_description"):
                        enriched["description"] = llm_data["suggested_description"]
                
                if enriched.get("goal_amount", 0) == 0 and llm_data.get("estimated_goal"):
                    try:
                        goal_str = str(llm_data["estimated_goal"]).replace("$", "").replace(",", "")
                        enriched["goal_amount"] = float(goal_str)
                    except:
                        pass
        except:
            pass
        
        return {
            "enriched_data": enriched,
            "enrichments_applied": list(set(enriched.keys()) - set(campaign_data.keys())),
        }
    
    async def _tool_discover_similar(self, reference_campaign: Dict[str, Any], limit: int = 5) -> Dict[str, Any]:
        """Discover similar campaigns using semantic understanding."""
        title = reference_campaign.get("title", "")
        description = reference_campaign.get("description", "")
        category = reference_campaign.get("category", "")
        
        # Use LLM to understand what makes this campaign unique
        prompt = f"""Analyze this campaign and identify what makes it unique:

Title: {title}
Description: {description[:500] if description else 'Not provided'}
Category: {category or 'Not specified'}

Identify:
1. Key themes and causes
2. Beneficiary characteristics
3. Urgency factors
4. Geographic context

Then suggest what similar campaigns would look like. Provide 3-5 similar campaign concepts in JSON:
{{
    "key_characteristics": ["char1", "char2"],
    "similar_campaigns": [
        {{
            "title": "similar campaign title",
            "description": "brief description",
            "category": "category",
            "why_similar": "explanation"
        }}
    ]
}}"""
        
        llm = get_llm_service()
        response = await llm._provider.complete(
            prompt=prompt,
            system_prompt="You are an expert at understanding campaign themes and finding similar causes.",
            temperature=0.5,
        )
        
        similar_campaigns = []
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                llm_data = json.loads(json_match.group())
                similar_campaigns = llm_data.get("similar_campaigns", [])[:limit]
                
                # Convert to CampaignData format
                for i, camp in enumerate(similar_campaigns):
                    camp["campaign_id"] = f"similar_{hash(camp.get('title', ''))}"
                    camp["goal_amount"] = camp.get("goal_amount", reference_campaign.get("goal_amount", 5000))
                    camp["raised_amount"] = camp.get("raised_amount", 0)
                    camp["donor_count"] = camp.get("donor_count", 0)
                    # Don't generate fake URLs - only use if provided
                    if "url" not in camp:
                        camp["url"] = ""
        except:
            # Fallback: generate synthetic similar campaigns
            similar_campaigns = []
            for _ in range(min(limit, 3)):
                synthetic = self._synthetic_generator.generate_campaign()
                if category:
                    synthetic["category"] = category
                # Don't generate fake URLs - only use if provided
                if "url" not in synthetic:
                    synthetic["url"] = ""
                similar_campaigns.append(synthetic)
        
        return {
            "similar_campaigns": similar_campaigns,
            "count": len(similar_campaigns),
        }
    
    async def _tool_import_json(self, json_data: str) -> Dict[str, Any]:
        """Import campaign data from JSON."""
        try:
            # Try to parse as JSON string first
            if json_data.strip().startswith('{') or json_data.strip().startswith('['):
                data = json.loads(json_data)
            else:
                # Assume it's a file path
                with open(json_data, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            # Handle both single object and array
            if isinstance(data, list):
                campaigns = data
            else:
                campaigns = [data]
            
            imported = []
            for camp in campaigns:
                validated = await self._tool_validate_data(camp)
                if validated["is_valid"]:
                    imported.append(validated["validated_data"])
            
            return {
                "imported_count": len(imported),
                "campaigns": imported,
            }
        except Exception as e:
            return {
                "error": str(e),
                "imported_count": 0,
                "campaigns": [],
            }
    
    async def _tool_import_csv(self, csv_path: str) -> Dict[str, Any]:
        """Import campaign data from CSV."""
        try:
            campaigns = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Map CSV columns to campaign data structure
                    campaign = {
                        "title": row.get("title", ""),
                        "description": row.get("description", ""),
                        "category": row.get("category", ""),
                        "goal_amount": float(row.get("goal_amount", 0) or 0),
                        "raised_amount": float(row.get("raised_amount", 0) or 0),
                        "donor_count": int(row.get("donor_count", 0) or 0),
                        "location": row.get("location", ""),
                        "organizer_name": row.get("organizer_name", ""),
                    }
                    validated = await self._tool_validate_data(campaign)
                    if validated["is_valid"]:
                        campaigns.append(validated["validated_data"])
            
            return {
                "imported_count": len(campaigns),
                "campaigns": campaigns,
            }
        except Exception as e:
            return {
                "error": str(e),
                "imported_count": 0,
                "campaigns": [],
            }
    
    async def _tool_generate_demo(self, count: int = 5, categories: List[str] = None) -> Dict[str, Any]:
        """Generate realistic campaign data for demonstration."""
        campaigns = []
        for _ in range(count):
            campaign = self._synthetic_generator.generate_campaign()
            if categories:
                # Override category if specified
                import random
                campaign["category"] = random.choice(categories)
            campaigns.append(campaign)
        
        return {
            "generated_count": len(campaigns),
            "campaigns": campaigns,
        }
    
    async def _tool_process_url(self, url: str) -> Dict[str, Any]:
        """Fetch and parse real campaign data from a GoFundMe URL."""
        import re
        import structlog
        import httpx
        from bs4 import BeautifulSoup
        
        logger = structlog.get_logger()
        
        logger.info("Fetching campaign data from URL", url=url)
        
        # Validate URL
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")
        
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        campaign_data = {
            "campaign_id": "",
            "url": url,
            "title": "",
            "description": "",
            "category": "",
            "goal_amount": 0.0,
            "raised_amount": 0.0,
            "donor_count": 0,
            "location": "",
            "organizer_name": "",
        }
        
        try:
            # Fetch the page (reduced timeout)
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                html_content = response.text
            
            logger.info("Successfully fetched page", status_code=response.status_code, content_length=len(html_content))
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Extract campaign ID from URL
            slug_match = re.search(r'/f/([^/?]+)', url)
            if slug_match:
                campaign_data["campaign_id"] = slug_match.group(1)
            else:
                campaign_data["campaign_id"] = url.split('/')[-1].split('?')[0]
            
            # Extract title - try multiple selectors
            title_selectors = [
                'h1[class*="title"]',
                'h1.campaign-title',
                'h1',
                '[data-testid="campaign-title"]',
                'meta[property="og:title"]',
            ]
            
            for selector in title_selectors:
                element = soup.select_one(selector)
                if element:
                    title = element.get_text(strip=True) or element.get('content', '')
                    if title and len(title) > 3:
                        campaign_data["title"] = title
                        break
            
            # Extract description - try multiple selectors
            desc_selectors = [
                '[class*="description"]',
                '[data-testid="campaign-description"]',
                'meta[property="og:description"]',
                'meta[name="description"]',
            ]
            
            for selector in desc_selectors:
                element = soup.select_one(selector)
                if element:
                    desc = element.get_text(strip=True) or element.get('content', '')
                    if desc and len(desc) > 10:
                        campaign_data["description"] = desc[:1000]  # Limit length
                        break
            
            # Extract goal and raised amounts
            # Look for patterns like "$50,000" or "50000"
            amount_patterns = [
                r'\$?([\d,]+)',
                r'([\d,]+)\s*(?:USD|dollars?)',
            ]
            
            # Try to find goal amount
            goal_text = soup.find(string=re.compile(r'goal|target|raising', re.I))
            if goal_text:
                for pattern in amount_patterns:
                    match = re.search(pattern, goal_text)
                    if match:
                        try:
                            amount_str = match.group(1).replace(',', '')
                            campaign_data["goal_amount"] = float(amount_str)
                            break
                        except ValueError:
                            continue
            
            # Try meta tags for goal
            goal_meta = soup.find('meta', {'property': 'gofundme:goal'}) or soup.find('meta', {'name': 'goal'})
            if goal_meta:
                try:
                    campaign_data["goal_amount"] = float(goal_meta.get('content', '0').replace(',', ''))
                except (ValueError, AttributeError):
                    pass
            
            # Try to find raised amount
            raised_text = soup.find(string=re.compile(r'raised|donated|collected', re.I))
            if raised_text:
                for pattern in amount_patterns:
                    match = re.search(pattern, raised_text)
                    if match:
                        try:
                            amount_str = match.group(1).replace(',', '')
                            campaign_data["raised_amount"] = float(amount_str)
                            break
                        except ValueError:
                            continue
            
            # Try meta tags for raised
            raised_meta = soup.find('meta', {'property': 'gofundme:raised'}) or soup.find('meta', {'name': 'raised'})
            if raised_meta:
                try:
                    campaign_data["raised_amount"] = float(raised_meta.get('content', '0').replace(',', ''))
                except (ValueError, AttributeError):
                    pass
            
            # Extract donor count
            donor_text = soup.find(string=re.compile(r'(\d+)\s*(?:donor|supporter|contribution)', re.I))
            if donor_text:
                match = re.search(r'(\d+)', donor_text)
                if match:
                    try:
                        campaign_data["donor_count"] = int(match.group(1))
                    except ValueError:
                        pass
            
            # Extract location
            location_selectors = [
                '[class*="location"]',
                '[data-testid="location"]',
                'meta[property="gofundme:location"]',
            ]
            
            for selector in location_selectors:
                element = soup.select_one(selector)
                if element:
                    location = element.get_text(strip=True) or element.get('content', '')
                    if location:
                        campaign_data["location"] = location
                        break
            
            # Extract organizer name
            organizer_selectors = [
                '[class*="organizer"]',
                '[class*="creator"]',
                '[data-testid="organizer"]',
            ]
            
            for selector in organizer_selectors:
                element = soup.select_one(selector)
                if element:
                    organizer = element.get_text(strip=True)
                    if organizer:
                        campaign_data["organizer_name"] = organizer
                        break
            
            # Use LLM to infer category if not found
            if not campaign_data.get("category"):
                llm = get_llm_service()
                category_prompt = f"""Based on this campaign information, determine the category:

Title: {campaign_data.get('title', '')}
Description: {campaign_data.get('description', '')[:200]}

Categories: medical, education, community, emergency, animals, other

Return only the category name (one word)."""
                
                try:
                    category_response = await llm._provider.complete(
                        prompt=category_prompt,
                        system_prompt="You are an expert at categorizing campaigns. Return only the category name.",
                        temperature=0.2,
                    )
                    category = category_response.strip().lower()
                    if category in ['medical', 'education', 'community', 'emergency', 'animals', 'other']:
                        campaign_data["category"] = category
                    else:
                        campaign_data["category"] = "other"
                except Exception as e:
                    logger.warning("Failed to infer category", error=str(e))
                    campaign_data["category"] = "other"
            
            logger.info("Extracted campaign data", 
                       title=campaign_data.get("title"),
                       goal=campaign_data.get("goal_amount"),
                       raised=campaign_data.get("raised_amount"))
            
        except httpx.HTTPError as e:
            logger.error("HTTP error fetching URL", error=str(e), url=url)
            raise ValueError(f"Failed to fetch URL: {str(e)}")
        except Exception as e:
            logger.error("Error processing URL", error=str(e), url=url)
            raise ValueError(f"Error processing URL: {str(e)}")
        
        # Validate and enrich
        logger.info("Validating and enriching campaign data", campaign_title=campaign_data.get("title"))
        validated = await self._tool_validate_data(campaign_data)
        if validated["is_valid"]:
            enriched = await self._tool_enrich_data(validated["validated_data"])
            result = {
                "campaign": enriched["enriched_data"],
                "source": "url",
                "url": url,
            }
            logger.info("Campaign data processed successfully", 
                       title=enriched["enriched_data"].get("title"),
                       url=url)
            return result
        else:
            result = {
                "campaign": campaign_data,
                "source": "url",
                "url": url,
                "validation_issues": validated["issues"],
            }
            logger.warning("Campaign data has validation issues", 
                          issues=validated["issues"],
                          url=url)
            return result
    
    # ==================== Public API ====================
    
    async def acquire_campaigns(
        self,
        source: str,
        data: Any = None
    ) -> List[Dict[str, Any]]:
        """
        Autonomously acquire campaign data from various sources.
        
        Args:
            source: "manual", "json", "csv", "demo", or "similar"
            data: Source-specific data (dict for manual, string for json/csv, dict for similar)
        """
        goal = f"""Acquire campaign data from source: {source}
        
Available methods:
- manual: Validate and enrich manually entered campaign data
- json: Import from JSON format
- csv: Import from CSV file
- demo: Generate realistic demonstration campaigns
- similar: Discover similar campaigns to a reference

Process the data, validate it, enrich if needed, and return complete campaign data."""
        
        context = {
            "source": source,
            "data": data,
        }
        
        # Run autonomous acquisition
        result = await self.run(goal, context, max_iterations=10)
        
        # Extract campaigns from tool results
        campaigns = []
        for task in self._memory.tasks.values():
            if task.status.value == "completed" and task.result:
                if "campaigns" in task.result:
                    campaigns.extend(task.result["campaigns"])
                elif "validated_data" in task.result:
                    campaigns.append(task.result["validated_data"])
                elif "enriched_data" in task.result:
                    campaigns.append(task.result["enriched_data"])
        
        return campaigns
    
    def get_campaign(self, campaign_id: str) -> Optional[CampaignData]:
        """Get a stored campaign."""
        return self._campaigns.get(campaign_id)
    
    def store_campaign(self, campaign_data: Dict[str, Any]) -> CampaignData:
        """Store a campaign."""
        campaign = CampaignData(**campaign_data)
        self._campaigns[campaign.campaign_id] = campaign
        return campaign

