"""
Agent 3: Local Community Discovery Agent (Autonomous Agent)

An LLM-based autonomous agent that surfaces campaigns based on social proximity.
Uses planning, reasoning, and tool-use to:
- Map social connections and networks
- Find campaigns within degrees of separation
- Identify workplace giving opportunities
- Discover community-based campaigns
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import math
import json
import re

import structlog

from src.core.autonomous_agent import AutonomousAgent, Tool, Task, ReasoningStep, ReasoningType
from src.core.llm_service import get_llm_service
from src.core.agent_collaboration import get_collaborator

logger = structlog.get_logger()


class ConnectionType(str, Enum):
    """Types of social connections."""
    FRIEND = "friend"
    FAMILY = "family"
    COLLEAGUE = "colleague"
    CLASSMATE = "classmate"
    NEIGHBOR = "neighbor"
    CONGREGATION = "congregation"
    CLUB_MEMBER = "club_member"
    ALUMNI = "alumni"
    PROFESSIONAL_NETWORK = "professional_network"


class ProximityType(str, Enum):
    """Types of proximity to campaigns."""
    DIRECT_CONNECTION = "direct_connection"
    SECOND_DEGREE = "second_degree"
    SAME_EMPLOYER = "same_employer"
    SAME_SCHOOL = "same_school"
    SAME_NEIGHBORHOOD = "same_neighborhood"
    SAME_CITY = "same_city"
    SAME_REGION = "same_region"
    SAME_ORGANIZATION = "same_organization"


@dataclass
class SocialConnection:
    """Represents a connection between two users."""
    user_id: str
    connected_user_id: str
    connection_type: ConnectionType
    strength: float = 0.5
    is_mutual: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """User profile for community discovery."""
    user_id: str
    location: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: str = "US"
    coordinates: Optional[Tuple[float, float]] = None
    employer: Optional[str] = None
    employer_id: Optional[str] = None
    job_title: Optional[str] = None
    schools: List[str] = field(default_factory=list)
    graduation_years: Dict[str, int] = field(default_factory=dict)
    religious_affiliation: Optional[str] = None
    congregation: Optional[str] = None
    clubs_organizations: List[str] = field(default_factory=list)
    connections: List[SocialConnection] = field(default_factory=list)


@dataclass
class CampaignProximity:
    """Represents a campaign's proximity to a user."""
    campaign_id: str
    campaign_title: str
    proximity_types: List[ProximityType] = field(default_factory=list)
    proximity_score: float = 0.0
    connection_path: List[str] = field(default_factory=list)
    shared_context: Dict[str, Any] = field(default_factory=dict)
    narrative: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "campaign_title": self.campaign_title,
            "proximity_types": [p.value for p in self.proximity_types],
            "proximity_score": self.proximity_score,
            "connection_path": self.connection_path,
            "shared_context": self.shared_context,
            "narrative": self.narrative,
        }


@dataclass
class CommunityCluster:
    """A cluster of related campaigns in a community."""
    cluster_id: str
    name: str
    description: str
    campaigns: List[str] = field(default_factory=list)
    members: List[str] = field(default_factory=list)
    total_raised: float = 0.0
    total_goal: float = 0.0
    community_type: str = "general"
    llm_summary: str = ""


class CommunityDiscoveryAgent(AutonomousAgent[CampaignProximity]):
    """
    Autonomous agent for discovering community-connected campaigns.
    
    This agent:
    1. Maps social networks and connections
    2. Reasons about proximity and relevance
    3. Finds campaigns within degrees of separation
    4. Generates personalized connection narratives
    5. Identifies community giving opportunities
    """
    
    SYSTEM_PROMPT = """You are an expert in social network analysis and community engagement.
Your expertise includes:
- Understanding social connections and their strength
- Mapping degrees of separation in networks
- Identifying meaningful community ties
- Understanding workplace and alumni networks
- Geographic and cultural community dynamics

You help donors discover campaigns connected to their social circles, making giving
more personal and meaningful through community connections."""

    def __init__(self):
        super().__init__(
            agent_id="community_discovery",
            name="Local Community Discovery Agent",
            description="Autonomous agent for discovering community-connected campaigns",
            system_prompt=self.SYSTEM_PROMPT,
        )
        
        # Data stores
        self._user_profiles: Dict[str, UserProfile] = {}
        self._campaigns: Dict[str, Dict[str, Any]] = {}
        self._social_graph: Dict[str, Set[str]] = defaultdict(set)
        self._community_clusters: Dict[str, CommunityCluster] = {}
        
        # Register domain-specific tools
        for tool in self._get_domain_tools():
            self.register_tool(tool)
        
        # Initialize collaborator for A2A communication
        self._collaborator = get_collaborator(self.agent_id)
        
        # Register A2A handler
        self.register_handler("discover_communities", self._handle_discover_communities)
    
    def _get_domain_system_prompt(self) -> str:
        return """
## Domain Expertise: Community Discovery

You specialize in:

1. **Social Graph Analysis**: Understanding connection networks
   - Direct connections (1 degree)
   - Friends of friends (2 degrees)
   - Shared affiliations (employer, school, etc.)

2. **Proximity Scoring**: Evaluating campaign relevance
   - Connection strength
   - Shared context
   - Geographic closeness
   - Affiliation overlap

3. **Narrative Generation**: Creating connection stories
   - "Your colleague Sarah is raising funds for..."
   - "Three people from your company donated to..."
   - "A fellow Stanford alum needs help with..."

4. **Community Clustering**: Identifying giving communities
   - Workplace giving campaigns
   - School/alumni initiatives
   - Neighborhood projects
   - Religious community efforts
"""
    
    def _get_domain_tools(self) -> List[Tool]:
        return [
            Tool(
                name="map_user_network",
                description="Map a user's social network and connections",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "max_depth": {"type": "integer", "default": 2},
                    },
                    "required": ["user_id"],
                },
                function=self._tool_map_network,
            ),
            Tool(
                name="find_connected_campaigns",
                description="Find campaigns connected to a user's network",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "max_degree": {"type": "integer", "default": 2},
                    },
                    "required": ["user_id"],
                },
                function=self._tool_find_connected,
            ),
            Tool(
                name="calculate_proximity",
                description="Calculate proximity score between user and campaign",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "campaign_id": {"type": "string"},
                    },
                    "required": ["user_id", "campaign_id"],
                },
                function=self._tool_calculate_proximity,
            ),
            Tool(
                name="generate_connection_narrative",
                description="Generate a personalized narrative about the connection",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "campaign_id": {"type": "string"},
                        "proximity": {"type": "object"},
                    },
                    "required": ["user_id", "campaign_id", "proximity"],
                },
                function=self._tool_generate_narrative,
            ),
            Tool(
                name="find_workplace_campaigns",
                description="Find campaigns related to user's workplace",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                    },
                    "required": ["user_id"],
                },
                function=self._tool_find_workplace,
            ),
            Tool(
                name="find_alumni_campaigns",
                description="Find campaigns related to user's schools/alumni network",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                    },
                    "required": ["user_id"],
                },
                function=self._tool_find_alumni,
            ),
            Tool(
                name="find_local_campaigns",
                description="Find campaigns in user's geographic area",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "radius_miles": {"type": "number", "default": 25},
                    },
                    "required": ["user_id"],
                },
                function=self._tool_find_local,
            ),
            Tool(
                name="cluster_community_campaigns",
                description="Identify clusters of related campaigns in communities",
                parameters={
                    "type": "object",
                    "properties": {
                        "community_type": {"type": "string"},
                        "community_id": {"type": "string"},
                    },
                    "required": ["community_type"],
                },
                function=self._tool_cluster_campaigns,
            ),
        ]
    
    # ==================== Tool Implementations ====================
    
    async def _tool_map_network(
        self,
        user_id: str,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """Map a user's social network."""
        profile = self._user_profiles.get(user_id)
        if not profile:
            return {"error": "User not found", "network": {}}
        
        network = {
            "user_id": user_id,
            "direct_connections": [],
            "second_degree": [],
            "affiliations": {
                "employer": profile.employer,
                "schools": profile.schools,
                "organizations": profile.clubs_organizations,
            },
        }
        
        # Direct connections
        for conn in profile.connections:
            network["direct_connections"].append({
                "user_id": conn.connected_user_id,
                "type": conn.connection_type.value,
                "strength": conn.strength,
            })
        
        # Second degree (if max_depth >= 2)
        if max_depth >= 2:
            seen = {user_id}
            for conn in profile.connections:
                seen.add(conn.connected_user_id)
            
            for conn in profile.connections:
                friend_profile = self._user_profiles.get(conn.connected_user_id)
                if friend_profile:
                    for friend_conn in friend_profile.connections:
                        if friend_conn.connected_user_id not in seen:
                            network["second_degree"].append({
                                "user_id": friend_conn.connected_user_id,
                                "through": conn.connected_user_id,
                                "path_strength": conn.strength * friend_conn.strength,
                            })
                            seen.add(friend_conn.connected_user_id)
        
        return network
    
    async def _tool_find_connected(
        self,
        user_id: str,
        max_degree: int = 2
    ) -> Dict[str, Any]:
        """Find campaigns connected to user's network."""
        network = await self._tool_map_network(user_id, max_degree)
        
        connected_campaigns = []
        
        # Check direct connections
        for conn in network.get("direct_connections", []):
            for campaign_id, campaign in self._campaigns.items():
                if campaign.get("organizer_id") == conn["user_id"]:
                    connected_campaigns.append({
                        "campaign_id": campaign_id,
                        "title": campaign.get("title", ""),
                        "degree": 1,
                        "connection": conn,
                    })
        
        # Check second degree
        for conn in network.get("second_degree", []):
            for campaign_id, campaign in self._campaigns.items():
                if campaign.get("organizer_id") == conn["user_id"]:
                    connected_campaigns.append({
                        "campaign_id": campaign_id,
                        "title": campaign.get("title", ""),
                        "degree": 2,
                        "through": conn["through"],
                    })
        
        return {
            "user_id": user_id,
            "connected_campaigns": connected_campaigns,
            "count": len(connected_campaigns),
        }
    
    async def _tool_calculate_proximity(
        self,
        user_id: str,
        campaign_id: str
    ) -> Dict[str, Any]:
        """Calculate proximity score between user and campaign."""
        profile = self._user_profiles.get(user_id)
        campaign = self._campaigns.get(campaign_id)
        
        if not profile or not campaign:
            return {"error": "User or campaign not found", "score": 0}
        
        proximity_types = []
        scores = []
        shared_context = {}
        
        # Check direct connection
        organizer_id = campaign.get("organizer_id")
        for conn in profile.connections:
            if conn.connected_user_id == organizer_id:
                proximity_types.append(ProximityType.DIRECT_CONNECTION)
                scores.append(0.9 * conn.strength)
                shared_context["connection_type"] = conn.connection_type.value
                break
        
        # Check second degree
        if ProximityType.DIRECT_CONNECTION not in proximity_types:
            for conn in profile.connections:
                friend_profile = self._user_profiles.get(conn.connected_user_id)
                if friend_profile:
                    for friend_conn in friend_profile.connections:
                        if friend_conn.connected_user_id == organizer_id:
                            proximity_types.append(ProximityType.SECOND_DEGREE)
                            scores.append(0.5 * conn.strength * friend_conn.strength)
                            shared_context["through"] = conn.connected_user_id
                            break
        
        # Check employer
        if profile.employer and profile.employer == campaign.get("organizer_employer"):
            proximity_types.append(ProximityType.SAME_EMPLOYER)
            scores.append(0.6)
            shared_context["employer"] = profile.employer
        
        # Check schools
        campaign_schools = campaign.get("organizer_schools", [])
        common_schools = set(profile.schools) & set(campaign_schools)
        if common_schools:
            proximity_types.append(ProximityType.SAME_SCHOOL)
            scores.append(0.5)
            shared_context["schools"] = list(common_schools)
        
        # Check geographic proximity
        if profile.city and profile.city == campaign.get("city"):
            proximity_types.append(ProximityType.SAME_CITY)
            scores.append(0.4)
            shared_context["city"] = profile.city
        elif profile.state and profile.state == campaign.get("state"):
            proximity_types.append(ProximityType.SAME_REGION)
            scores.append(0.2)
            shared_context["state"] = profile.state
        
        total_score = min(1.0, sum(scores)) if scores else 0.0
        
        return {
            "user_id": user_id,
            "campaign_id": campaign_id,
            "proximity_types": [p.value for p in proximity_types],
            "proximity_score": round(total_score, 3),
            "shared_context": shared_context,
        }
    
    async def _tool_generate_narrative(
        self,
        user_id: str,
        campaign_id: str,
        proximity: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a personalized connection narrative using LLM."""
        campaign = self._campaigns.get(campaign_id, {})
        profile = self._user_profiles.get(user_id)
        
        prompt = f"""Generate a brief, personal narrative (1-2 sentences) explaining the connection between this donor and campaign.

Donor Info:
- Location: {profile.city if profile else 'Unknown'}, {profile.state if profile else ''}
- Employer: {profile.employer if profile else 'Unknown'}

Campaign: {campaign.get('title', 'Unknown')}
Campaign Location: {campaign.get('city', '')}, {campaign.get('state', '')}

Connection Details:
{json.dumps(proximity, indent=2)}

The narrative should feel personal and highlight the meaningful connection.
Examples:
- "Your colleague at Google, Sarah, is raising funds for..."
- "Someone from your Stanford network needs help with..."
- "A neighbor in your community is supporting..."
"""

        narrative = await self._llm._provider.complete(
            prompt=prompt,
            system_prompt="You write warm, personal connection narratives for a giving platform.",
            temperature=0.7,
        )
        
        return {
            "narrative": narrative.strip(),
            "proximity": proximity,
        }
    
    async def _tool_find_workplace(self, user_id: str) -> Dict[str, Any]:
        """Find campaigns related to user's workplace."""
        profile = self._user_profiles.get(user_id)
        if not profile or not profile.employer:
            return {"campaigns": [], "message": "No employer information"}
        
        workplace_campaigns = []
        
        for campaign_id, campaign in self._campaigns.items():
            if campaign.get("organizer_employer") == profile.employer:
                workplace_campaigns.append({
                    "campaign_id": campaign_id,
                    "title": campaign.get("title", ""),
                    "organizer": campaign.get("organizer_name", ""),
                    "raised": campaign.get("raised_amount", 0),
                })
        
        return {
            "employer": profile.employer,
            "campaigns": workplace_campaigns,
            "count": len(workplace_campaigns),
        }
    
    async def _tool_find_alumni(self, user_id: str) -> Dict[str, Any]:
        """Find campaigns related to user's schools."""
        profile = self._user_profiles.get(user_id)
        if not profile or not profile.schools:
            return {"campaigns": [], "message": "No school information"}
        
        alumni_campaigns = []
        
        for campaign_id, campaign in self._campaigns.items():
            campaign_schools = set(campaign.get("organizer_schools", []))
            common = set(profile.schools) & campaign_schools
            if common:
                alumni_campaigns.append({
                    "campaign_id": campaign_id,
                    "title": campaign.get("title", ""),
                    "shared_schools": list(common),
                    "raised": campaign.get("raised_amount", 0),
                })
        
        return {
            "schools": profile.schools,
            "campaigns": alumni_campaigns,
            "count": len(alumni_campaigns),
        }
    
    async def _tool_find_local(
        self,
        user_id: str,
        radius_miles: float = 25
    ) -> Dict[str, Any]:
        """Find campaigns in user's geographic area."""
        profile = self._user_profiles.get(user_id)
        if not profile:
            return {"campaigns": [], "message": "User not found"}
        
        local_campaigns = []
        
        for campaign_id, campaign in self._campaigns.items():
            # Simple city/state matching (could be enhanced with coordinates)
            if profile.city and campaign.get("city") == profile.city:
                local_campaigns.append({
                    "campaign_id": campaign_id,
                    "title": campaign.get("title", ""),
                    "location": f"{campaign.get('city', '')}, {campaign.get('state', '')}",
                    "distance": "same city",
                })
            elif profile.state and campaign.get("state") == profile.state:
                local_campaigns.append({
                    "campaign_id": campaign_id,
                    "title": campaign.get("title", ""),
                    "location": f"{campaign.get('city', '')}, {campaign.get('state', '')}",
                    "distance": "same state",
                })
        
        return {
            "user_location": f"{profile.city}, {profile.state}",
            "campaigns": local_campaigns,
            "count": len(local_campaigns),
        }
    
    async def _tool_cluster_campaigns(
        self,
        community_type: str,
        community_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Identify clusters of related campaigns in a community."""
        clusters = []
        
        if community_type == "workplace":
            # Group campaigns by employer
            by_employer = defaultdict(list)
            for cid, campaign in self._campaigns.items():
                employer = campaign.get("organizer_employer")
                if employer:
                    by_employer[employer].append(cid)
            
            for employer, campaign_ids in by_employer.items():
                if len(campaign_ids) >= 2:
                    total_raised = sum(
                        self._campaigns[cid].get("raised_amount", 0)
                        for cid in campaign_ids
                    )
                    clusters.append({
                        "name": f"{employer} Giving",
                        "type": "workplace",
                        "campaigns": campaign_ids,
                        "total_raised": total_raised,
                    })
        
        elif community_type == "school":
            # Group by school
            by_school = defaultdict(list)
            for cid, campaign in self._campaigns.items():
                for school in campaign.get("organizer_schools", []):
                    by_school[school].append(cid)
            
            for school, campaign_ids in by_school.items():
                if len(campaign_ids) >= 2:
                    clusters.append({
                        "name": f"{school} Alumni",
                        "type": "school",
                        "campaigns": list(set(campaign_ids)),
                    })
        
        elif community_type == "geographic":
            # Group by city
            by_city = defaultdict(list)
            for cid, campaign in self._campaigns.items():
                city = campaign.get("city")
                if city:
                    by_city[city].append(cid)
            
            for city, campaign_ids in by_city.items():
                if len(campaign_ids) >= 3:
                    clusters.append({
                        "name": f"{city} Community",
                        "type": "geographic",
                        "campaigns": campaign_ids,
                    })
        
        return {
            "community_type": community_type,
            "clusters": clusters,
            "count": len(clusters),
        }
    
    # ==================== Public API ====================
    
    def register_user(self, profile: UserProfile) -> None:
        """Register a user profile for community discovery."""
        self._user_profiles[profile.user_id] = profile
        
        # Build social graph
        for conn in profile.connections:
            self._social_graph[profile.user_id].add(conn.connected_user_id)
            if conn.is_mutual:
                self._social_graph[conn.connected_user_id].add(profile.user_id)
    
    def register_campaign(self, campaign: Dict[str, Any]) -> None:
        """Register a campaign for discovery."""
        campaign_id = campaign.get("campaign_id", str(hash(campaign.get("title", ""))))
        self._campaigns[campaign_id] = campaign
    
    async def discover_for_user(
        self,
        user_id: str,
        limit: int = 20
    ) -> List[CampaignProximity]:
        """
        Autonomously discover campaigns connected to a user.
        
        The agent will:
        1. Map the user's social network
        2. Find connected campaigns at various degrees
        3. Calculate proximity scores
        4. Generate personalized narratives
        5. Rank and return top discoveries
        """
        goal = f"Discover community-connected campaigns for user {user_id}"
        
        context = {
            "user_id": user_id,
            "limit": limit,
        }
        
        # Run autonomous discovery
        result = await self.run(goal, context)
        
        # Collect and rank discoveries
        discoveries = []
        
        # Get connected campaigns
        connected = await self._tool_find_connected(user_id, max_degree=2)
        
        for campaign_info in connected.get("connected_campaigns", []):
            campaign_id = campaign_info["campaign_id"]
            
            # Calculate proximity
            proximity = await self._tool_calculate_proximity(user_id, campaign_id)
            
            # Generate narrative
            narrative_result = await self._tool_generate_narrative(
                user_id, campaign_id, proximity
            )
            
            discoveries.append(CampaignProximity(
                campaign_id=campaign_id,
                campaign_title=campaign_info.get("title", ""),
                proximity_types=[
                    ProximityType(p) for p in proximity.get("proximity_types", [])
                ],
                proximity_score=proximity.get("proximity_score", 0),
                shared_context=proximity.get("shared_context", {}),
                narrative=narrative_result.get("narrative", ""),
            ))
        
        # Sort by proximity score
        discoveries.sort(key=lambda x: x.proximity_score, reverse=True)
        
        return discoveries[:limit]
    
    async def find_workplace_giving(self, user_id: str) -> Dict[str, Any]:
        """Find workplace giving opportunities."""
        return await self._tool_find_workplace(user_id)
    
    async def find_alumni_giving(self, user_id: str) -> Dict[str, Any]:
        """Find alumni network giving opportunities."""
        return await self._tool_find_alumni(user_id)
    
    async def get_community_clusters(
        self,
        community_type: str = "workplace"
    ) -> List[CommunityCluster]:
        """Get clusters of campaigns in communities."""
        result = await self._tool_cluster_campaigns(community_type)
        
        clusters = []
        for cluster_data in result.get("clusters", []):
            clusters.append(CommunityCluster(
                cluster_id=str(hash(cluster_data["name"])),
                name=cluster_data["name"],
                description=f"{cluster_data['type'].title()} giving community",
                campaigns=cluster_data.get("campaigns", []),
                community_type=cluster_data["type"],
            ))
        
        return clusters
    
    async def discover_communities(
        self,
        donor_profile: Dict[str, Any],
        location: str = ""
    ) -> CampaignProximity:
        """
        Discover community-connected campaigns for a donor.
        
        This is the main public method for community discovery.
        """
        user_id = donor_profile.get("donor_id", "unknown")
        
        # Use discover_for_user which does the actual work
        proximities = await self.discover_for_user(user_id, limit=10)
        
        # Return first proximity or create empty one
        if proximities:
            return proximities[0]
        else:
            return CampaignProximity(
                campaign_id="",
                campaign_title="",
                proximity_types=[],
                proximity_score=0.0,
            )
    
    async def _handle_discover_communities(self, message) -> Dict[str, Any]:
        """A2A handler for discover_communities requests."""
        donor_id = message.payload.get("donor_id", "")
        location = message.payload.get("location", "")
        donor_profile = message.payload.get("donor_profile", {})
        
        # Use the discover_communities method
        result = await self.discover_communities(
            donor_profile=donor_profile,
            location=location,
        )
        
        return result.to_dict() if hasattr(result, "to_dict") else result
