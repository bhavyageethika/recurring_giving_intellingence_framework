"""
FastAPI application for the Giving Intelligence platform.
Provides RESTful APIs for all agent capabilities.
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.core.a2a_protocol import A2AProtocol, get_protocol
from src.agents import (
    DonorAffinityProfiler,
    CampaignMatchingEngine,
    CommunityDiscoveryAgent,
    RecurringCuratorAgent,
    GivingCircleOrchestrator,
    EngagementAgent,
)

# Request/Response Models
class DonationRecord(BaseModel):
    """A single donation record."""
    campaign_id: str
    campaign_title: str
    campaign_category: str = ""
    campaign_description: str = ""
    amount: float
    timestamp: str
    is_anonymous: bool = False
    is_recurring: bool = False
    source: str = "direct"


class DonorMetadata(BaseModel):
    """Donor metadata."""
    location: str = ""
    joined_date: str = ""
    social_connections: int = 0


class BuildProfileRequest(BaseModel):
    """Request to build a donor profile."""
    donor_id: str
    donations: List[DonationRecord]
    donor_metadata: DonorMetadata = Field(default_factory=DonorMetadata)


class CampaignData(BaseModel):
    """Campaign data for analysis."""
    campaign_id: str
    title: str
    description: str = ""
    category: str = ""
    location: str = ""
    goal_amount: float = 0
    raised_amount: float = 0
    donor_count: int = 0
    share_count: int = 0
    comment_count: int = 0
    organizer: Dict[str, Any] = Field(default_factory=dict)
    updates: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str = ""


class MatchRequest(BaseModel):
    """Request to match campaigns to a donor."""
    donor_id: str
    limit: int = 10


class CauseAffinity(BaseModel):
    """Cause affinity for profile matching."""
    category: str
    score: float


class DonorProfileSummary(BaseModel):
    """Summary of donor profile for matching."""
    cause_affinities: List[CauseAffinity]
    giving_motivators: Dict[str, float] = Field(default_factory=dict)
    average_donation: float = 50
    location: str = ""


class PortfolioSuggestionRequest(BaseModel):
    """Request for portfolio suggestion."""
    donor_profile: DonorProfileSummary
    monthly_budget: float = 25
    diversification: str = "medium"  # low, medium, high


class CreateCircleRequest(BaseModel):
    """Request to create a giving circle."""
    name: str
    circle_type: str = "friends"
    description: str = ""
    creator_id: str
    creator_name: str
    voting_threshold: float = 0.5
    min_contribution: float = 0


class NominateRequest(BaseModel):
    """Request to nominate a campaign."""
    circle_id: str
    campaign_id: str
    campaign_title: str
    nominated_by: str
    reason: str = ""
    requested_amount: float = 0


class VoteRequest(BaseModel):
    """Request to vote on a nomination."""
    circle_id: str
    nomination_id: str
    user_id: str
    vote: str  # "for" or "against"


class ContributeRequest(BaseModel):
    """Request to contribute to a circle."""
    circle_id: str
    user_id: str
    amount: float


class RecordDonationRequest(BaseModel):
    """Request to record a donation for engagement."""
    donor_id: str
    campaign_id: str
    campaign_title: str
    amount: float
    timestamp: str = ""


# Global protocol instance
protocol: Optional[A2AProtocol] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global protocol
    
    # Initialize protocol and agents
    protocol = get_protocol()
    
    # Create and register agents
    agents = [
        DonorAffinityProfiler(),
        CampaignMatchingEngine(),
        CommunityDiscoveryAgent(),
        RecurringCuratorAgent(),
        GivingCircleOrchestrator(),
        EngagementAgent(),
    ]
    
    for agent in agents:
        protocol.register_agent(agent)
    
    # Initialize all agents
    await protocol.initialize_all_agents()
    
    yield
    
    # Shutdown
    await protocol.shutdown_all_agents()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Giving Intelligence Platform",
        description="Multi-agent system for relationship-driven donor engagement",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "agents": protocol.list_agents() if protocol else []}


# Agent listing
@app.get("/agents")
async def list_agents():
    """List all registered agents."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return {"agents": protocol.list_agents()}


# ============================================================================
# Donor Affinity Profiler Endpoints
# ============================================================================

@app.post("/api/v1/donors/profile")
async def build_donor_profile(request: BuildProfileRequest):
    """Build a comprehensive donor profile from donation history."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    from src.core.base_agent import AgentMessage
    
    message = AgentMessage(
        sender="api",
        recipient="donor_affinity_profiler",
        action="build_profile",
        payload={
            "donor_id": request.donor_id,
            "donations": [d.model_dump() for d in request.donations],
            "donor_metadata": request.donor_metadata.model_dump(),
        },
    )
    
    response = await protocol.send_message(
        "api", "donor_affinity_profiler", "build_profile", message.payload
    )
    
    return response.payload


@app.get("/api/v1/donors/{donor_id}/profile")
async def get_donor_profile(donor_id: str):
    """Get an existing donor profile."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "donor_affinity_profiler", "get_profile", {"donor_id": donor_id}
    )
    
    if response.payload.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Donor profile not found")
    
    return response.payload


@app.get("/api/v1/donors/{donor_id}/affinities")
async def get_donor_affinities(donor_id: str, top_n: int = Query(5, ge=1, le=20)):
    """Get top cause affinities for a donor."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "donor_affinity_profiler", "get_affinities",
        {"donor_id": donor_id, "top_n": top_n}
    )
    
    return response.payload


# ============================================================================
# Campaign Matching Engine Endpoints
# ============================================================================

@app.post("/api/v1/campaigns/analyze")
async def analyze_campaign(campaign: CampaignData):
    """Analyze a campaign and add to catalog."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "campaign_matching_engine", "analyze_campaign",
        campaign.model_dump()
    )
    
    return response.payload


@app.get("/api/v1/campaigns/{campaign_id}")
async def get_campaign_analysis(campaign_id: str):
    """Get analysis for a specific campaign."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "campaign_matching_engine", "get_campaign",
        {"campaign_id": campaign_id}
    )
    
    if response.payload.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    return response.payload


@app.get("/api/v1/campaigns")
async def search_campaigns(
    category: Optional[str] = None,
    urgency_min: float = Query(0, ge=0, le=1),
    legitimacy_min: float = Query(0, ge=0, le=1),
    funding_max: float = Query(100, ge=0, le=100),
    is_recurring: Optional[bool] = None,
    limit: int = Query(10, ge=1, le=100),
):
    """Search campaigns by criteria."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "campaign_matching_engine", "search_campaigns",
        {
            "category": category,
            "urgency_min": urgency_min,
            "legitimacy_min": legitimacy_min,
            "funding_max": funding_max,
            "is_recurring": is_recurring,
            "limit": limit,
        }
    )
    
    return response.payload


@app.get("/api/v1/campaigns/{campaign_id}/similar")
async def find_similar_campaigns(campaign_id: str, limit: int = Query(5, ge=1, le=20)):
    """Find campaigns similar to a given campaign."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "campaign_matching_engine", "find_similar",
        {"campaign_id": campaign_id, "limit": limit}
    )
    
    return response.payload


@app.post("/api/v1/campaigns/match")
async def match_campaigns_to_profile(request: MatchRequest):
    """Match campaigns to a donor's profile."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # First get the donor profile
    profile_response = await protocol.send_message(
        "api", "donor_affinity_profiler", "get_profile",
        {"donor_id": request.donor_id}
    )
    
    if profile_response.payload.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Donor profile not found")
    
    # Then match to campaigns
    response = await protocol.send_message(
        "api", "campaign_matching_engine", "match_to_profile",
        {
            "donor_profile": profile_response.payload.get("profile", {}),
            "limit": request.limit,
        }
    )
    
    return response.payload


# ============================================================================
# Community Discovery Endpoints
# ============================================================================

@app.post("/api/v1/community/users")
async def register_user_for_community(user_data: Dict[str, Any]):
    """Register a user for community discovery."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "community_discovery", "register_user", user_data
    )
    
    return response.payload


@app.post("/api/v1/community/campaigns")
async def register_campaign_for_community(campaign_data: Dict[str, Any]):
    """Register a campaign for community discovery."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "community_discovery", "register_campaign", campaign_data
    )
    
    return response.payload


@app.get("/api/v1/community/{user_id}/campaigns")
async def find_proximate_campaigns(
    user_id: str,
    max_degrees: int = Query(2, ge=1, le=3),
    include_local: bool = True,
    include_workplace: bool = True,
    radius_miles: float = Query(25, ge=1, le=100),
    limit: int = Query(10, ge=1, le=50),
):
    """Find campaigns with proximity to a user."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "community_discovery", "find_proximate_campaigns",
        {
            "user_id": user_id,
            "max_degrees": max_degrees,
            "include_local": include_local,
            "include_workplace": include_workplace,
            "radius_miles": radius_miles,
            "limit": limit,
        }
    )
    
    return response.payload


@app.get("/api/v1/community/{user_id}/workplace")
async def find_workplace_opportunities(user_id: str, limit: int = Query(10, ge=1, le=50)):
    """Find workplace giving opportunities."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "community_discovery", "find_workplace_opportunities",
        {"user_id": user_id, "limit": limit}
    )
    
    return response.payload


# ============================================================================
# Recurring Opportunity Endpoints
# ============================================================================

@app.post("/api/v1/recurring/analyze")
async def analyze_for_recurring(campaign: CampaignData):
    """Analyze a campaign for recurring giving potential."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "recurring_curator", "analyze_for_recurring",
        campaign.model_dump()
    )
    
    return response.payload


@app.get("/api/v1/recurring/campaigns")
async def get_recurring_campaigns(
    need_type: Optional[str] = None,
    category: Optional[str] = None,
    min_sustainability: float = Query(0, ge=0, le=1),
    limit: int = Query(10, ge=1, le=50),
):
    """Get recurring campaigns."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "recurring_curator", "get_recurring_campaigns",
        {
            "need_type": need_type,
            "category": category,
            "min_sustainability": min_sustainability,
            "limit": limit,
        }
    )
    
    return response.payload


@app.post("/api/v1/recurring/portfolio/suggest")
async def suggest_portfolio(request: PortfolioSuggestionRequest):
    """Suggest an optimal giving portfolio."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "recurring_curator", "suggest_portfolio",
        {
            "donor_profile": request.donor_profile.model_dump(),
            "monthly_budget": request.monthly_budget,
            "diversification": request.diversification,
        }
    )
    
    return response.payload


@app.get("/api/v1/recurring/{donor_id}/impact")
async def get_recurring_impact(donor_id: str):
    """Get impact report for recurring giving."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "recurring_curator", "get_impact_report",
        {"donor_id": donor_id}
    )
    
    return response.payload


# ============================================================================
# Giving Circle Endpoints
# ============================================================================

@app.post("/api/v1/circles")
async def create_giving_circle(request: CreateCircleRequest):
    """Create a new giving circle."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "giving_circle_orchestrator", "create_circle",
        request.model_dump()
    )
    
    return response.payload


@app.get("/api/v1/circles/{circle_id}")
async def get_giving_circle(circle_id: str):
    """Get giving circle details."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "giving_circle_orchestrator", "get_circle",
        {"circle_id": circle_id}
    )
    
    if response.payload.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Circle not found")
    
    return response.payload


@app.post("/api/v1/circles/{circle_id}/join")
async def join_circle(circle_id: str, user_id: str, display_name: str, role: str = "member"):
    """Join a giving circle."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "giving_circle_orchestrator", "join_circle",
        {
            "circle_id": circle_id,
            "user_id": user_id,
            "display_name": display_name,
            "role": role,
        }
    )
    
    return response.payload


@app.post("/api/v1/circles/contribute")
async def contribute_to_circle(request: ContributeRequest):
    """Contribute to a circle's pool."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "giving_circle_orchestrator", "contribute",
        request.model_dump()
    )
    
    return response.payload


@app.post("/api/v1/circles/nominate")
async def nominate_campaign(request: NominateRequest):
    """Nominate a campaign for circle funding."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "giving_circle_orchestrator", "nominate_campaign",
        request.model_dump()
    )
    
    return response.payload


@app.post("/api/v1/circles/vote")
async def vote_on_nomination(request: VoteRequest):
    """Vote on a campaign nomination."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "giving_circle_orchestrator", "vote",
        request.model_dump()
    )
    
    return response.payload


@app.get("/api/v1/circles/{circle_id}/nominations")
async def get_nominations(circle_id: str, status: Optional[str] = None):
    """Get nominations for a circle."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "giving_circle_orchestrator", "get_nominations",
        {"circle_id": circle_id, "status": status}
    )
    
    return response.payload


@app.get("/api/v1/circles/{circle_id}/impact")
async def get_circle_impact(circle_id: str):
    """Get impact report for a giving circle."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "giving_circle_orchestrator", "get_impact_report",
        {"circle_id": circle_id}
    )
    
    return response.payload


@app.get("/api/v1/users/{user_id}/circles")
async def get_user_circles(user_id: str):
    """Get circles a user belongs to."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "giving_circle_orchestrator", "get_user_circles",
        {"user_id": user_id}
    )
    
    return response.payload


# ============================================================================
# Engagement Endpoints
# ============================================================================

@app.post("/api/v1/engagement/donation")
async def record_donation_for_engagement(request: RecordDonationRequest):
    """Record a donation for engagement tracking."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "engagement_agent", "record_donation",
        request.model_dump()
    )
    
    return response.payload


@app.get("/api/v1/engagement/{donor_id}/status")
async def get_engagement_status(donor_id: str):
    """Get donor engagement status."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "engagement_agent", "get_donor_status",
        {"donor_id": donor_id}
    )
    
    return response.payload


@app.get("/api/v1/engagement/{donor_id}/updates")
async def get_campaign_updates(donor_id: str, limit: int = Query(10, ge=1, le=50)):
    """Get campaign updates for a donor."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "engagement_agent", "get_updates_for_donor",
        {"donor_id": donor_id, "limit": limit}
    )
    
    return response.payload


@app.get("/api/v1/engagement/lapsed")
async def get_lapsed_donors(
    days_threshold: int = Query(365, ge=90, le=730),
    limit: int = Query(50, ge=1, le=200),
):
    """Get lapsed donors for re-engagement."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "engagement_agent", "get_lapsed_donors",
        {"days_threshold": days_threshold, "limit": limit}
    )
    
    return response.payload


@app.post("/api/v1/engagement/reactivate")
async def generate_reactivation_campaign(
    donor_ids: Optional[List[str]] = None,
    limit: int = Query(50, ge=1, le=200),
):
    """Generate reactivation campaign for lapsed donors."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await protocol.send_message(
        "api", "engagement_agent", "generate_reactivation",
        {"donor_ids": donor_ids, "limit": limit}
    )
    
    return response.payload


@app.get("/api/v1/engagement/{donor_id}/recommendations")
async def get_recommendations(donor_id: str, limit: int = Query(5, ge=1, le=20)):
    """Get campaign recommendations for a donor."""
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Get available campaigns from the catalog
    catalog_response = await protocol.send_message(
        "api", "campaign_matching_engine", "catalog_stats", {}
    )
    
    response = await protocol.send_message(
        "api", "engagement_agent", "get_recommendations",
        {
            "donor_id": donor_id,
            "available_campaigns": [],  # Would be populated from catalog
            "limit": limit,
        }
    )
    
    return response.payload


# ============================================================================
# Orchestration Endpoints (Cross-Agent Workflows)
# ============================================================================

@app.post("/api/v1/orchestrate/donor-journey")
async def orchestrate_donor_journey(donor_id: str):
    """
    Orchestrate a complete donor journey:
    1. Get donor profile (Affinity Profiler)
    2. Find matching campaigns (Campaign Matching Engine)
    3. Add community context (Community Discovery)
    4. Identify recurring opportunities (Recurring Curator)
    5. Check giving circles (Giving Circle Orchestrator)
    """
    if not protocol:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    results = {}
    
    # Step 1: Get donor profile
    profile_response = await protocol.send_message(
        "api", "donor_affinity_profiler", "get_profile",
        {"donor_id": donor_id}
    )
    results["profile"] = profile_response.payload
    
    if profile_response.payload.get("status") == "not_found":
        return {"status": "no_profile", "message": "Build a profile first with donation history"}
    
    profile = profile_response.payload.get("profile", {})
    
    # Step 2: Get matched campaigns
    match_response = await protocol.send_message(
        "api", "campaign_matching_engine", "match_to_profile",
        {"donor_profile": profile, "limit": 10}
    )
    results["matched_campaigns"] = match_response.payload.get("matches", [])
    
    # Step 3: Get community campaigns
    community_response = await protocol.send_message(
        "api", "community_discovery", "find_proximate_campaigns",
        {"user_id": donor_id, "limit": 5}
    )
    results["community_campaigns"] = community_response.payload.get("campaigns", [])
    
    # Step 4: Get recurring opportunities
    recurring_response = await protocol.send_message(
        "api", "recurring_curator", "match_to_donor",
        {"donor_profile": profile, "limit": 5}
    )
    results["recurring_opportunities"] = recurring_response.payload.get("matches", [])
    
    # Step 5: Get giving circles
    circles_response = await protocol.send_message(
        "api", "giving_circle_orchestrator", "get_user_circles",
        {"user_id": donor_id}
    )
    results["giving_circles"] = circles_response.payload.get("circles", [])
    
    # Step 6: Get engagement status
    engagement_response = await protocol.send_message(
        "api", "engagement_agent", "get_donor_status",
        {"donor_id": donor_id}
    )
    results["engagement"] = engagement_response.payload
    
    return {
        "status": "success",
        "donor_id": donor_id,
        "journey": results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

