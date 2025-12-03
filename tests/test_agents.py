"""
Tests for the agent implementations.
"""

import pytest
from datetime import datetime, timedelta

from src.core.base_agent import AgentMessage
from src.agents import (
    DonorAffinityProfiler,
    CampaignMatchingEngine,
    RecurringOpportunityCurator,
    GivingCircleOrchestrator,
    EngagementAgent,
)
from src.data.synthetic import SyntheticDataGenerator


@pytest.fixture
def generator():
    """Create a synthetic data generator."""
    return SyntheticDataGenerator(seed=42)


@pytest.fixture
def sample_donations(generator):
    """Generate sample donations."""
    _, donations = generator.generate_donor_with_history(num_donations=10)
    return donations


@pytest.fixture
def sample_campaign(generator):
    """Generate a sample campaign."""
    return generator.generate_campaign(category="medical")


class TestDonorAffinityProfiler:
    """Tests for the Donor Affinity Profiler agent."""
    
    @pytest.fixture
    def agent(self):
        return DonorAffinityProfiler()
    
    @pytest.mark.asyncio
    async def test_initialization(self, agent):
        """Test agent initialization."""
        await agent.initialize()
        assert agent.state.is_ready
    
    @pytest.mark.asyncio
    async def test_build_profile(self, agent, sample_donations):
        """Test building a donor profile."""
        await agent.initialize()
        
        message = AgentMessage(
            sender="test",
            recipient=agent.agent_id,
            action="build_profile",
            payload={
                "donor_id": "test-donor-1",
                "donations": sample_donations,
                "donor_metadata": {"location": "New York, NY"},
            },
        )
        
        response = await agent.process_message(message)
        
        assert response.payload.get("status") == "success"
        profile = response.payload.get("profile", {})
        assert profile.get("donor_id") == "test-donor-1"
        assert profile.get("donation_count") == len(sample_donations)
        assert len(profile.get("cause_affinities", [])) > 0
    
    @pytest.mark.asyncio
    async def test_get_affinities(self, agent, sample_donations):
        """Test getting donor affinities."""
        await agent.initialize()
        
        # First build a profile
        build_message = AgentMessage(
            sender="test",
            recipient=agent.agent_id,
            action="build_profile",
            payload={
                "donor_id": "test-donor-2",
                "donations": sample_donations,
            },
        )
        await agent.process_message(build_message)
        
        # Then get affinities
        get_message = AgentMessage(
            sender="test",
            recipient=agent.agent_id,
            action="get_affinities",
            payload={"donor_id": "test-donor-2", "top_n": 3},
        )
        
        response = await agent.process_message(get_message)
        
        assert response.payload.get("status") == "success"
        affinities = response.payload.get("affinities", [])
        assert len(affinities) <= 3


class TestCampaignMatchingEngine:
    """Tests for the Campaign Matching Engine agent."""
    
    @pytest.fixture
    def agent(self):
        return CampaignMatchingEngine()
    
    @pytest.mark.asyncio
    async def test_analyze_campaign(self, agent, sample_campaign):
        """Test campaign analysis."""
        await agent.initialize()
        
        message = AgentMessage(
            sender="test",
            recipient=agent.agent_id,
            action="analyze_campaign",
            payload=sample_campaign,
        )
        
        response = await agent.process_message(message)
        
        assert response.payload.get("status") == "success"
        analysis = response.payload.get("analysis", {})
        assert analysis.get("campaign_id") == sample_campaign["campaign_id"]
        assert "taxonomy" in analysis
        assert "urgency_level" in analysis
        assert "legitimacy_score" in analysis
    
    @pytest.mark.asyncio
    async def test_search_campaigns(self, agent, generator):
        """Test campaign search."""
        await agent.initialize()
        
        # Add some campaigns
        for _ in range(5):
            campaign = generator.generate_campaign(category="medical")
            message = AgentMessage(
                sender="test",
                recipient=agent.agent_id,
                action="analyze_campaign",
                payload=campaign,
            )
            await agent.process_message(message)
        
        # Search
        search_message = AgentMessage(
            sender="test",
            recipient=agent.agent_id,
            action="search_campaigns",
            payload={"category": "medical", "limit": 3},
        )
        
        response = await agent.process_message(search_message)
        
        assert response.payload.get("status") == "success"
        assert response.payload.get("count", 0) <= 3


class TestRecurringOpportunityCurator:
    """Tests for the Recurring Opportunity Curator agent."""
    
    @pytest.fixture
    def agent(self):
        return RecurringOpportunityCurator()
    
    @pytest.mark.asyncio
    async def test_analyze_for_recurring(self, agent):
        """Test analyzing campaign for recurring potential."""
        await agent.initialize()
        
        campaign = {
            "campaign_id": "test-campaign-1",
            "title": "Help John with ongoing dialysis treatment",
            "description": "John needs monthly dialysis treatments. This is a chronic condition requiring ongoing support.",
            "category": "medical",
            "goal_amount": 50000,
            "raised_amount": 15000,
            "updates": [
                {"date": "2024-01-01", "content": "Update 1"},
                {"date": "2024-02-01", "content": "Update 2"},
            ],
            "organizer_response_rate": 0.8,
            "created_at": (datetime.now() - timedelta(days=60)).isoformat(),
        }
        
        message = AgentMessage(
            sender="test",
            recipient=agent.agent_id,
            action="analyze_for_recurring",
            payload=campaign,
        )
        
        response = await agent.process_message(message)
        
        assert response.payload.get("status") == "success"
        assert response.payload.get("is_recurring") == True
    
    @pytest.mark.asyncio
    async def test_suggest_portfolio(self, agent):
        """Test portfolio suggestion."""
        await agent.initialize()
        
        message = AgentMessage(
            sender="test",
            recipient=agent.agent_id,
            action="suggest_portfolio",
            payload={
                "donor_profile": {
                    "cause_affinities": [{"category": "medical", "score": 0.8}],
                    "average_donation": 50,
                },
                "monthly_budget": 25,
                "diversification": "low",
            },
        )
        
        response = await agent.process_message(message)
        
        assert response.payload.get("status") == "success"
        suggestion = response.payload.get("suggestion", {})
        assert "allocations" in suggestion


class TestGivingCircleOrchestrator:
    """Tests for the Giving Circle Orchestrator agent."""
    
    @pytest.fixture
    def agent(self):
        return GivingCircleOrchestrator()
    
    @pytest.mark.asyncio
    async def test_create_circle(self, agent):
        """Test creating a giving circle."""
        await agent.initialize()
        
        message = AgentMessage(
            sender="test",
            recipient=agent.agent_id,
            action="create_circle",
            payload={
                "name": "Test Family Circle",
                "circle_type": "family",
                "creator_id": "user-1",
                "creator_name": "John Doe",
            },
        )
        
        response = await agent.process_message(message)
        
        assert response.payload.get("status") == "success"
        circle = response.payload.get("circle", {})
        assert circle.get("name") == "Test Family Circle"
        assert circle.get("member_count") == 1
    
    @pytest.mark.asyncio
    async def test_contribute_to_circle(self, agent):
        """Test contributing to a circle."""
        await agent.initialize()
        
        # Create circle first
        create_message = AgentMessage(
            sender="test",
            recipient=agent.agent_id,
            action="create_circle",
            payload={
                "name": "Test Circle",
                "circle_type": "friends",
                "creator_id": "user-1",
                "creator_name": "Jane Doe",
            },
        )
        create_response = await agent.process_message(create_message)
        circle_id = create_response.payload.get("circle", {}).get("circle_id")
        
        # Contribute
        contribute_message = AgentMessage(
            sender="test",
            recipient=agent.agent_id,
            action="contribute",
            payload={
                "circle_id": circle_id,
                "user_id": "user-1",
                "amount": 100,
            },
        )
        
        response = await agent.process_message(contribute_message)
        
        assert response.payload.get("status") == "success"
        assert response.payload.get("amount") == 100
        assert response.payload.get("new_pool_balance") == 100


class TestEngagementAgent:
    """Tests for the Engagement Agent."""
    
    @pytest.fixture
    def agent(self):
        return EngagementAgent()
    
    @pytest.mark.asyncio
    async def test_record_donation(self, agent):
        """Test recording a donation for engagement."""
        await agent.initialize()
        
        message = AgentMessage(
            sender="test",
            recipient=agent.agent_id,
            action="record_donation",
            payload={
                "donor_id": "donor-1",
                "campaign_id": "campaign-1",
                "campaign_title": "Test Campaign",
                "amount": 50,
                "timestamp": datetime.now().isoformat(),
            },
        )
        
        response = await agent.process_message(message)
        
        assert response.payload.get("status") == "success"
        assert response.payload.get("donor_status") in ["new", "active"]
        assert response.payload.get("thank_you_queued") == True
    
    @pytest.mark.asyncio
    async def test_get_lapsed_donors(self, agent):
        """Test getting lapsed donors."""
        await agent.initialize()
        
        # Record an old donation
        old_timestamp = (datetime.now() - timedelta(days=400)).isoformat()
        
        record_message = AgentMessage(
            sender="test",
            recipient=agent.agent_id,
            action="record_donation",
            payload={
                "donor_id": "lapsed-donor-1",
                "campaign_id": "campaign-1",
                "campaign_title": "Old Campaign",
                "amount": 25,
                "timestamp": old_timestamp,
            },
        )
        await agent.process_message(record_message)
        
        # Get lapsed donors
        lapsed_message = AgentMessage(
            sender="test",
            recipient=agent.agent_id,
            action="get_lapsed_donors",
            payload={"days_threshold": 365, "limit": 10},
        )
        
        response = await agent.process_message(lapsed_message)
        
        assert response.payload.get("status") == "success"
        lapsed = response.payload.get("lapsed_donors", [])
        assert len(lapsed) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])





