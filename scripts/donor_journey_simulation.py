"""
Donor Journey Simulation

Enter past donations → Watch agents build your giving identity in real-time

Shows:
- AG-UI streaming agent thoughts as they analyze
- Profile emerging with cause affinities, motivators, patterns
- Personalized campaign recommendations with explanations
- Suggested giving circle based on inferred interests
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.donor_affinity_profiler import DonorAffinityProfiler
from src.agents.campaign_matching_engine import CampaignMatchingEngine
from src.agents.recurring_curator import RecurringCuratorAgent
from src.agents.giving_circle_orchestrator import GivingCircleOrchestrator
from src.agents.community_discovery import CommunityDiscoveryAgent

import structlog

logger = structlog.get_logger()


class StreamingLogger:
    """Streams agent thoughts for AG-UI visualization."""
    
    def __init__(self):
        self.thoughts: List[Dict[str, Any]] = []
    
    def stream_thought(self, agent_name: str, thought: str, step: str = ""):
        """Stream a thought from an agent."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "step": step,
            "thought": thought,
        }
        self.thoughts.append(entry)
        print(f"[{agent_name}] {thought}")
    
    def stream_insight(self, agent_name: str, insight: str, data: Dict[str, Any] = None):
        """Stream an insight with data."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "type": "insight",
            "insight": insight,
            "data": data or {},
        }
        self.thoughts.append(entry)
        print(f"💡 [{agent_name}] {insight}")
        if data:
            for key, value in data.items():
                print(f"   {key}: {value}")
    
    def stream_profile_evolution(self, profile_data: Dict[str, Any]):
        """Stream profile evolution."""
        print("\n" + "=" * 80)
        print("PROFILE EVOLVING...")
        print("=" * 80)
        print(f"Donor ID: {profile_data.get('donor_id', 'unknown')}")
        print(f"Total Giving: ${profile_data.get('total_lifetime_giving', 0):,.2f}")
        print(f"Average Donation: ${profile_data.get('average_donation', 0):,.2f}")
        
        if profile_data.get('cause_affinities'):
            print("\nCause Affinities:")
            for aff in profile_data['cause_affinities'][:5]:
                print(f"  • {aff.get('category', 'Unknown')}: {aff.get('affinity_score', 0):.1%}")
        
        if profile_data.get('giving_motivators'):
            print("\nGiving Motivators:")
            for motivator in profile_data['giving_motivators'][:5]:
                print(f"  • {motivator}")
        
        if profile_data.get('primary_pattern'):
            print(f"\nGiving Pattern: {profile_data['primary_pattern']}")
        
        if profile_data.get('personality_summary'):
            print(f"\nPersonality Insights:")
            print(f"  {profile_data['personality_summary'][:200]}...")
        
        print("=" * 80 + "\n")
    
    def get_thoughts(self) -> List[Dict[str, Any]]:
        """Get all streamed thoughts."""
        return self.thoughts


# Monkey-patch agents to stream thoughts
original_run = None

def create_streaming_wrapper(agent, streamer: StreamingLogger):
    """Create a wrapper that streams agent thoughts."""
    original_run = agent.run
    
    async def streaming_run(goal: str, context: Dict[str, Any]):
        streamer.stream_thought(agent.name, f"Starting: {goal[:100]}...", "goal")
        
        # Stream planning
        streamer.stream_thought(agent.name, "Reasoning about approach...", "reasoning")
        
        result = await original_run(goal, context)
        
        # Stream reflection
        if hasattr(agent, '_memory') and agent._memory.reasoning_steps:
            for step in agent._memory.reasoning_steps[-3:]:  # Last 3 steps
                if hasattr(step, 'thought'):
                    streamer.stream_thought(agent.name, step.thought, "reasoning")
        
        streamer.stream_thought(agent.name, "Completed analysis", "complete")
        return result
    
    agent.run = streaming_run
    return agent


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80 + "\n")


def print_section(text: str):
    """Print a formatted section."""
    print("\n" + "-" * 80)
    print(text)
    print("-" * 80 + "\n")


async def main():
    """Main demo function."""
    print_header("DONOR JOURNEY SIMULATION")
    print("Enter past donations → Watch agents build your giving identity in real-time")
    print("\nThis demo shows:")
    print("  • AG-UI streaming agent thoughts as they analyze")
    print("  • Profile emerging with cause affinities, motivators, patterns")
    print("  • Personalized campaign recommendations with explanations")
    print("  • Suggested giving circle based on inferred interests")
    
    # Initialize streaming
    streamer = StreamingLogger()
    
    # Initialize agents
    print("\nInitializing agents...")
    profiler = DonorAffinityProfiler()
    matcher = CampaignMatchingEngine()
    curator = RecurringCuratorAgent()
    giving_circle = GivingCircleOrchestrator()
    community = CommunityDiscoveryAgent()
    
    # Enable streaming
    profiler = create_streaming_wrapper(profiler, streamer)
    matcher = create_streaming_wrapper(matcher, streamer)
    curator = create_streaming_wrapper(curator, streamer)
    giving_circle = create_streaming_wrapper(giving_circle, streamer)
    community = create_streaming_wrapper(community, streamer)
    
    print("✓ Agents initialized with streaming enabled\n")
    
    # Get donor input
    print_section("STEP 1: ENTER YOUR DONATION HISTORY")
    print("Enter your past donations to build your giving identity")
    print("(Press Enter after each donation, empty line to finish)\n")
    
    donations = []
    donor_id = input("Your Donor ID (or press Enter for 'donor_journey'): ").strip() or "donor_journey"
    name = input("Your Name: ").strip() or "Alex"
    email = input("Your Email: ").strip() or "alex@example.com"
    location = input("Your Location (city, state): ").strip() or "San Francisco, CA"
    
    print("\nEnter donations (format: amount category description, or press Enter for sample):")
    donation_input = input("Donation 1: ").strip()
    
    if not donation_input:
        # Use sample donations
        print("Using sample donations...")
        donations = [
            {
                "amount": 100.0,
                "category": "medical",
                "campaign_title": "Help Sarah Fight Cancer",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "amount": 50.0,
                "category": "education",
                "campaign_title": "Scholarship Fund for Students",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "amount": 75.0,
                "category": "community",
                "campaign_title": "Local Food Bank Support",
                "timestamp": datetime.now().isoformat(),
            },
        ]
    else:
        # Parse donations
        i = 1
        while donation_input:
            parts = donation_input.split(" ", 2)
            if len(parts) >= 2:
                try:
                    amount = float(parts[0])
                    category = parts[1]
                    description = parts[2] if len(parts) > 2 else ""
                    donations.append({
                        "amount": amount,
                        "category": category,
                        "campaign_title": description,
                        "timestamp": datetime.now().isoformat(),
                    })
                    i += 1
                    donation_input = input(f"Donation {i}: ").strip()
                except ValueError:
                    print("Invalid format. Use: amount category description")
                    donation_input = input(f"Donation {i}: ").strip()
            else:
                break
    
    print(f"\n✓ {len(donations)} donations entered")
    
    # Build profile with streaming
    print_section("STEP 2: BUILDING YOUR GIVING IDENTITY (REAL-TIME)")
    print("Watch agents analyze your donations and build your profile...\n")
    
    streamer.stream_thought("System", "Starting profile building process...")
    
    profile = await profiler.build_profile(
        donor_id=donor_id,
        donations=donations,
        metadata={
            "name": name,
            "email": email,
            "location": location,
        },
    )
    
    # Stream profile evolution
    profile_dict = profile.to_dict() if hasattr(profile, "to_dict") else profile
    streamer.stream_profile_evolution(profile_dict)
    
    # Get campaign recommendations
    print_section("STEP 3: PERSONALIZED CAMPAIGN RECOMMENDATIONS")
    print("Finding campaigns that match your giving identity...\n")
    
    # Sample campaigns
    sample_campaigns = [
        {
            "campaign_id": "campaign_1",
            "title": "Help Sarah Fight Cancer - Medical Treatment Fund",
            "description": "Sarah is a 32-year-old mother of two who was recently diagnosed with stage 3 breast cancer.",
            "category": "medical",
            "goal_amount": 50000.0,
            "raised_amount": 15000.0,
            "donor_count": 45,
        },
        {
            "campaign_id": "campaign_2",
            "title": "Scholarship Fund for Underprivileged Students",
            "description": "Supporting students from low-income families to pursue higher education.",
            "category": "education",
            "goal_amount": 30000.0,
            "raised_amount": 12000.0,
            "donor_count": 80,
        },
        {
            "campaign_id": "campaign_3",
            "title": "Community Food Bank Expansion",
            "description": "Expanding our food bank to serve more families in need.",
            "category": "community",
            "goal_amount": 25000.0,
            "raised_amount": 18000.0,
            "donor_count": 120,
        },
    ]
    
    streamer.stream_thought("Campaign Matcher", f"Analyzing {len(sample_campaigns)} campaigns...")
    
    recommendations = []
    for campaign in sample_campaigns:
        analysis = await matcher.analyze_campaign(campaign_data=campaign)
        match_result = await matcher._tool_match_to_donor(
            campaign=campaign,
            donor_profile=profile_dict,
        )
        
        recommendations.append({
            "campaign": campaign,
            "analysis": analysis.to_dict() if hasattr(analysis, "to_dict") else analysis,
            "match": match_result,
        })
        
        streamer.stream_insight(
            "Campaign Matcher",
            f"Found match: {campaign['title']}",
            {
                "match_score": f"{match_result.get('match_score', 0):.1%}",
                "reasons": match_result.get("reasons", [])[:3],
            },
        )
    
    # Sort by match score
    recommendations.sort(key=lambda x: x["match"].get("match_score", 0), reverse=True)
    
    print("\nTop Recommendations:")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"\n{i}. {rec['campaign']['title']}")
        print(f"   Match Score: {rec['match'].get('match_score', 0):.1%}")
        print(f"   Why it matches:")
        for reason in rec["match"].get("reasons", [])[:3]:
            print(f"     • {reason}")
    
    # Recurring opportunities
    print_section("STEP 4: RECURRING GIVING OPPORTUNITIES")
    print("Identifying campaigns suitable for recurring support...\n")
    
    opportunities = await curator.curate_opportunities(
        donor_profile=profile_dict,
        campaigns=[r["campaign"] for r in recommendations],
    )
    
    if opportunities:
        print(f"\nFound {len(opportunities)} recurring opportunities:\n")
        for i, opp in enumerate(opportunities[:3], 1):
            print(f"{i}. {opp.campaign_title}")
            print(f"   Recommended: ${opp.recommended_monthly_amount:.2f}/month")
            print(f"   Suitability: {opp.suitability_score:.1%}")
            print(f"   Why: {opp.why_suitable[:100]}...")
    else:
        print("No recurring opportunities found at this time.")
    
    # Giving circle suggestion
    print_section("STEP 5: SUGGESTED GIVING CIRCLE")
    print("Based on your interests, suggesting a giving circle...\n")
    
    community_insights = await community.discover_communities(
        donor_profile=profile_dict,
        location=location,
    )
    
    circles = await giving_circle.orchestrate_circle(
        donor_profile=profile_dict,
        community_context=community_insights.to_dict() if hasattr(community_insights, "to_dict") else {},
    )
    
    if circles:
        circle = circles[0]
        print(f"\nSuggested Circle: {circle.name if hasattr(circle, 'name') else 'Community Giving Circle'}")
        print(f"Mission: {circle.mission if hasattr(circle, 'mission') else 'Supporting causes you care about'}")
        print(f"Members: {circle.member_count if hasattr(circle, 'member_count') else 'TBD'}")
        print(f"Focus Areas:")
        for area in (circle.focus_areas if hasattr(circle, 'focus_areas') else profile_dict.get('cause_affinities', [])[:3]):
            if isinstance(area, dict):
                print(f"  • {area.get('category', area)}")
            else:
                print(f"  • {area}")
    else:
        print("No giving circle suggestions at this time.")
    
    # Save results
    print_section("SAVING RESULTS")
    output_file = f"donor_journey_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    output_data = {
        "donor": {
            "donor_id": donor_id,
            "name": name,
            "email": email,
            "location": location,
        },
        "donations": donations,
        "profile": profile_dict,
        "recommendations": [
            {
                "campaign": rec["campaign"],
                "match_score": rec["match"].get("match_score", 0),
                "reasons": rec["match"].get("reasons", []),
            }
            for rec in recommendations
        ],
        "recurring_opportunities": [
            {
                "campaign_title": opp.campaign_title,
                "recommended_monthly_amount": opp.recommended_monthly_amount,
                "suitability_score": opp.suitability_score,
                "why_suitable": opp.why_suitable,
            }
            for opp in opportunities
        ],
        "agent_thoughts": streamer.get_thoughts(),
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Results saved to: {output_file}")
    
    print_header("DEMO COMPLETE")
    print("This demonstrates:")
    print("  ✓ Real-time agent reasoning and thought streaming")
    print("  ✓ Dynamic profile building from donation history")
    print("  ✓ Personalized campaign matching with explanations")
    print("  ✓ Recurring giving opportunity identification")
    print("  ✓ Giving circle suggestions based on inferred interests")
    print("\nNet new: GoFundMe doesn't understand why donors give, just what they gave to.")


if __name__ == "__main__":
    asyncio.run(main())

