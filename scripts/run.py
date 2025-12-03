#!/usr/bin/env python3
"""
Real execution script for the Giving Intelligence Platform.
Actually runs the autonomous agents with live LLM calls.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.donor_affinity_profiler import DonorAffinityProfiler, DonorProfile
from src.agents.campaign_matching_engine import CampaignMatchingEngine
from src.agents.community_discovery import CommunityDiscoveryAgent, UserProfile, SocialConnection, ConnectionType
from src.agents.recurring_curator import RecurringCuratorAgent
from src.agents.giving_circle_orchestrator import GivingCircleOrchestrator
from src.agents.engagement_agent import EngagementAgent
from src.data.synthetic import SyntheticDataGenerator


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}  {text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.END}\n")


def print_section(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- {text} ---{Colors.END}")


def print_step(text: str):
    print(f"{Colors.GREEN}> {text}{Colors.END}")


def print_result(label: str, value: str):
    print(f"  {Colors.YELLOW}{label}:{Colors.END} {value}")


def print_agent_thought(thought: str):
    print(f"  {Colors.BLUE}[Agent Thought]: {thought[:100]}...{Colors.END}" if len(thought) > 100 else f"  {Colors.BLUE}[Agent Thought]: {thought}{Colors.END}")


async def run_donor_profiler():
    """Run the Donor Affinity Profiler agent."""
    print_header("AGENT 1: DONOR AFFINITY PROFILER")
    
    print("Creating agent and generating test data...")
    profiler = DonorAffinityProfiler()
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate a donor with history
    donor, donations = generator.generate_donor_with_history(num_donations=12)
    
    print_section("Input Data")
    print_result("Donor ID", donor["donor_id"])
    print_result("Name", donor["display_name"])
    print_result("Location", f"{donor['city']}, {donor['state']}")
    print_result("Donations", str(len(donations)))
    print_result("Total Given", f"${sum(d['amount'] for d in donations):,.2f}")
    
    print_section("Agent Execution")
    print_step("Agent is planning analysis approach...")
    
    # Actually run the agent
    profile = await profiler.build_profile(
        donor_id=donor["donor_id"],
        donations=donations,
        metadata={
            "location": f"{donor['city']}, {donor['state']}",
            "employer": donor.get("employer"),
        }
    )
    
    # Show reasoning trace
    print_section("Agent Reasoning Trace")
    for i, step in enumerate(profiler._memory.reasoning_history[:5], 1):
        print_agent_thought(step.thought)
    
    # Show tasks executed
    print_section("Tasks Executed")
    for task_id, task in list(profiler._memory.tasks.items())[:5]:
        status_icon = "[DONE]" if task.status.value == "completed" else "[    ]"
        print(f"  {status_icon} {task.description[:60]}...")
    
    print_section("Generated Profile")
    print_result("Profile Completeness", f"{profile.profile_completeness * 100:.0f}%")
    print_result("Total Lifetime Giving", f"${profile.total_lifetime_giving:,.2f}")
    print_result("Average Donation", f"${profile.average_donation:,.2f}")
    print_result("Donation Count", str(profile.donation_count))
    
    print("\n  Top Cause Affinities:")
    for aff in profile.cause_affinities[:3]:
        print(f"    • {aff.category.value}: score={aff.score:.2f}, confidence={aff.confidence:.2f}")
        if aff.llm_insights:
            print(f"      {Colors.CYAN}Insight: {aff.llm_insights[:80]}...{Colors.END}")
    
    print("\n  Giving Motivators:")
    for motivator, score in list(profile.giving_motivators.items())[:3]:
        print(f"    • {motivator.value}: {score:.2f}")
    
    print("\n  Engagement Scores:")
    print(f"    • Engagement: {profile.engagement_score:.2f}")
    print(f"    • Recency: {profile.recency_score:.2f}")
    print(f"    • Loyalty: {profile.loyalty_score:.2f}")
    
    if profile.personality_summary:
        print(f"\n  {Colors.BOLD}LLM-Generated Personality Summary:{Colors.END}")
        print(f"  {profile.personality_summary[:300]}...")
    
    if profile.engagement_recommendations:
        print(f"\n  {Colors.BOLD}Engagement Recommendations:{Colors.END}")
        for rec in profile.engagement_recommendations[:3]:
            print(f"    -> {rec}")
    
    return profiler, profile, donor, donations


async def run_campaign_matcher(donor_profile: DonorProfile):
    """Run the Campaign Matching Engine agent."""
    print_header("AGENT 2: CAMPAIGN MATCHING ENGINE")
    
    print("Creating agent and generating campaigns...")
    matcher = CampaignMatchingEngine()
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate campaigns
    campaigns = [generator.generate_campaign() for _ in range(5)]
    
    print_section("Analyzing Campaigns")
    analyzed = []
    for i, campaign in enumerate(campaigns, 1):
        print_step(f"Analyzing campaign {i}: {campaign['title'][:40]}...")
        
        analysis = await matcher.analyze_campaign(campaign)
        analyzed.append(analysis)
        
        print_result("Category", analysis.taxonomy.primary_category)
        print_result("Urgency", f"{analysis.urgency_level.value} ({analysis.urgency_score:.2f})")
        print_result("Legitimacy", f"{analysis.legitimacy_score:.2f}")
        if analysis.summary:
            print_result("Summary", analysis.summary[:80] + "...")
        print()
    
    # Show agent state
    print_section("Agent State")
    agent_state = matcher.get_state()
    print_result("Campaigns Analyzed", str(len(matcher._campaign_catalog)))
    print_result("Reasoning Steps", str(agent_state['reasoning_steps']))
    print_result("Successful Actions", str(agent_state['successful_actions']))
    
    # Match to donor
    print_section("Matching Campaigns to Donor Profile")
    print_step("Finding campaigns that match donor interests...")
    
    matches = await matcher.match_campaigns_to_donor(
        donor_profile=donor_profile.to_dict(),
        limit=3
    )
    
    print("\n  Top Matches:")
    for i, match in enumerate(matches, 1):
        campaign = match["campaign"]
        print(f"\n  {i}. {campaign['title'][:50]}...")
        print(f"     Match Score: {match['match_score']:.2f}")
        print(f"     Reasons: {', '.join(match['reasons'][:2])}")
        if match.get("explanation"):
            print(f"     {Colors.CYAN}Explanation: {match['explanation'][:100]}...{Colors.END}")
    
    return matcher, analyzed


async def run_community_discovery():
    """Run the Community Discovery agent."""
    print_header("AGENT 3: COMMUNITY DISCOVERY")
    
    print("Creating agent and setting up social network...")
    agent = CommunityDiscoveryAgent()
    generator = SyntheticDataGenerator(seed=42)
    
    # Create user profiles with connections
    user1 = UserProfile(
        user_id="user_001",
        city="San Francisco",
        state="CA",
        employer="Google",
        schools=["Stanford University", "MIT"],
    )
    
    user2 = UserProfile(
        user_id="user_002",
        city="San Francisco",
        state="CA",
        employer="Google",
        schools=["Stanford University"],
    )
    
    # Add connection
    user1.connections.append(SocialConnection(
        user_id="user_001",
        connected_user_id="user_002",
        connection_type=ConnectionType.COLLEAGUE,
        strength=0.8,
    ))
    
    agent.register_user(user1)
    agent.register_user(user2)
    
    # Register campaigns
    campaigns = [generator.generate_campaign() for _ in range(5)]
    for campaign in campaigns:
        campaign["organizer_id"] = "user_002"
        campaign["organizer_employer"] = "Google"
        campaign["organizer_schools"] = ["Stanford University"]
        campaign["city"] = "San Francisco"
        campaign["state"] = "CA"
        agent.register_campaign(campaign)
    
    print_section("User Profile")
    print_result("User", user1.user_id)
    print_result("Location", f"{user1.city}, {user1.state}")
    print_result("Employer", user1.employer)
    print_result("Schools", ", ".join(user1.schools))
    print_result("Connections", str(len(user1.connections)))
    
    print_section("Discovering Community Campaigns")
    print_step("Mapping social network...")
    print_step("Finding connected campaigns...")
    
    discoveries = await agent.discover_for_user(user1.user_id, limit=5)
    
    print("\n  Discovered Campaigns:")
    for i, discovery in enumerate(discoveries[:3], 1):
        print(f"\n  {i}. {discovery.campaign_title[:50]}...")
        print(f"     Proximity Score: {discovery.proximity_score:.2f}")
        print(f"     Connection Types: {[p.value for p in discovery.proximity_types]}")
        if discovery.narrative:
            print(f"     {Colors.CYAN}Narrative: {discovery.narrative[:100]}...{Colors.END}")
    
    # Find workplace campaigns
    print_section("Workplace Giving Opportunities")
    print_step("Finding campaigns from colleagues...")
    
    workplace = await agent.find_workplace_giving(user1.user_id)
    print_result("Employer", workplace.get("employer", "N/A"))
    print_result("Campaigns Found", str(workplace.get("count", 0)))
    
    return agent


async def run_recurring_curator():
    """Run the Recurring Curator agent."""
    print_header("AGENT 4: RECURRING OPPORTUNITY CURATOR")
    
    print("Creating agent and generating campaigns...")
    curator = RecurringCuratorAgent()
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate campaigns suitable for recurring
    campaigns = []
    for i in range(5):
        campaign = generator.generate_campaign()
        campaign["description"] += " This is an ongoing need requiring sustained monthly support."
        campaign["updates"] = [{"date": datetime.utcnow().isoformat()} for _ in range(4)]
        campaigns.append(campaign)
    
    print_section("Curating Recurring Opportunities")
    print_step("Assessing campaigns for recurring suitability...")
    
    opportunities = await curator.curate_opportunities(campaigns)
    
    print("\n  Curated Opportunities:")
    for i, opp in enumerate(opportunities[:3], 1):
        print(f"\n  {i}. {opp.campaign_title[:50]}...")
        print(f"     Suitability: {opp.suitability.value} ({opp.suitability_score:.2f})")
        print(f"     Recommended: ${opp.recommended_amount:.0f}/{opp.recommended_frequency.value}")
        if opp.impact_per_period:
            print(f"     Impact: {opp.impact_per_period[:60]}...")
        if opp.pitch:
            print(f"     {Colors.CYAN}Pitch: {opp.pitch[:100]}...{Colors.END}")
    
    # Create a donor plan
    print_section("Creating Personalized Giving Plan")
    print_step("Designing optimal recurring plan for donor...")
    
    donor_profile = {
        "donor_id": "test_donor",
        "average_donation": 50,
        "cause_affinities": [
            {"category": "medical", "score": 0.8},
            {"category": "education", "score": 0.6},
        ],
    }
    
    plan = await curator.create_donor_plan(
        donor_id="test_donor",
        donor_profile=donor_profile,
        monthly_budget=75,
    )
    
    print_result("Plan ID", plan.plan_id)
    print_result("Total Monthly", f"${plan.total_monthly_commitment:.0f}")
    print_result("Opportunities", str(len(plan.opportunities)))
    if plan.plan_summary:
        print(f"\n  {Colors.BOLD}Plan Summary:{Colors.END}")
        print(f"  {plan.plan_summary[:200]}...")
    
    return curator


async def run_giving_circle():
    """Run the Giving Circle Orchestrator agent."""
    print_header("AGENT 5: GIVING CIRCLE ORCHESTRATOR")
    
    print("Creating agent...")
    orchestrator = GivingCircleOrchestrator()
    generator = SyntheticDataGenerator(seed=42)
    
    # Register some campaigns
    campaigns = [generator.generate_campaign() for _ in range(5)]
    for campaign in campaigns:
        orchestrator.register_campaign(campaign)
    
    print_section("Creating Giving Circle")
    print_step("Forming new giving circle...")
    
    circle = await orchestrator.create_circle(
        name="Bay Area Medical Support Circle",
        founder_id="founder_001",
        description="Supporting medical causes in the Bay Area community",
        circle_type="interest_based",
        cause_focus=["medical", "community"],
    )
    
    print_result("Circle ID", circle.circle_id)
    print_result("Name", circle.name)
    print_result("Status", circle.status.value)
    print_result("Members", str(len(circle.members)))
    if circle.mission_statement:
        print(f"\n  {Colors.BOLD}Mission Statement:{Colors.END}")
        print(f"  {circle.mission_statement[:200]}...")
    
    # Add members
    print_section("Growing the Circle")
    print_step("Adding members...")
    
    for i in range(3):
        result = await orchestrator.join_circle(
            circle_id=circle.circle_id,
            user_id=f"member_{i+1}",
            display_name=f"Member {i+1}",
        )
        print(f"  Added member_{i+1}: {result.get('status')}")
    
    # Get recommendations
    print_section("Campaign Recommendations")
    print_step("Finding campaigns for the circle to consider...")
    
    recommendations = await orchestrator._tool_recommend_campaigns(circle.circle_id, limit=3)
    
    print("\n  Recommended Campaigns:")
    for i, rec in enumerate(recommendations.get("recommendations", [])[:3], 1):
        print(f"\n  {i}. {rec['title'][:50]}...")
        print(f"     Match Score: {rec['match_score']:.2f}")
        print(f"     Reasons: {', '.join(rec['reasons'][:2])}")
    
    return orchestrator


async def run_engagement_agent():
    """Run the Engagement & Re-activation agent."""
    print_header("AGENT 6: ENGAGEMENT & RE-ACTIVATION")
    
    print("Creating agent...")
    agent = EngagementAgent()
    generator = SyntheticDataGenerator(seed=42)
    
    # Register donors with varying engagement levels
    donors = []
    for i in range(5):
        donor, donations = generator.generate_donor_with_history(num_donations=5)
        agent.register_donor(donor["donor_id"], donor)
        donors.append((donor, donations))
    
    print_section("Assessing Donor Engagement")
    
    for donor, donations in donors[:3]:
        print_step(f"Analyzing {donor['display_name']}...")
        
        # Calculate last donation date
        last_donation = max(d["timestamp"] for d in donations) if donations else None
        
        activity_data = {
            "last_donation": last_donation,
            "donation_count": len(donations),
            "last_login": (datetime.utcnow() - timedelta(days=15)).isoformat(),
            "interactions": 8,
            "email_opens": 5,
            "emails_sent": 10,
        }
        
        engagement = await agent.assess_donor_engagement(
            donor_id=donor["donor_id"],
            activity_data=activity_data,
        )
        
        print_result("Level", engagement.level.value)
        print_result("Score", f"{engagement.score:.2f}")
        print_result("Churn Risk", f"{engagement.churn_risk:.2f}")
        print()
    
    # Generate a nudge
    print_section("Generating Personalized Nudge")
    print_step("Creating engagement message...")
    
    donor = donors[0][0]
    nudge = await agent.send_nudge(
        donor_id=donor["donor_id"],
        nudge_type="impact_update",
        context={"recent_campaign": "Local Food Bank", "impact": "Fed 50 families"},
    )
    
    print_result("Nudge Type", nudge.nudge_type.value)
    print_result("Channel", nudge.channel.value)
    if nudge.subject:
        print_result("Subject", nudge.subject)
    if nudge.message:
        print(f"\n  {Colors.BOLD}Message:{Colors.END}")
        print(f"  {nudge.message[:200]}...")
    
    # Get at-risk donors
    print_section("At-Risk Donor Identification")
    print_step("Identifying donors needing attention...")
    
    at_risk = await agent.get_at_risk_donors(threshold=0.3)
    print_result("At-Risk Count", str(len(at_risk)))
    
    # Get trends
    print_section("Engagement Trends")
    trends = await agent.get_engagement_trends()
    print_result("Total Donors", str(trends.get("total_donors", 0)))
    print_result("Healthy %", f"{trends.get('healthy_percentage', 0):.1f}%")
    print_result("At-Risk %", f"{trends.get('at_risk_percentage', 0):.1f}%")
    
    return agent


async def run_full_pipeline():
    """Run the complete multi-agent pipeline."""
    print_header("FULL MULTI-AGENT PIPELINE EXECUTION")
    
    print(f"""
{Colors.BOLD}This will run all 6 autonomous agents in sequence:{Colors.END}

  1. Donor Affinity Profiler    - Build donor profile
  2. Campaign Matching Engine   - Analyze and match campaigns
  3. Community Discovery        - Find connected campaigns
  4. Recurring Curator          - Curate recurring opportunities
  5. Giving Circle Orchestrator - Facilitate collective giving
  6. Engagement Agent           - Manage donor relationships

Each agent uses LLM-powered planning, reasoning, and tool execution.
    """)
    
    start_time = datetime.now()
    
    # Run Agent 1
    profiler, profile, donor, donations = await run_donor_profiler()
    
    # Run Agent 2
    matcher, analyzed = await run_campaign_matcher(profile)
    
    # Run Agent 3
    community_agent = await run_community_discovery()
    
    # Run Agent 4
    curator = await run_recurring_curator()
    
    # Run Agent 5
    circle_agent = await run_giving_circle()
    
    # Run Agent 6
    engagement_agent = await run_engagement_agent()
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print_header("PIPELINE COMPLETE")
    
    print(f"""
{Colors.BOLD}Execution Summary:{Colors.END}
  • Total Duration: {duration:.1f} seconds
  • Agents Executed: 6
  • Donor Profiles Built: 1
  • Campaigns Analyzed: {len(matcher._campaign_catalog)}
  • Recurring Opportunities: {len(curator._opportunities)}
  • Giving Circles: {len(circle_agent._circles)}

{Colors.BOLD}Agent Performance:{Colors.END}
  • Profiler Reasoning Steps: {len(profiler._memory.reasoning_history)}
  • Matcher Reasoning Steps: {len(matcher._memory.reasoning_history)}
  • Engagement Actions: {engagement_agent.get_state()['successful_actions']}

{Colors.GREEN}All agents executed successfully with autonomous planning and reasoning.{Colors.END}
    """)


async def main():
    """Main entry point."""
    print(f"""
{Colors.BOLD}{Colors.HEADER}
========================================================================
         GIVING INTELLIGENCE PLATFORM - AUTONOMOUS AGENTS             
                        REAL EXECUTION MODE                           
========================================================================
{Colors.END}

This script executes the autonomous agents with real LLM calls.
Each agent performs planning, reasoning, and tool execution.

{Colors.BOLD}Options:{Colors.END}
  1. Run full pipeline (all 6 agents)
  2. Run Donor Affinity Profiler only
  3. Run Campaign Matching Engine only
  4. Run Community Discovery only
  5. Run Recurring Curator only
  6. Run Giving Circle Orchestrator only
  7. Run Engagement Agent only
  0. Exit

""")
    
    # Check for command-line argument
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        print(f"Running option {choice} from command line...")
    else:
        choice = input(f"{Colors.YELLOW}Enter your choice (1-7, 0 to exit): {Colors.END}").strip()
    
    if choice == "0":
        print("Goodbye!")
        return
    elif choice == "1":
        await run_full_pipeline()
    elif choice == "2":
        await run_donor_profiler()
    elif choice == "3":
        # Need a profile first
        profiler, profile, _, _ = await run_donor_profiler()
        await run_campaign_matcher(profile)
    elif choice == "4":
        await run_community_discovery()
    elif choice == "5":
        await run_recurring_curator()
    elif choice == "6":
        await run_giving_circle()
    elif choice == "7":
        await run_engagement_agent()
    else:
        print("Invalid choice. Running full pipeline...")
        await run_full_pipeline()


if __name__ == "__main__":
    asyncio.run(main())

