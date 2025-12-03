#!/usr/bin/env python3
"""
Demo script for the Giving Intelligence Platform.
Demonstrates autonomous agent capabilities including planning, reasoning, and tool use.
"""

import asyncio
import json
from datetime import datetime

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.donor_affinity_profiler import DonorAffinityProfiler
from src.agents.campaign_matching_engine import CampaignMatchingEngine
from src.data.synthetic import SyntheticDataGenerator


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title: str):
    """Print a subsection header."""
    print(f"\n  --- {title} ---")


def print_json(data: dict, indent: int = 2):
    """Pretty print JSON data."""
    print(json.dumps(data, indent=indent, default=str))


async def demo_autonomous_profiler():
    """Demonstrate the autonomous Donor Affinity Profiler."""
    print_section("🤖 AUTONOMOUS DONOR AFFINITY PROFILER")
    
    print("""
    This agent autonomously:
    1. Plans how to analyze the donor
    2. Reasons through the data step-by-step
    3. Uses specialized tools to extract insights
    4. Generates LLM-powered personality analysis
    5. Reflects on confidence and completeness
    """)
    
    # Create the autonomous agent
    profiler = DonorAffinityProfiler()
    
    # Generate synthetic donor data
    generator = SyntheticDataGenerator(seed=42)
    donor, donations = generator.generate_donor_with_history(num_donations=15)
    
    print_subsection("Input: Donor Data")
    print(f"  Donor: {donor['display_name']}")
    print(f"  Location: {donor['city']}, {donor['state']}")
    print(f"  Employer: {donor.get('employer', 'N/A')}")
    print(f"  Donations to analyze: {len(donations)}")
    
    print_subsection("Agent Execution (Autonomous)")
    print("  Starting autonomous profile building...")
    print("  The agent will plan, reason, and execute tools autonomously.\n")
    
    # Run the autonomous agent
    profile = await profiler.build_profile(
        donor_id=donor["donor_id"],
        donations=donations,
        metadata={"location": f"{donor['city']}, {donor['state']}"}
    )
    
    # Display the agent's state
    print_subsection("Agent State After Execution")
    agent_state = profiler.get_state()
    print(f"  Tasks executed: {len(agent_state['tasks'])}")
    print(f"  Reasoning steps: {agent_state['reasoning_steps']}")
    print(f"  Successful actions: {agent_state['successful_actions']}")
    print(f"  Failed actions: {agent_state['failed_actions']}")
    
    # Display reasoning trace (sample)
    print_subsection("Sample Reasoning Trace")
    for step in profiler._memory.reasoning_history[:3]:
        print(f"  [{step.reasoning_type.value}] {step.thought[:100]}...")
    
    # Display the profile
    print_subsection("Generated Profile")
    print(f"  Profile Completeness: {profile.profile_completeness * 100:.0f}%")
    print(f"\n  Financial Summary:")
    print(f"    - Total Lifetime Giving: ${profile.total_lifetime_giving:,.2f}")
    print(f"    - Average Donation: ${profile.average_donation:,.2f}")
    print(f"    - Donation Count: {profile.donation_count}")
    
    print(f"\n  Top Cause Affinities:")
    for aff in profile.cause_affinities[:3]:
        print(f"    - {aff.category.value}: {aff.score:.2f} (confidence: {aff.confidence:.2f})")
        if aff.llm_insights:
            print(f"      LLM Insight: {aff.llm_insights[:80]}...")
    
    print(f"\n  Giving Motivators:")
    for motivator, score in list(profile.giving_motivators.items())[:3]:
        print(f"    - {motivator.value}: {score:.2f}")
    
    print(f"\n  Engagement Scores:")
    print(f"    - Engagement: {profile.engagement_score:.2f}")
    print(f"    - Recency: {profile.recency_score:.2f}")
    print(f"    - Loyalty: {profile.loyalty_score:.2f}")
    
    if profile.personality_summary:
        print(f"\n  🧠 LLM-Generated Personality Summary:")
        print(f"    {profile.personality_summary[:200]}...")
    
    if profile.engagement_recommendations:
        print(f"\n  📋 Engagement Recommendations:")
        for rec in profile.engagement_recommendations[:3]:
            print(f"    • {rec}")
    
    return profiler, profile, donor


async def demo_autonomous_matching():
    """Demonstrate the autonomous Campaign Matching Engine."""
    print_section("🎯 AUTONOMOUS CAMPAIGN MATCHING ENGINE")
    
    print("""
    This agent autonomously:
    1. Deeply analyzes campaign content using LLM
    2. Builds rich taxonomy beyond simple categories
    3. Assesses urgency and legitimacy through reasoning
    4. Generates semantic embeddings for matching
    5. Creates personalized match explanations
    """)
    
    # Create the autonomous agent
    matcher = CampaignMatchingEngine()
    
    # Generate synthetic campaigns
    generator = SyntheticDataGenerator(seed=42)
    campaigns = [generator.generate_campaign() for _ in range(5)]
    
    print_subsection("Analyzing Campaigns Autonomously")
    
    analyzed = []
    for i, campaign in enumerate(campaigns[:3]):
        print(f"\n  Campaign {i+1}: {campaign['title'][:50]}...")
        print(f"  Agent is planning and executing analysis...")
        
        analysis = await matcher.analyze_campaign(campaign)
        analyzed.append(analysis)
        
        print(f"    ✓ Category: {analysis.taxonomy.primary_category}")
        print(f"    ✓ Urgency: {analysis.urgency_level.value} ({analysis.urgency_score:.2f})")
        print(f"    ✓ Legitimacy: {analysis.legitimacy_score:.2f}")
        if analysis.summary:
            print(f"    ✓ Summary: {analysis.summary[:80]}...")
    
    # Show agent state
    print_subsection("Agent State")
    agent_state = matcher.get_state()
    print(f"  Campaigns in catalog: {len(matcher._campaign_catalog)}")
    print(f"  Reasoning steps: {agent_state['reasoning_steps']}")
    
    return matcher, analyzed


async def demo_matching_workflow(profiler, profile, matcher):
    """Demonstrate matching campaigns to a donor profile."""
    print_section("🔗 AUTONOMOUS MATCHING WORKFLOW")
    
    print("""
    Now we'll match campaigns to the donor profile.
    The agent will:
    1. Reason about donor-campaign alignment
    2. Generate personalized match explanations
    3. Rank matches by relevance
    """)
    
    # Match campaigns to the donor
    matches = await matcher.match_campaigns_to_donor(
        donor_profile=profile.to_dict(),
        limit=5
    )
    
    print_subsection("Top Matches")
    for i, match in enumerate(matches[:3], 1):
        campaign = match["campaign"]
        print(f"\n  {i}. {campaign['title'][:50]}...")
        print(f"     Match Score: {match['match_score']:.2f}")
        print(f"     Reasons: {', '.join(match['reasons'][:2])}")
        if match.get("explanation"):
            print(f"     🤖 Personalized Explanation:")
            print(f"        {match['explanation'][:150]}...")


async def demo_ad_hoc_reasoning():
    """Demonstrate ad-hoc reasoning capabilities."""
    print_section("💭 AD-HOC REASONING DEMONSTRATION")
    
    print("""
    Autonomous agents can answer arbitrary questions through reasoning.
    Let's ask the profiler agent to analyze a specific question.
    """)
    
    profiler = DonorAffinityProfiler()
    generator = SyntheticDataGenerator(seed=123)
    donor, donations = generator.generate_donor_with_history(num_donations=10)
    
    # First build a profile
    profile = await profiler.build_profile(donor["donor_id"], donations)
    
    # Now ask an ad-hoc question
    question = "What type of campaign would most likely inspire this donor to make a larger than usual donation?"
    
    print_subsection(f"Question")
    print(f"  {question}")
    
    print_subsection("Agent Reasoning...")
    response = await profiler.analyze_donor(
        donor_id=donor["donor_id"],
        question=question,
        donations=donations
    )
    
    print_subsection("Agent Response")
    print(f"  {response[:500]}...")


async def main():
    """Run the complete demo."""
    print("\n" + "=" * 70)
    print("  🎁 GIVING INTELLIGENCE PLATFORM - AUTONOMOUS AGENTS DEMO")
    print("=" * 70)
    
    print("""
    This demo showcases LLM-based autonomous agents that perform:
    • Planning - Breaking down goals into executable tasks
    • Reasoning - Chain-of-thought analysis
    • Tool Use - Executing specialized functions
    • Task Decomposition - Managing multi-step workflows
    • Reflection - Learning and adjusting strategies
    """)
    
    # Demo 1: Autonomous Donor Profiling
    profiler, profile, donor = await demo_autonomous_profiler()
    
    # Demo 2: Autonomous Campaign Analysis
    matcher, analyzed = await demo_autonomous_matching()
    
    # Demo 3: Matching Workflow
    await demo_matching_workflow(profiler, profile, matcher)
    
    # Demo 4: Ad-hoc Reasoning
    await demo_ad_hoc_reasoning()
    
    # Summary
    print_section("✅ DEMO COMPLETE")
    print("""
    Key Takeaways:
    
    1. AUTONOMOUS OPERATION
       Agents plan and execute without step-by-step instructions
    
    2. LLM-POWERED REASONING
       Deep semantic understanding through chain-of-thought
    
    3. TOOL ORCHESTRATION
       Agents select and use tools based on task requirements
    
    4. PERSONALIZATION
       Every insight and recommendation is contextual
    
    5. TRANSPARENCY
       Full reasoning traces available for inspection
    
    The platform transforms transactional giving into
    relationship-driven donor engagement through intelligent,
    autonomous agent collaboration.
    """)


if __name__ == "__main__":
    asyncio.run(main())
