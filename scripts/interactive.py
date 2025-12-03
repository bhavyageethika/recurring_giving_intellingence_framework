#!/usr/bin/env python3
"""
Interactive Pipeline - Real Data Input

This script allows you to input your own data and get meaningful analysis
from the autonomous agents.
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.donor_affinity_profiler import DonorAffinityProfiler, DonorProfile
from src.agents.campaign_matching_engine import CampaignMatchingEngine
from src.agents.recurring_curator import RecurringCuratorAgent
from src.agents.engagement_agent import EngagementAgent


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}  {text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}\n")


def print_section(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- {text} ---{Colors.END}")


def get_user_donations() -> List[Dict[str, Any]]:
    """Interactively collect donation history from user."""
    print_section("Enter Your Donation History")
    print("""
I'll ask you about your past donations. This helps me understand
your giving patterns and preferences.

For each donation, I'll ask:
- Campaign name/cause
- Amount donated
- Category (medical, education, emergency, community, animals, etc.)
- When you donated (approximate)
""")
    
    donations = []
    
    while True:
        print(f"\n{Colors.YELLOW}Donation #{len(donations) + 1}{Colors.END}")
        
        # Campaign name
        campaign = input("Campaign name (or 'done' to finish): ").strip()
        if campaign.lower() == 'done':
            break
        if not campaign:
            continue
            
        # Amount
        try:
            amount = float(input("Amount donated ($): ").strip().replace('$', '').replace(',', ''))
        except ValueError:
            print("Invalid amount, skipping...")
            continue
        
        # Category
        print("Categories: medical, education, emergency, community, animals, children, veterans, environment, other")
        category = input("Category: ").strip().lower() or "other"
        
        # Date
        date_str = input("When? (e.g., '2 months ago', 'last year', or YYYY-MM-DD): ").strip()
        donation_date = parse_relative_date(date_str)
        
        # Why did you donate?
        reason = input("Why did you donate? (optional): ").strip()
        
        donations.append({
            "campaign_title": campaign,
            "campaign_category": category,
            "amount": amount,
            "timestamp": donation_date.isoformat(),
            "source": "direct",
            "notes": reason,
        })
        
        print(f"{Colors.GREEN}Added: ${amount:.2f} to '{campaign}'{Colors.END}")
    
    if not donations:
        print("\nNo donations entered. Using sample data for demo...")
        donations = get_sample_donations()
    
    return donations


def parse_relative_date(date_str: str) -> datetime:
    """Parse relative date strings like '2 months ago'."""
    date_str = date_str.lower().strip()
    now = datetime.now()
    
    if not date_str:
        return now - timedelta(days=30)
    
    try:
        return datetime.fromisoformat(date_str)
    except:
        pass
    
    if 'yesterday' in date_str:
        return now - timedelta(days=1)
    elif 'last week' in date_str:
        return now - timedelta(weeks=1)
    elif 'last month' in date_str:
        return now - timedelta(days=30)
    elif 'last year' in date_str:
        return now - timedelta(days=365)
    elif 'month' in date_str:
        try:
            months = int(''.join(filter(str.isdigit, date_str)) or '1')
            return now - timedelta(days=months * 30)
        except:
            return now - timedelta(days=30)
    elif 'week' in date_str:
        try:
            weeks = int(''.join(filter(str.isdigit, date_str)) or '1')
            return now - timedelta(weeks=weeks)
        except:
            return now - timedelta(weeks=1)
    elif 'year' in date_str:
        try:
            years = int(''.join(filter(str.isdigit, date_str)) or '1')
            return now - timedelta(days=years * 365)
        except:
            return now - timedelta(days=365)
    elif 'day' in date_str:
        try:
            days = int(''.join(filter(str.isdigit, date_str)) or '1')
            return now - timedelta(days=days)
        except:
            return now - timedelta(days=1)
    
    return now - timedelta(days=30)


def get_sample_donations() -> List[Dict[str, Any]]:
    """Return sample donations for demo."""
    now = datetime.now()
    return [
        {
            "campaign_title": "Help Sarah fight cancer",
            "campaign_category": "medical",
            "amount": 100,
            "timestamp": (now - timedelta(days=15)).isoformat(),
            "source": "direct",
        },
        {
            "campaign_title": "Local food bank support",
            "campaign_category": "community",
            "amount": 50,
            "timestamp": (now - timedelta(days=45)).isoformat(),
            "source": "shared_link",
        },
        {
            "campaign_title": "School supplies for kids",
            "campaign_category": "education",
            "amount": 25,
            "timestamp": (now - timedelta(days=90)).isoformat(),
            "source": "direct",
        },
    ]


def get_user_info() -> Dict[str, Any]:
    """Get basic user information."""
    print_section("Tell Me About Yourself")
    
    name = input("Your name (or press Enter to skip): ").strip() or "Anonymous Donor"
    city = input("Your city: ").strip() or "Unknown"
    state = input("Your state: ").strip() or "Unknown"
    employer = input("Your employer (optional): ").strip()
    
    return {
        "donor_id": f"donor_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "display_name": name,
        "city": city,
        "state": state,
        "employer": employer,
    }


def get_campaigns_to_analyze() -> List[Dict[str, Any]]:
    """Get campaigns the user wants analyzed."""
    print_section("Campaigns to Analyze")
    print("""
Enter campaigns you're considering donating to.
I'll analyze them and tell you which ones match your giving profile.
""")
    
    campaigns = []
    
    while True:
        print(f"\n{Colors.YELLOW}Campaign #{len(campaigns) + 1}{Colors.END}")
        
        title = input("Campaign title (or 'done' to finish): ").strip()
        if title.lower() == 'done':
            break
        if not title:
            continue
        
        description = input("Brief description: ").strip()
        
        print("Categories: medical, education, emergency, community, animals, children, veterans, environment, other")
        category = input("Category: ").strip().lower() or "other"
        
        try:
            goal = float(input("Funding goal ($): ").strip().replace('$', '').replace(',', '') or "5000")
        except:
            goal = 5000
            
        try:
            raised = float(input("Amount raised so far ($): ").strip().replace('$', '').replace(',', '') or "0")
        except:
            raised = 0
        
        location = input("Campaign location (city, state): ").strip()
        
        campaigns.append({
            "campaign_id": f"camp_{len(campaigns)+1}",
            "title": title,
            "description": description,
            "category": category,
            "goal_amount": goal,
            "raised_amount": raised,
            "location": location,
            "created_at": datetime.now().isoformat(),
        })
        
        print(f"{Colors.GREEN}Added: '{title}'{Colors.END}")
    
    if not campaigns:
        print("\nNo campaigns entered. Using sample campaigns...")
        campaigns = get_sample_campaigns()
    
    return campaigns


def get_sample_campaigns() -> List[Dict[str, Any]]:
    """Return sample campaigns for demo."""
    return [
        {
            "campaign_id": "camp_1",
            "title": "Help John recover from surgery",
            "description": "John needs help with medical bills after an unexpected surgery. He's a single father of two.",
            "category": "medical",
            "goal_amount": 15000,
            "raised_amount": 8500,
            "location": "Dallas, TX",
            "created_at": datetime.now().isoformat(),
        },
        {
            "campaign_id": "camp_2", 
            "title": "Community garden project",
            "description": "Building a community garden to provide fresh produce for local families in need.",
            "category": "community",
            "goal_amount": 5000,
            "raised_amount": 2100,
            "location": "Austin, TX",
            "created_at": datetime.now().isoformat(),
        },
        {
            "campaign_id": "camp_3",
            "title": "Scholarship fund for first-gen students",
            "description": "Helping first-generation college students afford tuition and books.",
            "category": "education",
            "goal_amount": 25000,
            "raised_amount": 12000,
            "location": "Houston, TX",
            "created_at": datetime.now().isoformat(),
        },
    ]


def display_profile(profile: DonorProfile):
    """Display the donor profile in a readable format."""
    print_section("YOUR GIVING PROFILE")
    
    print(f"\n{Colors.BOLD}Profile Completeness:{Colors.END} {profile.profile_completeness * 100:.0f}%")
    
    print(f"\n{Colors.BOLD}Financial Summary:{Colors.END}")
    print(f"  Total Lifetime Giving: ${profile.total_lifetime_giving:,.2f}")
    print(f"  Average Donation: ${profile.average_donation:,.2f}")
    print(f"  Number of Donations: {profile.donation_count}")
    
    if profile.cause_affinities:
        print(f"\n{Colors.BOLD}Your Top Cause Affinities:{Colors.END}")
        for i, aff in enumerate(profile.cause_affinities[:5], 1):
            bar = "=" * int(aff.score * 20)
            print(f"  {i}. {aff.category.value.title()}: {bar} ({aff.score:.0%})")
            if aff.llm_insights:
                print(f"     {Colors.CYAN}Insight: {aff.llm_insights[:80]}...{Colors.END}")
    
    if profile.giving_motivators:
        print(f"\n{Colors.BOLD}What Motivates You to Give:{Colors.END}")
        for motivator, score in sorted(profile.giving_motivators.items(), key=lambda x: x[1], reverse=True)[:5]:
            bar = "=" * int(score * 20)
            print(f"  - {motivator.value.replace('_', ' ').title()}: {bar} ({score:.0%})")
    
    print(f"\n{Colors.BOLD}Engagement Scores:{Colors.END}")
    print(f"  Engagement Level: {profile.engagement_score:.0%}")
    print(f"  Recency (how recent): {profile.recency_score:.0%}")
    print(f"  Loyalty: {profile.loyalty_score:.0%}")
    
    if profile.primary_pattern:
        print(f"\n{Colors.BOLD}Your Giving Style:{Colors.END} {profile.primary_pattern.value.replace('_', ' ').title()}")
    
    if profile.personality_summary:
        print(f"\n{Colors.BOLD}AI-Generated Personality Summary:{Colors.END}")
        print(f"  {profile.personality_summary}")
    
    if profile.engagement_recommendations:
        print(f"\n{Colors.BOLD}Recommendations for You:{Colors.END}")
        for i, rec in enumerate(profile.engagement_recommendations[:5], 1):
            print(f"  {i}. {rec}")
    
    if profile.predicted_interests:
        print(f"\n{Colors.BOLD}Predicted Future Interests:{Colors.END}")
        for interest in profile.predicted_interests[:5]:
            print(f"  - {interest}")


def display_matches(matches: List[Dict[str, Any]], profile: DonorProfile):
    """Display campaign matches."""
    print_section("CAMPAIGN RECOMMENDATIONS FOR YOU")
    
    if not matches:
        print("No strong matches found. Try adding more campaigns to analyze.")
        return
    
    for i, match in enumerate(matches, 1):
        campaign = match.get("campaign", {})
        score = match.get("match_score", 0)
        
        print(f"\n{Colors.BOLD}{Colors.GREEN}#{i}: {campaign.get('title', 'Unknown')}{Colors.END}")
        print(f"    Match Score: {'*' * int(score * 10)} ({score:.0%})")
        print(f"    Category: {campaign.get('taxonomy', {}).get('primary_category', 'Unknown')}")
        
        if match.get("reasons"):
            print(f"    Why it matches you:")
            for reason in match["reasons"][:3]:
                print(f"      - {reason}")
        
        if match.get("explanation"):
            print(f"\n    {Colors.CYAN}AI Explanation:{Colors.END}")
            print(f"    {match['explanation'][:200]}...")


async def run_interactive_pipeline():
    """Run the full interactive pipeline."""
    print_header("GIVING INTELLIGENCE - INTERACTIVE MODE")
    
    print(f"""
{Colors.BOLD}Welcome to the Giving Intelligence Platform!{Colors.END}

This system will:
1. Learn about your giving history
2. Build your personalized donor profile
3. Analyze campaigns you're interested in
4. Match you with campaigns that fit your values
5. Suggest recurring giving opportunities

Let's get started!
""")
    
    # Step 1: Get user info
    user_info = get_user_info()
    
    # Step 2: Get donation history
    donations = get_user_donations()
    
    print(f"\n{Colors.GREEN}Collected {len(donations)} donations.{Colors.END}")
    print(f"Total given: ${sum(d['amount'] for d in donations):,.2f}")
    
    # Step 3: Build donor profile
    print_header("ANALYZING YOUR GIVING PROFILE")
    print("Running Donor Affinity Profiler agent...")
    
    profiler = DonorAffinityProfiler()
    profile = await profiler.build_profile(
        donor_id=user_info["donor_id"],
        donations=donations,
        metadata={
            "location": f"{user_info['city']}, {user_info['state']}",
            "employer": user_info.get("employer"),
        }
    )
    
    display_profile(profile)
    
    # Step 4: Get campaigns to analyze
    input(f"\n{Colors.YELLOW}Press Enter to continue to campaign analysis...{Colors.END}")
    
    campaigns = get_campaigns_to_analyze()
    
    # Step 5: Analyze and match campaigns
    print_header("ANALYZING CAMPAIGNS")
    print("Running Campaign Matching Engine agent...")
    
    matcher = CampaignMatchingEngine()
    
    for campaign in campaigns:
        print(f"  Analyzing: {campaign['title'][:40]}...")
        await matcher.analyze_campaign(campaign)
    
    # Step 6: Match to donor profile
    print("\nMatching campaigns to your profile...")
    matches = await matcher.match_campaigns_to_donor(
        donor_profile=profile.to_dict(),
        limit=len(campaigns)
    )
    
    display_matches(matches, profile)
    
    # Step 7: Recurring opportunities
    print_header("RECURRING GIVING OPPORTUNITIES")
    
    curator = RecurringCuratorAgent()
    
    # Register campaigns
    for campaign in campaigns:
        campaign["description"] = campaign.get("description", "") + " This is an ongoing need."
        campaign["updates"] = [{"date": datetime.now().isoformat()}] * 3
        curator.register_campaign(campaign)
    
    opportunities = await curator.curate_opportunities()
    
    if opportunities:
        print(f"\n{Colors.BOLD}Campaigns Suitable for Monthly Giving:{Colors.END}")
        for i, opp in enumerate(opportunities[:3], 1):
            print(f"\n  {i}. {opp.campaign_title}")
            print(f"     Suitability: {opp.suitability.value}")
            print(f"     Recommended: ${opp.recommended_amount:.0f}/month")
            if opp.impact_per_period:
                print(f"     Impact: {opp.impact_per_period}")
            if opp.pitch:
                print(f"     {Colors.CYAN}{opp.pitch[:150]}...{Colors.END}")
    
    # Step 8: Monthly budget planning
    print_section("CREATE YOUR GIVING PLAN")
    
    try:
        budget = float(input("\nHow much would you like to give monthly? ($): ").strip().replace('$', '') or "50")
    except:
        budget = 50
    
    plan = await curator.create_donor_plan(
        donor_id=user_info["donor_id"],
        donor_profile=profile.to_dict(),
        monthly_budget=budget,
    )
    
    if plan.opportunities:
        print(f"\n{Colors.BOLD}Your Personalized Giving Plan:{Colors.END}")
        print(f"  Total Monthly: ${plan.total_monthly_commitment:.2f}")
        print(f"\n  Allocation:")
        for opp in plan.opportunities:
            print(f"    - ${opp.recommended_amount:.0f}/month -> {opp.campaign_title[:40]}")
        
        if plan.plan_summary:
            print(f"\n  {Colors.CYAN}Summary: {plan.plan_summary}{Colors.END}")
    
    # Final summary
    print_header("SESSION COMPLETE")
    
    print(f"""
{Colors.BOLD}What We Learned About You:{Colors.END}

- You've donated ${profile.total_lifetime_giving:,.2f} across {profile.donation_count} campaigns
- Your top cause: {profile.cause_affinities[0].category.value if profile.cause_affinities else 'Unknown'}
- Engagement level: {profile.engagement_score:.0%}
- Recommended monthly giving: ${plan.total_monthly_commitment:.2f}/month

{Colors.BOLD}Next Steps:{Colors.END}
1. Review the campaign matches above
2. Consider the recurring giving plan
3. Run this again with more donation history for better insights

{Colors.GREEN}Thank you for using Giving Intelligence!{Colors.END}
""")
    
    # Save results
    save = input("Save your profile to a file? (y/n): ").strip().lower()
    if save == 'y':
        output = {
            "user_info": user_info,
            "donations": donations,
            "profile": profile.to_dict(),
            "matches": matches,
            "plan": plan.to_dict() if plan else None,
            "generated_at": datetime.now().isoformat(),
        }
        
        filename = f"giving_profile_{user_info['donor_id']}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n{Colors.GREEN}Saved to {filename}{Colors.END}")


async def run_quick_demo():
    """Run a quick demo with sample data."""
    print_header("QUICK DEMO MODE")
    
    print("Using sample data to demonstrate the system...\n")
    
    # Sample user
    user_info = {
        "donor_id": "demo_donor",
        "display_name": "Demo User",
        "city": "Dallas",
        "state": "TX",
        "employer": "Tech Company",
    }
    
    # Sample donations
    donations = get_sample_donations()
    
    print(f"Sample donor: {user_info['display_name']} from {user_info['city']}, {user_info['state']}")
    print(f"Sample donations: {len(donations)} totaling ${sum(d['amount'] for d in donations):,.2f}")
    
    # Build profile
    print("\n[1/4] Building donor profile...")
    profiler = DonorAffinityProfiler()
    profile = await profiler.build_profile(
        donor_id=user_info["donor_id"],
        donations=donations,
        metadata={"location": f"{user_info['city']}, {user_info['state']}"}
    )
    
    display_profile(profile)
    
    # Analyze campaigns
    print("\n[2/4] Analyzing sample campaigns...")
    campaigns = get_sample_campaigns()
    
    matcher = CampaignMatchingEngine()
    for campaign in campaigns:
        await matcher.analyze_campaign(campaign)
    
    # Match
    print("\n[3/4] Matching campaigns to profile...")
    matches = await matcher.match_campaigns_to_donor(
        donor_profile=profile.to_dict(),
        limit=3
    )
    
    display_matches(matches, profile)
    
    # Recurring
    print("\n[4/4] Finding recurring opportunities...")
    curator = RecurringCuratorAgent()
    for campaign in campaigns:
        campaign["updates"] = [{"date": datetime.now().isoformat()}] * 3
        curator.register_campaign(campaign)
    
    opportunities = await curator.curate_opportunities()
    
    print_section("RECURRING OPPORTUNITIES")
    for opp in opportunities[:3]:
        print(f"  - {opp.campaign_title}: ${opp.recommended_amount}/month")
    
    print_header("DEMO COMPLETE")
    print("Run 'python scripts/interactive.py' and choose option 1 for full interactive mode!")


async def main():
    print(f"""
{Colors.BOLD}GIVING INTELLIGENCE PLATFORM{Colors.END}
{Colors.CYAN}Interactive Data Input Mode{Colors.END}

Choose an option:
  1. Full Interactive Mode (enter your own data)
  2. Quick Demo (use sample data)
  0. Exit
""")
    
    choice = input("Your choice (1/2/0): ").strip()
    
    if choice == "1":
        await run_interactive_pipeline()
    elif choice == "2":
        await run_quick_demo()
    else:
        print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())





