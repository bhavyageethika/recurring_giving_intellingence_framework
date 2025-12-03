#!/usr/bin/env python3
"""
Campaign Intelligence Analyzer

An agentic system that analyzes campaigns using AI agents.
Uses intelligent data acquisition (manual entry, JSON import, semantic discovery)
instead of web scraping.
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.donor_affinity_profiler import DonorAffinityProfiler
from src.agents.campaign_matching_engine import CampaignMatchingEngine
from src.agents.recurring_curator import RecurringCuratorAgent
from src.agents.campaign_data_agent import CampaignDataAgent


# ANSI colors for Windows
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def safe_print(text: str):
    """Print with encoding safety for Windows."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode())


def print_header(text: str):
    safe_print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.END}")
    safe_print(f"{Colors.BOLD}{Colors.HEADER}  {text}{Colors.END}")
    safe_print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.END}\n")


def print_section(text: str):
    safe_print(f"\n{Colors.BOLD}{Colors.CYAN}--- {text} ---{Colors.END}")


def display_campaign(campaign: Dict[str, Any], index: int = 0):
    """Display campaign details."""
    safe_print(f"\n{Colors.BOLD}[{index}] {campaign.get('title', 'Unknown Campaign')}{Colors.END}")
    
    safe_print(f"    Category: {campaign.get('category', 'Unknown')}")
    safe_print(f"    Location: {campaign.get('location', 'Unknown')}")
    safe_print(f"    Organizer: {campaign.get('organizer_name', 'Unknown')}")
    
    goal = campaign.get('goal_amount', 0)
    raised = campaign.get('raised_amount', 0)
    pct = (raised / goal * 100) if goal > 0 else 0
    
    safe_print(f"    Goal: ${goal:,.0f}")
    safe_print(f"    Raised: ${raised:,.0f} ({pct:.1f}%)")
    safe_print(f"    Donors: {campaign.get('donor_count', 0):,}")
    
    desc = campaign.get('description', '')[:150]
    if desc:
        safe_print(f"\n  Description: {desc}...")


async def analyze_with_agents(campaigns: List[Dict[str, Any]], donor_donations: List[Dict] = None) -> List[Any]:
    """Analyze campaigns using AI agents. Returns recurring opportunities."""
    
    print_section("AI AGENT ANALYSIS")
    
    # Initialize agents
    matcher = CampaignMatchingEngine()
    curator = RecurringCuratorAgent()
    profiler = DonorAffinityProfiler()
    
    # Build donor profile if we have donations
    donor_profile = None
    if donor_donations:
        safe_print("\n  Building your donor profile...")
        try:
            profile = await profiler.build_profile(
                donor_id=f"user_{datetime.now().strftime('%Y%m%d')}",
                donations=donor_donations,
            )
            donor_profile = profile.to_dict()
            
            safe_print(f"\n  {Colors.GREEN}Your Giving Profile:{Colors.END}")
            safe_print(f"    Total Given: ${profile.total_lifetime_giving:.0f}")
            
            # Derive giving style from primary pattern
            giving_style = profile.primary_pattern.value if profile.primary_pattern else "General"
            safe_print(f"    Giving Style: {giving_style}")
            
            # Get top causes from affinities
            top_causes = [ca.category.value for ca in sorted(profile.cause_affinities, key=lambda x: x.score, reverse=True)[:3]]
            if top_causes:
                safe_print(f"    Top Causes: {', '.join(top_causes)}")
            
            # Use personality summary as LLM insight
            if profile.personality_summary:
                safe_print(f"\n    {Colors.CYAN}AI Insight: {profile.personality_summary[:200]}...{Colors.END}")
        except Exception as e:
            safe_print(f"  {Colors.YELLOW}Profile building encountered issues: {str(e)}{Colors.END}")
            import traceback
            safe_print(f"  {Colors.YELLOW}Details: {traceback.format_exc()[:200]}...{Colors.END}")
            safe_print(f"  {Colors.CYAN}Continuing without donor profile...{Colors.END}")
    
    # Analyze each campaign
    safe_print("\n  Analyzing campaigns...")
    
    for i, campaign in enumerate(campaigns):
        safe_print(f"\n  [{i+1}/{len(campaigns)}] {campaign.get('title', 'Unknown')[:40]}...")
        
        # Prepare campaign data
        title = campaign.get("title", "")
        if not title:
            title = "Campaign"
        
        campaign_data = {
            "campaign_id": campaign.get("campaign_id", f"camp_{i}"),
            "title": title,
            "description": campaign.get("description", ""),
            "category": campaign.get("category", ""),
            "goal_amount": campaign.get("goal_amount", 0),
            "raised_amount": campaign.get("raised_amount", 0),
            "donor_count": campaign.get("donor_count", 0),
            "location": campaign.get("location", ""),
            "organizer_name": campaign.get("organizer_name", ""),
            "url": campaign.get("url", ""),
            "created_at": datetime.now().isoformat(),
            "updates": [{"date": datetime.now().isoformat()}],
        }
        
        try:
            # Analyze campaign
            analysis = await matcher.analyze_campaign(campaign_data)
            
            safe_print(f"    Primary Category: {analysis.taxonomy.primary_category}")
            safe_print(f"    Urgency: {analysis.urgency_level.value} ({analysis.urgency_score:.2f})")
            safe_print(f"    Legitimacy Score: {analysis.legitimacy_score:.2f}")
            
            if analysis.key_themes:
                safe_print(f"    Key Themes: {', '.join(analysis.key_themes[:3])}")
            
            if analysis.summary:
                safe_print(f"    {Colors.CYAN}AI Summary: {analysis.summary[:150]}...{Colors.END}")
            
            # Register for recurring analysis
            curator.register_campaign(campaign_data)
            
        except Exception as e:
            safe_print(f"    {Colors.YELLOW}Analysis encountered issues: {str(e)[:50]}{Colors.END}")
    
    # Match to donor profile
    if donor_profile:
        print_section("PERSONALIZED RECOMMENDATIONS")
        
        try:
            matches = await matcher.match_campaigns_to_donor(
                donor_profile=donor_profile,
                limit=len(campaigns)
            )
            
            if matches:
                safe_print(f"\n{Colors.BOLD}Campaigns Ranked by Match to YOUR Profile:{Colors.END}")
                
                for i, match in enumerate(matches, 1):
                    camp = match.get("campaign", {})
                    score = match.get("match_score", 0)
                    
                    stars = '*' * int(score * 10)
                    safe_print(f"\n  {i}. {camp.get('title', 'Unknown')[:45]}")
                    safe_print(f"     Match Score: {stars} ({score:.0%})")
                    
                    if match.get("reasons"):
                        safe_print(f"     Why: {', '.join(match['reasons'][:2])}")
                    
                    if match.get("explanation"):
                        safe_print(f"     {Colors.CYAN}{match['explanation'][:120]}...{Colors.END}")
        except Exception as e:
            safe_print(f"  {Colors.YELLOW}Matching encountered issues{Colors.END}")
    
    # Recurring opportunities
    print_section("RECURRING GIVING OPPORTUNITIES")
    
    opportunities = []
    try:
        # Only analyze real campaigns entered by user (no conceptual campaigns)
        safe_print(f"  Analyzing {len(campaigns)} real campaigns for recurring opportunities...")
        
        # Register all entered campaigns
        for campaign in campaigns:
            curator.register_campaign(campaign)
        
        opportunities = await curator.curate_opportunities()
        
        if opportunities:
            safe_print(f"\n{Colors.BOLD}✓ Found {len(opportunities)} Campaigns Good for Monthly Giving:{Colors.END}")
            
            for i, opp in enumerate(opportunities[:10], 1):
                safe_print(f"\n  {i}. {Colors.BOLD}{opp.campaign_title[:45]}{Colors.END}")
                
                # Show URL if available (real campaigns only)
                if opp.campaign_url and opp.campaign_url.strip():
                    if opp.campaign_url.startswith(("http://", "https://")) or "gofundme.com" in opp.campaign_url:
                        safe_print(f"     URL: {Colors.BLUE}{opp.campaign_url}{Colors.END}")
                
                safe_print(f"     Suitability: {opp.suitability.value} (Score: {opp.suitability_score:.2f})")
                safe_print(f"     Recommended: ${opp.recommended_amount:.0f}/month")
                
                if opp.suitability_reasons:
                    safe_print(f"     Why: {', '.join(opp.suitability_reasons[:2])}")
                
                if opp.impact_per_period:
                    safe_print(f"     Impact: {opp.impact_per_period[:80]}...")
                
                if opp.pitch:
                    safe_print(f"     {Colors.CYAN}{opp.pitch[:120]}...{Colors.END}")
        else:
            safe_print(f"  {Colors.YELLOW}⚠ No campaigns met the suitability threshold for recurring giving.{Colors.END}")
    except Exception as e:
        safe_print(f"  {Colors.RED}Recurring analysis error: {str(e)}{Colors.END}")
        import traceback
        safe_print(f"  {Colors.YELLOW}Traceback: {traceback.format_exc()[:300]}...{Colors.END}")
    
    return opportunities


async def main():
    print_header("CAMPAIGN INTELLIGENCE ANALYZER")
    
    safe_print(f"""
{Colors.BOLD}Analyze campaigns with AI-powered intelligence!{Colors.END}

Options:
  1. Enter campaign URL
  2. Enter campaign data manually
  3. Import from JSON file
  4. Import from CSV file
  5. Generate demo campaigns
  6. Quick demo with sample data
""")
    
    choice = input("Choose an option (1-6): ").strip()
    
    data_agent = CampaignDataAgent()
    campaigns = []
    
    if choice == "1":
        # URL entry
        print_section("CAMPAIGN URL ENTRY")
        safe_print("""
  Enter campaign URLs (one per line).
  Examples:
    - https://www.gofundme.com/f/help-sarah-fight-cancer
    - gofundme.com/f/support-local-school
    - help-john-medical-bills (just the slug)
  
  Type 'done' when finished.
""")
        
        urls = []
        while True:
            url = input(f"  URL #{len(urls) + 1} (or 'done'): ").strip()
            
            if url.lower() == 'done':
                break
            if not url:
                continue
            
            urls.append(url)
            safe_print(f"  {Colors.GREEN}Added{Colors.END}")
        
        if urls:
            safe_print("\n  Processing URLs with AI...")
            for url in urls:
                safe_print(f"    Processing: {url[:50]}...")
                result = await data_agent._tool_process_url(url)
                if result.get("campaign"):
                    campaign = result["campaign"]
                    campaigns.append(campaign)
                    safe_print(f"    {Colors.GREEN}✓ Generated: {campaign.get('title', 'Campaign')[:40]}{Colors.END}")
                    
                    # Show generated data and allow editing
                    safe_print(f"\n    {Colors.CYAN}Generated Campaign Data:{Colors.END}")
                    safe_print(f"      Title: {campaign.get('title', 'N/A')}")
                    safe_print(f"      Category: {campaign.get('category', 'N/A')}")
                    safe_print(f"      Goal: ${campaign.get('goal_amount', 0):,.0f}")
                    
                    edit = input("    Edit this campaign? (y/n): ").strip().lower()
                    if edit == 'y':
                        # Allow manual editing
                        new_title = input(f"      Title [{campaign.get('title', '')}]: ").strip()
                        if new_title:
                            campaign["title"] = new_title
                        
                        new_desc = input(f"      Description [{campaign.get('description', '')[:50]}...]: ").strip()
                        if new_desc:
                            campaign["description"] = new_desc
                        
                        new_cat = input(f"      Category [{campaign.get('category', '')}]: ").strip()
                        if new_cat:
                            campaign["category"] = new_cat
                        
                        new_goal = input(f"      Goal amount [{campaign.get('goal_amount', 0):,.0f}]: ").strip()
                        if new_goal:
                            try:
                                campaign["goal_amount"] = float(new_goal.replace('$', '').replace(',', ''))
                            except:
                                pass
                        
                        safe_print(f"    {Colors.GREEN}Updated campaign data{Colors.END}")
    
    elif choice == "2":
        # Manual entry
        print_section("MANUAL CAMPAIGN ENTRY")
        
        while True:
            safe_print("\n  Enter campaign details:\n")
            
            title = input("  Campaign title: ").strip()
            if not title:
                break
            
            description = input("  Description: ").strip()
            category = input("  Category (medical/education/community/emergency/animals/other): ").strip() or "other"
            
            try:
                goal = float(input("  Goal amount ($): ").replace('$', '').replace(',', '') or "10000")
            except:
                goal = 10000
            
            try:
                raised = float(input("  Amount raised ($): ").replace('$', '').replace(',', '') or "0")
            except:
                raised = 0
            
            try:
                donors = int(input("  Number of donors: ").replace(',', '') or "0")
            except:
                donors = 0
            
            location = input("  Location (city, state): ").strip()
            organizer = input("  Organizer name: ").strip()
            
            campaign = {
                "campaign_id": f"manual_{datetime.now().strftime('%H%M%S')}",
                "title": title,
                "description": description,
                "category": category,
                "goal_amount": goal,
                "raised_amount": raised,
                "donor_count": donors,
                "location": location,
                "organizer_name": organizer,
            }
            
            # Use agent to validate and enrich
            validated = await data_agent._tool_validate_data(campaign)
            if validated["is_valid"]:
                enriched = await data_agent._tool_enrich_data(validated["validated_data"])
                campaigns.append(enriched["enriched_data"])
                safe_print(f"\n  {Colors.GREEN}Added: {title}{Colors.END}")
            else:
                safe_print(f"  {Colors.YELLOW}Validation issues: {', '.join(validated['issues'])}{Colors.END}")
                campaigns.append(campaign)
            
            more = input("\n  Add another campaign? (y/n): ").strip().lower()
            if more != 'y':
                break
    
    elif choice == "3":
        # JSON import
        print_section("JSON IMPORT")
        json_path = input("  Path to JSON file: ").strip()
        
        result = await data_agent._tool_import_json(json_path)
        if result.get("imported_count", 0) > 0:
            campaigns = result["campaigns"]
            safe_print(f"  {Colors.GREEN}Imported {len(campaigns)} campaigns{Colors.END}")
        else:
            safe_print(f"  {Colors.RED}Import failed: {result.get('error', 'Unknown error')}{Colors.END}")
    
    elif choice == "4":
        # CSV import
        print_section("CSV IMPORT")
        csv_path = input("  Path to CSV file: ").strip()
        
        result = await data_agent._tool_import_csv(csv_path)
        if result.get("imported_count", 0) > 0:
            campaigns = result["campaigns"]
            safe_print(f"  {Colors.GREEN}Imported {len(campaigns)} campaigns{Colors.END}")
        else:
            safe_print(f"  {Colors.RED}Import failed: {result.get('error', 'Unknown error')}{Colors.END}")
    
    elif choice == "5":
        # Generate demo
        print_section("GENERATE DEMO CAMPAIGNS")
        try:
            count = int(input("  How many campaigns? (default 5): ").strip() or "5")
        except:
            count = 5
        
        result = await data_agent._tool_generate_demo(count=count)
        campaigns = result["campaigns"]
        safe_print(f"  {Colors.GREEN}Generated {len(campaigns)} demo campaigns{Colors.END}")
    
    elif choice == "6":
        # Quick demo
        print_section("QUICK DEMO")
        
        campaigns = [
            {
                "campaign_id": "demo_medical_1",
                "title": "Help Sarah Fight Cancer - Medical Treatment Fund",
                "description": "Sarah is a 32-year-old mother of two who was recently diagnosed with stage 3 breast cancer. She needs funds for chemotherapy, surgery, and ongoing treatment. Every dollar helps her family during this difficult time.",
                "category": "Medical",
                "goal_amount": 50000,
                "raised_amount": 23450,
                "donor_count": 312,
                "location": "Austin, TX",
                "organizer_name": "Michael Thompson",
            },
            {
                "campaign_id": "demo_education_1",
                "title": "STEM Lab for Riverside Elementary School",
                "description": "Help us build a state-of-the-art STEM laboratory for underprivileged students. This lab will provide hands-on learning experiences in science, technology, engineering, and math.",
                "category": "Education",
                "goal_amount": 25000,
                "raised_amount": 18750,
                "donor_count": 156,
                "location": "Denver, CO",
                "organizer_name": "Principal Johnson",
            },
            {
                "campaign_id": "demo_community_1",
                "title": "Rebuild Our Community Center After Fire",
                "description": "Our beloved community center was destroyed in a fire last month. This center served as a gathering place for seniors, after-school programs, and local events. Help us rebuild!",
                "category": "Community",
                "goal_amount": 100000,
                "raised_amount": 67800,
                "donor_count": 523,
                "location": "Portland, OR",
                "organizer_name": "Community Association",
            },
        ]
        
        safe_print("  Using sample campaign data for demonstration.\n")
    
    else:
        safe_print(f"{Colors.RED}Invalid choice.{Colors.END}")
        return
    
    if not campaigns:
        safe_print(f"\n{Colors.RED}No campaigns to analyze.{Colors.END}")
        return
    
    # Display campaigns
    print_section(f"CAMPAIGNS TO ANALYZE ({len(campaigns)})")
    for i, campaign in enumerate(campaigns, 1):
        display_campaign(campaign, i)
    
    # Ask about donor profile
    safe_print("\n")
    build_profile = input("Add your donation history for personalized matching? (y/n): ").strip().lower()
    
    donations = []
    if build_profile == 'y':
        print_section("YOUR DONATION HISTORY")
        safe_print("  Enter your past donations to get personalized recommendations.\n")
        
        while True:
            cause = input(f"  Donation #{len(donations) + 1} - Campaign/cause (or 'done'): ").strip()
            
            if cause.lower() == 'done':
                break
            if not cause:
                continue
            
            try:
                amount = float(input("    Amount ($): ").replace('$', '').replace(',', '') or "50")
            except:
                amount = 50
            
            category = input("    Category (medical/education/community/emergency/animals/other): ").strip() or "other"
            
            donations.append({
                "campaign_title": cause,
                "campaign_category": category,
                "amount": amount,
                "timestamp": datetime.now().isoformat(),
            })
            
            safe_print(f"    {Colors.GREEN}Added: ${amount:.0f} to '{cause}'{Colors.END}\n")
    
    # Run AI analysis
    opportunities = await analyze_with_agents(campaigns, donations if donations else None)
    
    # Save results
    print_section("SAVE RESULTS")
    save = input("\nSave analysis to file? (y/n): ").strip().lower()
    
    if save == 'y':
        output = {
            "analyzed_at": datetime.now().isoformat(),
            "campaigns": campaigns,
            "donation_history": donations,
            "recurring_opportunities": [opp.to_dict() for opp in opportunities] if opportunities else [],
            "opportunities_count": len(opportunities) if opportunities else 0,
        }
        
        filename = f"campaign_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, default=str)
        
        safe_print(f"{Colors.GREEN}Saved to {filename}{Colors.END}")
        if opportunities:
            safe_print(f"  Includes {len(opportunities)} recurring giving opportunities!")
    
    print_header("ANALYSIS COMPLETE")
    
    total_goal = sum(c.get('goal_amount', 0) for c in campaigns)
    total_raised = sum(c.get('raised_amount', 0) for c in campaigns)
    total_donors = sum(c.get('donor_count', 0) for c in campaigns)
    
    safe_print(f"""
{Colors.BOLD}Summary:{Colors.END}
  Campaigns analyzed: {len(campaigns)}
  Total goal: ${total_goal:,.0f}
  Total raised: ${total_raised:,.0f}
  Total donors: {total_donors:,}
  Recurring opportunities found: {len(opportunities) if opportunities else 0}

Thank you for using the Giving Intelligence Platform!
""")


if __name__ == "__main__":
    asyncio.run(main())

