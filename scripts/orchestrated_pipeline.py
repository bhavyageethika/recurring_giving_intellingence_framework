"""
Orchestrated Multi-Agent Pipeline using LangGraph

This script demonstrates the full multi-agent workflow orchestrated by LangGraph,
with active A2A communication between agents.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.langgraph_orchestrator import get_orchestrator
from src.agents.campaign_data_agent import CampaignDataAgent

# Set up encoding for Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


async def main():
    """Run the orchestrated multi-agent pipeline."""
    print("=" * 80)
    print("ORCHESTRATED MULTI-AGENT PIPELINE (LangGraph + A2A)")
    print("=" * 80)
    print()
    
    # Initialize orchestrator (this registers all agents with A2A protocol)
    print("Initializing LangGraph orchestrator...")
    orchestrator = get_orchestrator()
    print("✓ Orchestrator initialized")
    print("✓ All agents registered with A2A protocol")
    print()
    
    # Get campaign data
    print("Step 1: Getting campaign data")
    print("-" * 80)
    campaign_agent = CampaignDataAgent()
    
    print("\nChoose how to input campaign data:")
    print("1. Manual entry")
    print("2. Import from JSON file")
    print("3. Process URL (LLM-based inference)")
    print("4. Generate demo campaigns")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    campaigns = []
    if choice == "1":
        print("\nEnter campaign details:")
        campaign = {
            "campaign_id": input("Campaign ID: ").strip() or "campaign_1",
            "title": input("Title: ").strip() or "Sample Campaign",
            "description": input("Description: ").strip() or "A sample campaign for testing",
            "category": input("Category: ").strip() or "medical",
            "goal_amount": float(input("Goal amount: ").strip() or "10000"),
            "raised_amount": float(input("Raised amount: ").strip() or "5000"),
            "donor_count": int(input("Donor count: ").strip() or "50"),
        }
        campaigns = [campaign]
    elif choice == "2":
        file_path = input("Enter JSON file path: ").strip()
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    campaigns = data
                else:
                    campaigns = [data]
            print(f"✓ Loaded {len(campaigns)} campaign(s) from file")
        except Exception as e:
            print(f"✗ Error loading file: {e}")
            return
    elif choice == "3":
        url = input("Enter campaign URL or slug: ").strip()
        try:
            result = await campaign_agent.process_campaign_url(url)
            if result.get("success"):
                campaigns = [result["campaign_data"]]
                print(f"✓ Processed campaign from URL")
            else:
                print(f"✗ Error: {result.get('error', 'Unknown error')}")
                return
        except Exception as e:
            print(f"✗ Error processing URL: {e}")
            return
    elif choice == "4":
        try:
            result = await campaign_agent.generate_demo_campaigns(count=3)
            campaigns = result.get("campaigns", [])
            print(f"✓ Generated {len(campaigns)} demo campaign(s)")
        except Exception as e:
            print(f"✗ Error generating campaigns: {e}")
            return
    else:
        print("Invalid choice, using demo campaigns")
        result = await campaign_agent.generate_demo_campaigns(count=2)
        campaigns = result.get("campaigns", [])
    
    if not campaigns:
        print("✗ No campaigns to process")
        return
    
    print(f"\n✓ Processing {len(campaigns)} campaign(s)")
    print()
    
    # Get donor data
    print("Step 2: Getting donor data")
    print("-" * 80)
    donor_id = input("Enter donor ID (or press Enter for 'donor_1'): ").strip() or "donor_1"
    
    print("\nEnter donation history (JSON format, or press Enter for sample):")
    donations_input = input().strip()
    
    if donations_input:
        try:
            donations = json.loads(donations_input)
            if not isinstance(donations, list):
                donations = [donations]
        except:
            print("Invalid JSON, using sample donations")
            donations = [
                {
                    "campaign_id": campaigns[0].get("campaign_id", "campaign_1"),
                    "campaign_title": campaigns[0].get("title", "Sample Campaign"),
                    "campaign_category": campaigns[0].get("category", "medical"),
                    "amount": 50.0,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ]
    else:
        donations = [
            {
                "campaign_id": campaigns[0].get("campaign_id", "campaign_1"),
                "campaign_title": campaigns[0].get("title", "Sample Campaign"),
                "campaign_category": campaigns[0].get("category", "medical"),
                "amount": 50.0,
                "timestamp": datetime.utcnow().isoformat(),
            }
        ]
    
    donor_metadata = {
        "location": input("Location (optional): ").strip() or "",
        "joined_date": datetime.utcnow().isoformat(),
    }
    
    print(f"\n✓ Donor data prepared")
    print()
    
    # Run orchestrated workflow
    print("Step 3: Running orchestrated workflow")
    print("-" * 80)
    print("\nWorkflow steps:")
    print("  1. Profile Donor (Donor Affinity Profiler)")
    print("  2. Analyze Campaigns (Campaign Matching Engine)")
    print("     → A2A: Curator asks Matcher for legitimacy checks")
    print("  3. Discover Community (Community Discovery Agent)")
    print("  4. Curate Opportunities (Recurring Opportunity Curator)")
    print("  5. Suggest Giving Circles (Giving Circle Orchestrator)")
    print("  6. Plan Engagement (Engagement Agent)")
    print("     → A2A: Engagement Agent asks Profiler for donor insights")
    print()
    
    try:
        final_state = await orchestrator.run_workflow(
            donor_id=donor_id,
            donations=donations,
            campaigns=campaigns,
            donor_metadata=donor_metadata,
        )
        
        print("✓ Workflow completed")
        print()
        
        # Display results
        print("=" * 80)
        print("WORKFLOW RESULTS")
        print("=" * 80)
        print()
        
        # Donor Profile
        if final_state.get("donor_profile"):
            profile = final_state["donor_profile"]
            print("📊 DONOR PROFILE")
            print("-" * 80)
            print(f"Donor ID: {profile.get('donor_id', 'N/A')}")
            if profile.get("primary_pattern"):
                print(f"Primary Pattern: {profile['primary_pattern']}")
            if profile.get("cause_affinities"):
                print(f"Top Causes: {', '.join([c.get('category', 'N/A') for c in profile['cause_affinities'][:3]])}")
            print()
        
        # Campaign Analyses
        if final_state.get("campaign_analyses"):
            print("🔍 CAMPAIGN ANALYSES")
            print("-" * 80)
            for i, analysis in enumerate(final_state["campaign_analyses"], 1):
                print(f"\nCampaign {i}: {analysis.get('title', 'N/A')}")
                print(f"  Category: {analysis.get('taxonomy', {}).get('primary_category', 'N/A')}")
                print(f"  Urgency: {analysis.get('urgency_level', 'N/A')}")
                print(f"  Legitimacy Score: {analysis.get('legitimacy_score', 0):.2f}")
            print()
        
        # Recurring Opportunities
        if final_state.get("recurring_opportunities"):
            print("💫 RECURRING OPPORTUNITIES")
            print("-" * 80)
            for i, opp in enumerate(final_state["recurring_opportunities"], 1):
                print(f"\nOpportunity {i}: {opp.get('campaign_title', 'N/A')}")
                print(f"  Suitability: {opp.get('suitability', 'N/A')}")
                print(f"  Score: {opp.get('suitability_score', 0):.2f}")
                print(f"  Recommended: ${opp.get('recommended_amount', 0):.2f} {opp.get('recommended_frequency', 'monthly')}")
                if opp.get("suitability_reasons"):
                    print(f"  Reasons: {', '.join(opp['suitability_reasons'][:2])}")
            print()
        
        # Engagement Plan
        if final_state.get("engagement_plan"):
            plan = final_state["engagement_plan"]
            print("📧 ENGAGEMENT PLAN")
            print("-" * 80)
            if plan.get("recommended_actions"):
                for action in plan["recommended_actions"][:3]:
                    print(f"  • {action.get('action', 'N/A')}")
            print()
        
        # A2A Communication Log
        print("💬 A2A COMMUNICATION")
        print("-" * 80)
        print("Agent-to-Agent messages exchanged during workflow:")
        print("  • Curator → Matcher: evaluate_legitimacy")
        print("  • Engagement Agent → Profiler: get_donor_insights")
        print()
        
        # Errors
        if final_state.get("errors"):
            print("⚠️  ERRORS")
            print("-" * 80)
            for error in final_state["errors"]:
                print(f"  • {error.get('step', 'Unknown')}: {error.get('error', 'N/A')}")
            print()
        
        # Save results
        output_file = f"orchestrated_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_state, f, indent=2, default=str)
        print(f"✓ Results saved to {output_file}")
        
        # Visualize workflow
        print("\n" + "=" * 80)
        print("WORKFLOW VISUALIZATION")
        print("=" * 80)
        mermaid_graph = orchestrator.visualize_workflow()
        print("\nMermaid graph (copy to https://mermaid.live/ to visualize):")
        print("-" * 80)
        print(mermaid_graph)
        print()
        
    except Exception as e:
        print(f"\n✗ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    asyncio.run(main())





