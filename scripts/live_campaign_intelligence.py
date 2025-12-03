"""
Live Campaign Intelligence Demo

Paste any GoFundMe URL → Get instant AI analysis

Shows:
- Campaign quality score with specific improvement suggestions
- Predicted success probability with reasoning
- Similar successful campaigns and what made them work
- Optimal sharing strategy generated in real-time
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.campaign_data_agent import CampaignDataAgent
from src.agents.campaign_intelligence_agent import CampaignIntelligenceAgent

import structlog

logger = structlog.get_logger()


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
    print_header("LIVE CAMPAIGN INTELLIGENCE DEMO")
    print("Paste any GoFundMe URL → Get instant AI analysis")
    print("\nThis demo shows:")
    print("  • Campaign quality score with improvement suggestions")
    print("  • Predicted success probability with reasoning")
    print("  • Similar successful campaigns and what made them work")
    print("  • Optimal sharing strategy generated in real-time")
    
    # Initialize agents
    print("\nInitializing agents...")
    data_agent = CampaignDataAgent()
    intelligence_agent = CampaignIntelligenceAgent()
    print("✓ Agents initialized\n")
    
    # Get campaign URL or data
    print_section("STEP 1: CAMPAIGN INPUT")
    
    url_input = input("Enter GoFundMe URL (or press Enter to use demo campaign): ").strip()
    
    campaign_data = None
    
    if url_input:
        print(f"\nProcessing URL: {url_input}")
        # Use Campaign Data Agent to process URL
        result = await data_agent._tool_process_campaign_url(url_input)
        campaign_data = result.get("campaign_data", {})
        
        if not campaign_data:
            print("⚠ Could not extract campaign data from URL. Using demo campaign instead.")
            campaign_data = {
                "campaign_id": "demo_campaign",
                "title": "Help Sarah Fight Cancer - Medical Treatment Fund",
                "description": "Sarah is a 32-year-old mother of two who was recently diagnosed with stage 3 breast cancer. She needs funds for chemotherapy, surgery, and ongoing treatment. Her family is struggling to cover medical expenses.",
                "category": "medical",
                "goal_amount": 50000.0,
                "raised_amount": 15000.0,
                "donor_count": 45,
                "updates": ["Update 1: Started treatment", "Update 2: First round complete"],
            }
    else:
        # Use demo campaign
        print("Using demo campaign...")
        campaign_data = {
            "campaign_id": "demo_campaign",
            "title": "Help Sarah Fight Cancer - Medical Treatment Fund",
            "description": "Sarah is a 32-year-old mother of two who was recently diagnosed with stage 3 breast cancer. She needs funds for chemotherapy, surgery, and ongoing treatment. Her family is struggling to cover medical expenses.",
            "category": "medical",
            "goal_amount": 50000.0,
            "raised_amount": 15000.0,
            "donor_count": 45,
            "updates": ["Update 1: Started treatment", "Update 2: First round complete"],
        }
    
    print(f"\n✓ Campaign loaded: {campaign_data['title']}")
    
    # Generate intelligence
    print_section("STEP 2: GENERATING CAMPAIGN INTELLIGENCE")
    print("Analyzing campaign with AI...")
    print("  • Assessing quality across 8 dimensions")
    print("  • Predicting success probability")
    print("  • Finding similar successful campaigns")
    print("  • Generating optimal sharing strategy")
    print("\n⏳ Processing...\n")
    
    intelligence = await intelligence_agent.analyze_campaign(campaign_data)
    
    # Display results
    print_section("CAMPAIGN QUALITY SCORE")
    print(f"Overall Quality Score: {intelligence.quality_score.overall_score:.1%}")
    print("\nDimension Scores:")
    for dim, score in intelligence.quality_score.dimension_scores.items():
        bar = "█" * int(score * 20)
        print(f"  {dim.replace('_', ' ').title():20s} {score:.1%} {bar}")
    
    print("\n✓ Strengths:")
    for strength in intelligence.quality_score.strengths:
        print(f"  • {strength}")
    
    print("\n⚠ Weaknesses:")
    for weakness in intelligence.quality_score.weaknesses:
        print(f"  • {weakness}")
    
    print("\n💡 Priority Improvements (Top 3):")
    for i, improvement in enumerate(intelligence.quality_score.priority_improvements, 1):
        print(f"  {i}. {improvement}")
    
    print_section("SUCCESS PREDICTION")
    print(f"Success Probability: {intelligence.success_prediction.success_probability:.1%}")
    print(f"Confidence Level: {intelligence.success_prediction.confidence_level.upper()}")
    print(f"\nPredicted Outcomes:")
    print(f"  • Final Amount: ${intelligence.success_prediction.predicted_amount:,.0f}")
    print(f"  • Total Donors: {intelligence.success_prediction.predicted_donors}")
    
    print("\n✅ Key Success Factors:")
    for factor in intelligence.success_prediction.key_factors:
        print(f"  • {factor}")
    
    print("\n⚠ Risk Factors:")
    for risk in intelligence.success_prediction.risk_factors:
        print(f"  • {risk}")
    
    print(f"\n📊 Reasoning:")
    print(f"  {intelligence.success_prediction.reasoning}")
    
    print_section("SIMILAR SUCCESSFUL CAMPAIGNS")
    print(f"Found {len(intelligence.similar_campaigns)} similar successful campaigns:\n")
    
    for i, similar in enumerate(intelligence.similar_campaigns, 1):
        print(f"{i}. {similar.title}")
        print(f"   Similarity: {similar.similarity_score:.1%}")
        print(f"   Success: ${similar.success_metrics.get('raised', 0):,.0f} raised "
              f"({similar.success_metrics.get('donors', 0)} donors)")
        print(f"   What Made It Work:")
        for factor in similar.what_made_it_work:
            print(f"     • {factor}")
        print(f"   Applicable Lessons:")
        for lesson in similar.lessons_applicable:
            print(f"     • {lesson}")
        print()
    
    print_section("OPTIMAL SHARING STRATEGY")
    print("Primary Channels:")
    for channel in intelligence.sharing_strategy.primary_channels:
        reach = intelligence.sharing_strategy.expected_reach.get(channel, 0)
        print(f"  • {channel} (Expected reach: {reach:,})")
    
    print("\n⏰ Timing Recommendations:")
    for channel, timing in intelligence.sharing_strategy.timing_recommendations.items():
        print(f"  • {channel}: {timing}")
    
    print("\n📝 Messaging Variants:")
    for channel, message in intelligence.sharing_strategy.messaging_variants.items():
        print(f"\n  {channel}:")
        print(f"    {message[:100]}..." if len(message) > 100 else f"    {message}")
    
    print("\n🎯 Target Audiences:")
    for audience in intelligence.sharing_strategy.target_audiences:
        print(f"  • {audience}")
    
    print("\n💬 Engagement Tactics:")
    for tactic in intelligence.sharing_strategy.engagement_tactics:
        print(f"  • {tactic}")
    
    # Save results
    print_section("SAVING RESULTS")
    output_file = f"campaign_intelligence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    output_data = {
        "campaign": campaign_data,
        "intelligence": {
            "quality_score": {
                "overall_score": intelligence.quality_score.overall_score,
                "dimension_scores": intelligence.quality_score.dimension_scores,
                "strengths": intelligence.quality_score.strengths,
                "weaknesses": intelligence.quality_score.weaknesses,
                "improvement_suggestions": intelligence.quality_score.improvement_suggestions,
                "priority_improvements": intelligence.quality_score.priority_improvements,
            },
            "success_prediction": {
                "success_probability": intelligence.success_prediction.success_probability,
                "predicted_amount": intelligence.success_prediction.predicted_amount,
                "predicted_donors": intelligence.success_prediction.predicted_donors,
                "confidence_level": intelligence.success_prediction.confidence_level,
                "key_factors": intelligence.success_prediction.key_factors,
                "risk_factors": intelligence.success_prediction.risk_factors,
                "reasoning": intelligence.success_prediction.reasoning,
            },
            "similar_campaigns": [
                {
                    "campaign_id": sc.campaign_id,
                    "title": sc.title,
                    "success_metrics": sc.success_metrics,
                    "what_made_it_work": sc.what_made_it_work,
                    "similarity_score": sc.similarity_score,
                    "lessons_applicable": sc.lessons_applicable,
                }
                for sc in intelligence.similar_campaigns
            ],
            "sharing_strategy": {
                "primary_channels": intelligence.sharing_strategy.primary_channels,
                "timing_recommendations": intelligence.sharing_strategy.timing_recommendations,
                "messaging_variants": intelligence.sharing_strategy.messaging_variants,
                "target_audiences": intelligence.sharing_strategy.target_audiences,
                "engagement_tactics": intelligence.sharing_strategy.engagement_tactics,
                "expected_reach": intelligence.sharing_strategy.expected_reach,
            },
            "timestamp": intelligence.timestamp,
        },
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Results saved to: {output_file}")
    
    print_header("DEMO COMPLETE")
    print("This demonstrates:")
    print("  ✓ Real-time campaign quality assessment")
    print("  ✓ AI-powered success prediction")
    print("  ✓ Learning from similar successful campaigns")
    print("  ✓ Actionable sharing strategy generation")
    print("\nNet new: GoFundMe has no predictive analytics or coaching for organizers.")


if __name__ == "__main__":
    asyncio.run(main())

