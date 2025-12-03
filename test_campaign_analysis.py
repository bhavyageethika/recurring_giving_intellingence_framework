"""
Test script for campaign analysis endpoint.
Run this to test the campaign intelligence analysis from command line.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.api.ag_ui_server import _perform_campaign_analysis, CampaignIntelligenceRequest


async def test_campaign_analysis(url: str = ""):
    """Test the campaign analysis endpoint."""
    print(f"\n{'='*60}")
    print("Testing Campaign Analysis")
    print(f"{'='*60}\n")
    
    if url:
        print(f"Analyzing campaign URL: {url}\n")
    else:
        print("Using demo campaign (no URL provided)\n")
    
    try:
        request = CampaignIntelligenceRequest(url=url)
        result = await _perform_campaign_analysis(request)
        
        print(f"\n{'='*60}")
        print("Analysis Complete!")
        print(f"{'='*60}\n")
        
        # Print summary
        if "extracted_campaign_summary" in result:
            summary = result["extracted_campaign_summary"]
            print(f"Campaign: {summary.get('title', 'Unknown')}")
            print(f"Category: {summary.get('category', 'Unknown')}")
            print(f"Goal: ${summary.get('goal_amount', 0):,.0f}")
            print(f"Raised: ${summary.get('raised_amount', 0):,.0f}")
            print(f"Donors: {summary.get('donor_count', 0)}")
            print()
        
        if "tone_analysis" in result:
            tone = result["tone_analysis"]
            print(f"Tone Score: {tone.get('overall_tone_score', 0):.1%}")
            print()
        
        if "quality_score" in result:
            quality = result["quality_score"]
            print(f"Quality Score: {quality.get('overall_score', 0):.1%}")
            print()
        
        if "success_prediction" in result:
            prediction = result["success_prediction"]
            print(f"Success Probability: {prediction.get('success_probability', 0):.1%}")
            print(f"Predicted Amount: ${prediction.get('predicted_amount', 0):,.0f}")
            print(f"Predicted Donors: {prediction.get('predicted_donors', 0)}")
            print()
        
        if "ab_testing" in result:
            ab_test = result["ab_testing"]
            variants_count = len(ab_test.get("variants", []))
            print(f"A/B Testing: {variants_count} variants generated")
            if variants_count == 0:
                print(f"  Strategy: {ab_test.get('testing_strategy', 'N/A')}")
            print()
        
        if "error" in result:
            print(f"⚠️  Warning: {result['error']}\n")
        
        # Save full results to file
        output_file = "campaign_analysis_result.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Full results saved to: {output_file}")
        
        return result
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Get URL from command line argument or use empty string for demo
    url = sys.argv[1] if len(sys.argv) > 1 else ""
    
    result = asyncio.run(test_campaign_analysis(url))
    
    if result:
        print("\n[SUCCESS] Test completed successfully!")
        sys.exit(0)
    else:
        print("\n[FAILED] Test failed!")
        sys.exit(1)

