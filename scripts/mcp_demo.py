"""
MCP Integration Demo

Demonstrates Model Context Protocol (MCP) integration:
- Accessing contacts for personalization
- Sending personalized thank you emails
- Using MCP tools from agents
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.mcp_server import get_mcp_server
from src.agents.engagement_agent import EngagementAgent

# Set up encoding for Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


async def main():
    """Demo MCP integration with Engagement Agent."""
    print("=" * 80)
    print("MCP INTEGRATION DEMO")
    print("=" * 80)
    print()
    
    # Initialize MCP server
    print("Step 1: Initializing MCP Server")
    print("-" * 80)
    mcp = get_mcp_server()
    
    # Load sample contacts
    sample_contacts = [
        {
            "name": "Sarah Johnson",
            "email": "sarah.johnson@example.com",
            "phone": "+1-555-0101",
            "tags": ["friend", "medical_cause"],
            "notes": "Close friend, passionate about medical causes, prefers personal messages",
        },
        {
            "name": "Michael Chen",
            "email": "michael.chen@example.com",
            "phone": "+1-555-0102",
            "tags": ["colleague", "education"],
            "notes": "Work colleague, interested in education causes",
        },
        {
            "name": "Emily Rodriguez",
            "email": "emily.rodriguez@example.com",
            "phone": "+1-555-0103",
            "tags": ["family", "community"],
            "notes": "Family member, very active in community causes",
        },
    ]
    
    mcp.load_contacts(sample_contacts)
    print(f"✓ Loaded {len(sample_contacts)} contacts into MCP")
    print()
    
    # List available MCP tools
    print("Step 2: Available MCP Tools")
    print("-" * 80)
    tools = mcp.list_tools()
    for tool in tools:
        print(f"  • {tool['name']}: {tool['description'][:60]}...")
    print()
    
    # Initialize Engagement Agent
    print("Step 3: Initializing Engagement Agent with MCP")
    print("-" * 80)
    engagement_agent = EngagementAgent()
    
    # Register a sample donor
    engagement_agent.register_donor("donor_1", {
        "display_name": "Sarah Johnson",
        "cause_affinities": ["medical", "community"],
        "primary_pattern": "planned_giver",
    })
    print("✓ Engagement Agent initialized")
    print("✓ Donor registered")
    print()
    
    # Demo 1: Get contact info via MCP
    print("Step 4: Demo 1 - Get Contact Info via MCP")
    print("-" * 80)
    contact_result = await engagement_agent._tool_get_donor_contact_info(
        donor_email="sarah.johnson@example.com",
        donor_name="Sarah Johnson"
    )
    
    if contact_result.get("found"):
        contact = contact_result["contact"]
        print(f"✓ Found contact: {contact['name']}")
        print(f"  Email: {contact['email']}")
        print(f"  Phone: {contact.get('phone', 'N/A')}")
        print(f"  Tags: {', '.join(contact.get('tags', []))}")
        print(f"  Notes: {contact.get('notes', 'N/A')}")
    else:
        print("✗ Contact not found (using fallback)")
    print()
    
    # Demo 2: Send personalized thank you via MCP
    print("Step 5: Demo 2 - Send Personalized Thank You via MCP")
    print("-" * 80)
    print("Sending personalized thank you email...")
    print()
    
    result = await engagement_agent._tool_send_personalized_thank_you(
        donor_id="donor_1",
        donor_email="sarah.johnson@example.com",
        donation_amount=100.00,
        campaign_title="Help Sarah Fight Cancer",
        use_contact_info=True,
    )
    
    if result.get("success"):
        print("✓ Thank you email sent successfully!")
        print(f"  Message ID: {result.get('message_id', 'N/A')}")
        print(f"  Personalized with contact info: {result.get('personalized_with_contact', False)}")
        print(f"  Contact name used: {result.get('contact_name', 'N/A')}")
        print(f"  Subject: {result.get('subject', 'N/A')}")
    else:
        print(f"✗ Failed to send email: {result.get('error', 'Unknown error')}")
    print()
    
    # Demo 3: Show MCP email log
    print("Step 6: MCP Email Log")
    print("-" * 80)
    if hasattr(mcp, "_sent_emails") and mcp._sent_emails:
        for email in mcp._sent_emails[-1:]:  # Show last email
            print(f"To: {email['to']}")
            print(f"Subject: {email['subject']}")
            print(f"Body Preview: {email['body'][:200]}...")
            print(f"Sent At: {email['sent_at']}")
    print()
    
    # Summary
    print("=" * 80)
    print("MCP INTEGRATION SUMMARY")
    print("=" * 80)
    print()
    print("✓ MCP Server initialized with tools:")
    print("  • get_contacts - Access address book")
    print("  • send_email - Send emails")
    print("  • create_calendar_event - Schedule events")
    print("  • save_document - Save documents")
    print()
    print("✓ Engagement Agent enhanced with MCP tools:")
    print("  • get_donor_contact_info - Look up contacts via MCP")
    print("  • send_personalized_thank_you - Send emails via MCP")
    print()
    print("✓ Key Benefits:")
    print("  • Agents can access real external services")
    print("  • Personalized messages using contact information")
    print("  • Standardized tool interface (MCP)")
    print("  • Easy to add new integrations")
    print()
    print("In production, MCP would connect to:")
    print("  • Google Contacts API")
    print("  • Gmail/SendGrid for email")
    print("  • Google Calendar API")
    print("  • Google Docs API")
    print()


if __name__ == "__main__":
    asyncio.run(main())





