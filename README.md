# Community Giving Intelligence Platform

> An open-source multi-agent system for transforming transactional giving into relationship-driven donor engagement.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## TL;DR - What Does This System Do?

| INPUT | PROCESS | OUTPUT |
|-------|---------|--------|
| Donor's donation history | 6 autonomous LLM agents | Personalized donor profile |
| Campaign details | analyze, reason, and plan | Matched campaign recommendations |
| Social connections | using Claude/GPT | Community-connected opportunities |
| User preferences | | Recurring giving plans |
| | | Giving circles with voting |
| | | Personalized engagement messages |

**Run it:**
```bash
pip install -r requirements.txt
cp env.example .env  # Add your ANTHROPIC_API_KEY
python scripts/run.py 1  # Runs all 6 agents
```

---

## The Problem

One-time campaigns are successful but struggle with:
- **Donor Retention**: Donors give once and disappear
- **Community Formation**: No social fabric connecting givers
- **Recurring Giving**: Limited support for sustained giving relationships
- **Personalization**: Generic recommendations that don't understand donor motivations

## 💡 The Solution

Six **LLM-powered autonomous agents** that go beyond simple responses to perform:
- **🧠 Planning**: Breaking down complex goals into executable tasks
- **💭 Reasoning**: Chain-of-thought analysis for deep understanding
- **🔧 Tool Use**: Executing specialized functions to gather and process data
- **📋 Task Decomposition**: Managing multi-step workflows autonomously
- **🔄 Reflection**: Learning from outcomes and adjusting strategies

These agents work together to understand *why* donors give, connect them through community, and enable collective and recurring giving.

---

## 📥 System Inputs & 📤 Outputs

### What This System Takes as Input

| Input Type | Description | Example |
|------------|-------------|---------|
| **Donation History** | List of past donations with campaign details | `[{campaign: "Help Sarah", amount: 50, date: "2024-01-15", category: "medical"}]` |
| **Donor Metadata** | Basic donor information | `{name: "John", location: "Dallas, TX", employer: "Google"}` |
| **Campaign Data** | Campaign details for analysis | `{title: "Medical bills for Liam", description: "...", goal: 5000, raised: 2500}` |
| **Social Connections** | User's network (optional) | `[{connected_to: "user_002", type: "colleague"}]` |
| **User Preferences** | Giving preferences (optional) | `{preferred_frequency: "monthly", budget: 100}` |

### What This System Produces as Output

| Output Type | Description | Example |
|-------------|-------------|---------|
| **Donor Profile** | Comprehensive giving persona | See below |
| **Campaign Matches** | Ranked campaigns with explanations | `[{campaign: "...", match_score: 0.85, reason: "Matches your medical cause interest"}]` |
| **Community Campaigns** | Campaigns connected to donor's network | `[{campaign: "...", connection: "Your colleague Sarah started this"}]` |
| **Recurring Plan** | Personalized monthly giving portfolio | `{total: $75/month, opportunities: [...], impact: "Feeds 50 families"}` |
| **Giving Circle** | Collective giving group with voting | `{name: "Family Circle", members: 5, pool: $500}` |
| **Engagement Nudges** | Personalized messages | `{subject: "Your impact update", message: "Your $50 helped..."}` |

### Example: Donor Profile Output

```json
{
  "donor_id": "donor-123",
  "profile_completeness": 0.83,
  
  "financial_summary": {
    "total_lifetime_giving": 1027.04,
    "average_donation": 85.59,
    "donation_count": 12
  },
  
  "cause_affinities": [
    {"category": "community", "score": 0.55, "confidence": 1.0},
    {"category": "medical", "score": 0.25, "confidence": 0.6},
    {"category": "education", "score": 0.20, "confidence": 0.6}
  ],
  
  "giving_motivators": {
    "community_momentum": 0.58,
    "time_sensitivity": 0.17,
    "personal_connection": 0.12
  },
  
  "engagement_scores": {
    "engagement": 0.81,
    "recency": 0.93,
    "loyalty": 0.75
  },
  
  "llm_insights": {
    "personality_summary": "A community-focused giver who prioritizes local impact...",
    "giving_philosophy": "Believes in the power of small, consistent contributions...",
    "recommendations": [
      "Share personal updates from beneficiaries",
      "Highlight matching opportunities",
      "Invite to join a giving circle"
    ]
  },
  
  "predicted_interests": [
    "Local community health initiatives",
    "Youth education programs",
    "Emergency relief funds"
  ]
}
```

### Example: Campaign Match Output

```json
{
  "matches": [
    {
      "campaign_id": "camp-456",
      "title": "Help build a community park",
      "match_score": 0.87,
      "reasons": [
        "Matches your interest in community causes",
        "Located in your city (Dallas)",
        "Your colleague Mike donated"
      ],
      "explanation": "This campaign aligns perfectly with your demonstrated passion for local community projects. Three people from your network have already contributed, creating momentum you can join.",
      "urgency": "medium",
      "legitimacy_score": 0.85
    }
  ]
}
```

### Example: Recurring Giving Plan Output

```json
{
  "plan_id": "plan_donor123_20241129",
  "donor_id": "donor-123",
  "total_monthly_commitment": 75.00,
  
  "opportunities": [
    {
      "campaign": "Local Food Bank",
      "amount": 25,
      "frequency": "monthly",
      "impact": "Provides 100 meals per month"
    },
    {
      "campaign": "Children's Hospital Fund",
      "amount": 50,
      "frequency": "monthly",
      "impact": "Supports 2 hours of pediatric care"
    }
  ],
  
  "plan_summary": "Your $75/month creates sustained impact across causes you care about. Over a year, you'll provide 1,200 meals and 24 hours of pediatric care.",
  
  "diversification": {
    "medical": "67%",
    "community": "33%"
  }
}
```

---

## 🔄 Data Flow

```
Donor opens app
       │
       ▼
┌─────────────────────┐
│ Affinity Profiler   │──── "This donor cares about: pediatric health, local community, education"
└─────────────────────┘
       │
       ▼
┌─────────────────────┐      ┌─────────────────────┐
│ Campaign Matching   │◄────►│ Local Community     │
│ Engine              │      │ Discovery           │
└─────────────────────┘      └─────────────────────┘
       │                              │
       └──────────┬───────────────────┘
                  ▼
       "Here are 5 campaigns ranked by affinity + proximity"
                  │
                  ▼
┌─────────────────────┐
│ Recurring Curator   │──── "2 of these have ongoing needs—consider monthly giving"
└─────────────────────┘
                  │
                  ▼
┌─────────────────────┐
│ Giving Circle       │──── "Your family giving circle is voting on Q4 donations"
│ Orchestrator        │
└─────────────────────┘
                  │
                  ▼
┌─────────────────────┐
│ Engagement Agent    │──── "It's been 90 days—here's impact from your last gift"
└─────────────────────┘
```

## 🔗 LangGraph Orchestration & A2A Communication

The system uses **LangGraph** for workflow orchestration and **Agent-to-Agent (A2A) Protocol** for inter-agent communication, enabling true multi-agent collaboration.

### LangGraph Workflow Orchestration

The `LangGraphOrchestrator` creates a state machine that coordinates all six agents:

```
┌─────────────────┐
│  Profile Donor  │ (Donor Affinity Profiler)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analyze         │ (Campaign Matching Engine)
│ Campaigns       │
└────────┬────────┘
         │
         ├─── A2A: Curator asks Matcher for legitimacy checks
         │
         ▼
┌─────────────────┐
│ Discover        │ (Community Discovery Agent)
│ Community       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Curate          │ (Recurring Opportunity Curator)
│ Opportunities   │
└────────┬────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌─────────────────┐
│ Suggest         │  │ Plan            │ (Engagement Agent)
│ Giving Circles  │  │ Engagement      │
│                 │  │                 │ ── A2A: Asks Profiler for insights
└─────────────────┘  └─────────────────┘
```

**Features:**
- **Conditional Routing**: Workflow branches based on results (e.g., skip community discovery if no location data)
- **Parallel Execution**: Independent agents run simultaneously
- **State Management**: LangGraph checkpointer maintains workflow state
- **Visualization**: Generate Mermaid diagrams of the workflow

### Agent-to-Agent (A2A) Communication

Agents actively communicate with each other during execution:

**Example 1: Curator → Matcher**
```python
# Recurring Curator asks Campaign Matcher for legitimacy assessment
message = AgentMessage(
    sender="recurring_curator",
    recipient="campaign_matching_engine",
    action="evaluate_legitimacy",
    payload={"campaign_data": campaign}
)
response = await protocol.send_message(...)
if response.payload["is_legitimate"]:
    # Include in recurring opportunities
```

**Example 2: Engagement Agent → Profiler**
```python
# Engagement Agent asks Profiler for donor insights
message = AgentMessage(
    sender="engagement_agent",
    recipient="donor_affinity_profiler",
    action="get_donor_insights",
    payload={"donor_id": donor_id}
)
insights = await protocol.send_message(...)
# Use insights to personalize engagement plan
```

**A2A Protocol Features:**
- **Message Routing**: Automatic routing to registered agents
- **Agent Registry**: Discovery of available agents and capabilities
- **Request/Response**: Synchronous message exchange
- **Error Handling**: Graceful fallbacks if agents are unavailable

### Running the Orchestrated Pipeline

Use the orchestrated pipeline script to run the full workflow with LangGraph:

```bash
python scripts/orchestrated_pipeline.py
```

This script:
1. Initializes the LangGraph orchestrator
2. Registers all agents with the A2A protocol
3. Runs the complete workflow with agent collaboration
4. Displays results and A2A communication logs
5. Generates a workflow visualization (Mermaid diagram)

**Output includes:**
- Donor profile from Profiler
- Campaign analyses from Matcher
- Recurring opportunities from Curator (with legitimacy checks via A2A)
- Engagement plan from Engagement Agent (with donor insights via A2A)
- A2A communication log showing agent interactions

## 🔌 MCP (Model Context Protocol) Integration

The system includes **MCP (Model Context Protocol)** support, enabling agents to access external tools and services through a standardized interface.

### MCP Tools Available

**Contacts & Communication:**
- `get_contacts` - Access address book to retrieve contact information
- `send_email` - Send personalized emails via email service

**Productivity:**
- `create_calendar_event` - Schedule events and reminders
- `save_document` - Save content to documents (Google Docs, local files)

### MCP in Action: Personalized Thank You Notes

The **Engagement Agent** uses MCP to send personalized thank you emails:

1. **Access Contacts**: Uses `get_contacts` to retrieve donor contact information
2. **Personalize Message**: LLM generates personalized thank you using contact details (name, notes, tags)
3. **Send Email**: Uses `send_email` to deliver the personalized message

**Example:**
```python
# Engagement Agent automatically:
# 1. Looks up contact info via MCP
contact = await mcp.call_tool("get_contacts", {"filter": {"email": donor_email}})

# 2. Generates personalized message using contact details
message = generate_thank_you(contact_info, donation_amount, campaign)

# 3. Sends via MCP email
await mcp.call_tool("send_email", {"to": donor_email, "subject": subject, "body": message})
```

### Running the MCP Demo

```bash
python scripts/mcp_demo.py
```

This demonstrates:
- Loading contacts into MCP
- Retrieving contact information
- Sending personalized thank you emails
- MCP tool integration with agents

**Note:** For a real, end-to-end demo with actual agent execution, use `scripts/real_demo.py` instead.

### Production MCP Integrations

In production, MCP would connect to:
- **Google Contacts API** - Real contact data
- **Gmail/SendGrid** - Actual email delivery
- **Google Calendar API** - Real calendar events
- **Google Docs API** - Cloud document storage

The MCP server provides a clean abstraction layer, making it easy to swap implementations without changing agent code.

## 🤖 The Six Autonomous Agents

Each agent is built on the `AutonomousAgent` framework, enabling:
- **Goal-directed behavior**: Given a goal, agents plan and execute autonomously
- **Multi-step reasoning**: Chain-of-thought prompting for complex analysis
- **Tool orchestration**: Agents select and use tools based on task requirements
- **Memory management**: Working memory, task memory, and reasoning history
- **Self-reflection**: Agents assess their progress and adjust strategies

### 1. Donor Affinity Profiler
An autonomous agent that builds comprehensive giving identities through planning and reasoning.

**Autonomous Capabilities:**
```
Goal: "Build profile for donor X"
  ├── Plan: Decompose into analysis tasks
  ├── Execute: analyze_history → identify_affinities → detect_motivators
  ├── Reason: "This donor shows strong medical cause affinity because..."
  ├── Generate: LLM insights about giving personality
  └── Reflect: Assess confidence and suggest data gaps
```

**Tools Available:**
- `analyze_donation_history` - Extract patterns and metrics
- `identify_cause_affinities` - Determine cause preferences with LLM insights
- `detect_giving_motivators` - Understand what inspires giving
- `generate_donor_insights` - Deep LLM analysis of giving personality
- `predict_future_interests` - Forecast what campaigns might resonate

### 2. Campaign Taxonomy & Matching Engine
An autonomous agent for deep semantic understanding of campaigns.

**Autonomous Capabilities:**
```
Goal: "Analyze campaign and find matching donors"
  ├── Plan: Semantic analysis → taxonomy → urgency → legitimacy → embedding
  ├── Execute: Use LLM for deep content understanding
  ├── Reason: "This campaign is urgent because... legitimate because..."
  ├── Match: Generate personalized explanations for donor matches
  └── Reflect: Assess classification confidence
```

**Tools Available:**
- `analyze_campaign_semantics` - LLM-powered deep content analysis
- `build_taxonomy` - Rich hierarchical classification
- `assess_urgency` - Evaluate time-sensitivity with reasoning
- `evaluate_legitimacy` - Trust signal detection
- `match_to_donor` - Generate personalized match explanations

### 3. Local Community Discovery Agent
Surfaces campaigns based on social proximity—location, employer, alumni network, religious community.

**Key Features:**
- Social graph traversal (1-2 degrees of separation)
- Workplace giving opportunities
- Geographic proximity matching
- Alumni network campaigns

### 4. Recurring Opportunity Curator
Identifies campaigns with ongoing needs and matches donors to sustained giving opportunities.

**Key Features:**
- Recurring need detection (chronic illness, long-term recovery)
- Monthly micro-giving portfolios
- Subscription management
- Cumulative impact tracking

### 5. Giving Circle Orchestrator
Enables collective giving for families, workplaces, friend groups.

**Key Features:**
- Shared pool management
- Campaign nomination and voting
- Matching challenges
- Group impact reporting and milestones

### 6. Engagement & Re-activation Agent
Maintains donor relationships between giving moments.

**Key Features:**
- LLM-personalized campaign updates
- Timely notifications (Giving Tuesday, tax season, anniversaries)
- Lapsed donor re-engagement with personalized messaging
- "Campaigns like ones you've loved" recommendations
- **MCP Integration**: Sends personalized thank you emails using contact information
- **Contact Personalization**: Accesses address book to enrich messages

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                          │
│                     (REST API + WebSocket)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestrator                       │
│              (Workflow State Machine + Routing)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      A2A Protocol Layer                         │
│              (Agent-to-Agent Communication)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Autonomous Agent Framework                      │
│     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│     │   Planning  │  │  Reasoning  │  │  Tool Use   │          │
│     │   Engine    │  │   (CoT)     │  │  Executor   │          │
│     └─────────────┘  └─────────────┘  └─────────────┘          │
│     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│     │   Memory    │  │ Reflection  │  │    LLM      │          │
│     │  Manager    │  │   Module    │  │  Service    │          │
│     └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Affinity    │   │   Campaign    │   │   Community   │
│   Profiler    │   │   Matching    │   │   Discovery   │
│  (Autonomous) │   │  (Autonomous) │   │  (Autonomous) │
└───────────────┘   └───────────────┘   └───────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Recurring   │   │    Giving     │   │  Engagement   │
│   Curator     │   │    Circle     │   │    Agent      │
│  (Autonomous) │   │  (Autonomous) │   │  (Autonomous) │
└───────────────┘   └───────────────┘   └───────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Layer                                    │
│        (PostgreSQL + pgvector / ChromaDB for embeddings)        │
└─────────────────────────────────────────────────────────────────┘
```

### Autonomous Agent Framework

Each agent inherits from `AutonomousAgent` which provides:

```python
class AutonomousAgent:
    # Core capabilities
    async def reason(query, context) -> str       # Chain-of-thought reasoning
    async def plan(goal, context) -> List[Task]   # Task decomposition
    async def execute_plan() -> Dict              # Autonomous execution
    async def run(goal, context) -> Dict          # Full autonomous operation
    
    # Built-in tools
    - think()           # Record reasoning steps
    - create_plan()     # Decompose goals into tasks
    - reflect()         # Assess progress and adjust
    - complete_task()   # Mark tasks done
    
    # Memory management
    - Working memory (current context)
    - Task memory (execution state)
    - Reasoning history (chain of thought)
    - Episodic memory (past interactions)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Anthropic API Key (for Claude LLM) or OpenAI API Key
- PostgreSQL (optional, for production)
- Redis (optional, for background tasks)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/giving-intelligence.git
cd giving-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file and add your API key
cp env.example .env
# Edit .env: Set ANTHROPIC_API_KEY=your-key-here
```

### Running the Real Demo (Recommended)

Run the **REAL** multi-agent system demo that processes actual data and produces real outputs:

```bash
python scripts/real_demo.py
```

This script:
- Takes your actual input (campaign data, donor info, donations)
- Executes agents with real LLM reasoning (not mock responses)

### Live Campaign Intelligence Demo

**Option 1: Web UI (Recommended)**
1. Start the backend server:
   ```bash
   # Windows
   scripts\start_backend.bat
   
   # Linux/Mac
   bash scripts/start_backend.sh
   
   # Or manually:
   uvicorn src.api.ag_ui_server:app --reload --port 8000
   ```

2. Start the frontend:
   ```bash
   cd frontend
   npm install  # First time only
   npm run dev
   ```

3. Open http://localhost:3000 and click "Live Campaign Intelligence"

**Option 2: Command Line**
```bash
python scripts/live_campaign_intelligence.py
```

This demo shows:
- **Campaign quality score** with specific improvement suggestions
- **Predicted success probability** with reasoning
- **Similar successful campaigns** and what made them work
- **Optimal sharing strategy** generated in real-time

**Net new**: GoFundMe has no predictive analytics or coaching for organizers.

### Donor Journey Simulation

**Option 1: Web UI (Recommended)**
1. Start the backend server (see above)
2. Start the frontend (see above)
3. Open http://localhost:3000 and click "Donor Journey Simulation"

**Option 2: Command Line**
```bash
python scripts/donor_journey_simulation.py
```

This demo shows:
- **AG-UI streaming** agent thoughts as they analyze
- **Profile emerging** with cause affinities, motivators, patterns
- **Personalized campaign recommendations** with explanations
- **Suggested giving circle** based on inferred interests

**Net new**: GoFundMe doesn't understand why donors give, just what they gave to.
- Runs the complete LangGraph orchestrated workflow
- Shows real A2A agent-to-agent communication
- Uses MCP tools for real operations (contacts, email)
- Generates actual outputs (profiles, recommendations, personalized emails)
- Saves all results to a JSON file

**This is a REAL execution, not a demo with print statements!**

### Running the Orchestrated Pipeline

Run the full multi-agent workflow with LangGraph orchestration and A2A communication:

```bash
python scripts/orchestrated_pipeline.py
```

This script:
- Orchestrates all 6 agents using LangGraph state machine
- Enables A2A communication between agents
- Provides interactive campaign and donor data input
- Displays comprehensive results with A2A communication logs
- Generates workflow visualization (Mermaid diagram)

### Running Individual Agents

```bash
# Run all 6 agents in sequence
python scripts/run.py 1

# Or run individual agents:
python scripts/run.py 2    # Donor Affinity Profiler
python scripts/run.py 3    # Campaign Matching Engine
python scripts/run.py 4    # Community Discovery
python scripts/run.py 5    # Recurring Curator
python scripts/run.py 6    # Giving Circle Orchestrator
python scripts/run.py 7    # Engagement Agent

# Interactive mode (shows menu)
python scripts/run.py
```

### Analyze Real GoFundMe Campaigns

```bash
# Analyze campaigns with agentic intelligence
python scripts/campaign_analyzer.py

# Options:
# 1. Enter campaign data manually
# 2. Import from JSON file
# 3. Import from CSV file
# 4. Generate demo campaigns
# 5. Quick demo with sample data
```

This script:
- **Intelligently acquires campaign data** through multiple methods (manual entry, JSON/CSV import, semantic discovery)
- **Enriches incomplete data** using LLM reasoning
- **Discovers similar campaigns** using semantic understanding
- **Analyzes campaigns** using AI agents (taxonomy, urgency, legitimacy)
- **Matches to your profile** if you provide donation history
- **Suggests recurring opportunities** for campaigns suitable for monthly giving
- **Saves results** to JSON for further analysis

**Example:**
```bash
python scripts/campaign_analyzer.py
# Choose option 1
# Enter: https://www.gofundme.com/f/campaign-name
# Get AI-powered analysis of the real campaign!
```

### Running the API Server

```bash
# Start the FastAPI server
uvicorn src.api.main:app --reload
```

### Running with Docker

```bash
docker-compose up -d
```

### API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📚 API Examples

### Build a Donor Profile

```bash
curl -X POST "http://localhost:8000/api/v1/donors/profile" \
  -H "Content-Type: application/json" \
  -d '{
    "donor_id": "donor-123",
    "donations": [
      {
        "campaign_id": "camp-1",
        "campaign_title": "Help Sarah fight cancer",
        "campaign_category": "medical",
        "amount": 50,
        "timestamp": "2024-01-15T10:00:00Z"
      }
    ]
  }'
```

### Get Campaign Recommendations

```bash
curl "http://localhost:8000/api/v1/campaigns/match" \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"donor_id": "donor-123", "limit": 5}'
```

### Create a Giving Circle

```bash
curl -X POST "http://localhost:8000/api/v1/circles" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Smith Family Giving Circle",
    "circle_type": "family",
    "creator_id": "user-123",
    "creator_name": "John Smith"
  }'
```

### Get Complete Donor Journey

```bash
curl -X POST "http://localhost:8000/api/v1/orchestrate/donor-journey?donor_id=donor-123"
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_agents.py
```

## 📁 Project Structure

```
giving-intelligence/
├── src/
│   ├── agents/                 # The six specialized agents
│   │   ├── donor_affinity_profiler.py
│   │   ├── campaign_matching_engine.py
│   │   ├── community_discovery.py
│   │   ├── recurring_curator.py
│   │   ├── giving_circle_orchestrator.py
│   │   └── engagement_agent.py
│   ├── api/                    # FastAPI application
│   │   └── main.py
│   ├── core/                   # Core infrastructure
│   │   ├── base_agent.py       # Base agent class
│   │   └── a2a_protocol.py     # Agent communication
│   ├── data/                   # Data layer
│   │   ├── models.py           # SQLAlchemy models
│   │   └── synthetic.py        # Test data generation
│   └── config.py               # Configuration
├── tests/                      # Test suite
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 🔌 Integration

### REST API
All agent capabilities are exposed via RESTful APIs:

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/donors/profile` | Build donor profile |
| `GET /api/v1/donors/{id}/affinities` | Get cause affinities |
| `POST /api/v1/campaigns/analyze` | Analyze campaign |
| `GET /api/v1/campaigns` | Search campaigns |
| `GET /api/v1/community/{id}/campaigns` | Find proximate campaigns |
| `POST /api/v1/recurring/portfolio/suggest` | Suggest giving portfolio |
| `POST /api/v1/circles` | Create giving circle |
| `POST /api/v1/circles/nominate` | Nominate campaign |
| `POST /api/v1/circles/vote` | Vote on nomination |
| `GET /api/v1/engagement/{id}/updates` | Get campaign updates |

### Webhooks
Configure webhooks for real-time events:
- Donation received
- Campaign update posted
- Giving circle milestone reached
- Engagement touchpoint sent

### OAuth Integration
Ready for platform authentication with configurable OAuth providers.

## 🛣️ Roadmap

### Phase 1: MVP (Complete ✅)
- [x] Affinity Profiler + Campaign Matching Engine
- [x] Synthetic dataset generation
- [x] REST API + basic orchestration

### Phase 2: Differentiation
- [ ] Local Community Discovery with real social graph
- [ ] Recurring Opportunity Curator with payment integration
- [ ] Vector embeddings for semantic search

### Phase 3: Polish
- [ ] Giving Circle MVP with voting UI
- [ ] Engagement Agent with email integration
- [ ] Demo video and documentation

### Future
- [ ] ML-powered campaign legitimacy scoring
- [ ] Real-time recommendation updates
- [ ] Mobile SDK
- [ ] Platform marketplace integration

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the vision of transforming transactional giving into relationship-driven engagement
- Built with [FastAPI](https://fastapi.tiangolo.com/), [LangChain](https://langchain.com/), and [Pydantic](https://pydantic.dev/)

---

<p align="center">
  Made with ❤️ for the giving community
</p>

