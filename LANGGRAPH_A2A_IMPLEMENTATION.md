# LangGraph + A2A Implementation Summary

## What We Built

We've enhanced the multi-agent system with **LangGraph orchestration** and **active Agent-to-Agent (A2A) communication**, making it a true multi-agentic system.

## Key Components

### 1. LangGraph Orchestrator (`src/core/langgraph_orchestrator.py`)

A state machine that orchestrates all 6 agents:

**Features:**
- **State Machine**: Defines workflow nodes and edges
- **Conditional Routing**: Branches based on results (e.g., skip community discovery if no location)
- **Parallel Execution**: Independent agents run simultaneously
- **State Management**: LangGraph checkpointer maintains workflow state
- **Visualization**: Generates Mermaid diagrams

**Workflow:**
```
Profile Donor → Analyze Campaigns → Discover Community → Curate Opportunities
                                                              ↓
                                    Suggest Giving Circles ← ┘
                                    Plan Engagement ← ┘
```

### 2. Active A2A Communication

Agents now actively communicate with each other:

**Example 1: Curator → Matcher**
- Recurring Curator asks Campaign Matcher to evaluate campaign legitimacy
- Uses A2A protocol to send `evaluate_legitimacy` message
- Only includes legitimate campaigns in recurring opportunities

**Example 2: Engagement Agent → Profiler**
- Engagement Agent asks Donor Profiler for donor insights
- Uses A2A protocol to send `get_donor_insights` message
- Uses insights to personalize engagement plan

**A2A Handlers Added:**
- `CampaignMatchingEngine._handle_evaluate_legitimacy()` - Responds to legitimacy checks
- `DonorAffinityProfiler._handle_get_donor_insights()` - Responds to insight requests

### 3. Orchestrated Pipeline Script (`scripts/orchestrated_pipeline.py`)

Interactive script that runs the full workflow:

**Features:**
- Initializes LangGraph orchestrator
- Registers all agents with A2A protocol
- Interactive data input (campaigns, donations)
- Runs complete workflow with agent collaboration
- Displays results and A2A communication logs
- Generates workflow visualization

## How to Use

### Run the Orchestrated Pipeline

```bash
python scripts/orchestrated_pipeline.py
```

**Steps:**
1. Choose how to input campaign data (manual, JSON, URL, demo)
2. Enter donor data (ID, donations, metadata)
3. Watch the workflow execute with A2A communication
4. View results and communication logs
5. Get workflow visualization (Mermaid diagram)

### Example Output

```
ORCHESTRATED MULTI-AGENT PIPELINE (LangGraph + A2A)
================================================================================

Initializing LangGraph orchestrator...
✓ Orchestrator initialized
✓ All agents registered with A2A protocol

Step 1: Getting campaign data
--------------------------------------------------------------------------------
[Choose input method...]

Step 2: Getting donor data
--------------------------------------------------------------------------------
[Enter donor information...]

Step 3: Running orchestrated workflow
--------------------------------------------------------------------------------

Workflow steps:
  1. Profile Donor (Donor Affinity Profiler)
  2. Analyze Campaigns (Campaign Matching Engine)
     → A2A: Curator asks Matcher for legitimacy checks
  3. Discover Community (Community Discovery Agent)
  4. Curate Opportunities (Recurring Opportunity Curator)
  5. Suggest Giving Circles (Giving Circle Orchestrator)
  6. Plan Engagement (Engagement Agent)
     → A2A: Engagement Agent asks Profiler for donor insights

✓ Workflow completed

================================================================================
WORKFLOW RESULTS
================================================================================

📊 DONOR PROFILE
--------------------------------------------------------------------------------
Donor ID: donor_1
Primary Pattern: planned_giver
Top Causes: medical, community, education

🔍 CAMPAIGN ANALYSES
--------------------------------------------------------------------------------
Campaign 1: Help Sarah Fight Cancer
  Category: medical
  Urgency: high
  Legitimacy Score: 0.85

💫 RECURRING OPPORTUNITIES
--------------------------------------------------------------------------------
Opportunity 1: Help Sarah Fight Cancer
  Suitability: highly_suitable
  Score: 0.82
  Recommended: $50.00 monthly
  Reasons: Ongoing medical treatment, matches donor's medical cause interest

💬 A2A COMMUNICATION
--------------------------------------------------------------------------------
Agent-to-Agent messages exchanged during workflow:
  • Curator → Matcher: evaluate_legitimacy
  • Engagement Agent → Profiler: get_donor_insights

✓ Results saved to orchestrated_workflow_20241129_164205.json
```

## Architecture

```
┌─────────────────────────────────────┐
│   LangGraph Orchestrator            │
│   (State Machine + Routing)         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   A2A Protocol                      │
│   (Message Routing + Registry)      │
└──────────────┬──────────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
    ▼          ▼          ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ Profiler│ │ Matcher │ │ Curator │
│         │ │         │ │         │
│  ←──────┼─┼─────────┼─┼──────→  │
│         │ │         │ │         │
└─────────┘ └─────────┘ └─────────┘
    │          │          │
    └──────────┼──────────┘
               │
               ▼
         [Other Agents]
```

## Benefits

1. **True Multi-Agent Collaboration**: Agents work together, not in isolation
2. **Intelligent Workflows**: Conditional routing based on results
3. **Parallel Execution**: Independent agents run simultaneously
4. **Visual Debugging**: Mermaid diagrams show workflow execution
5. **Production Ready**: LangGraph provides robust state management

## Next Steps (Future Enhancements)

### Phase 2: MCP Integration
- Add Model Context Protocol (MCP) server
- Connect to external services (email, calendar, payments)
- Enable agents to use real-world tools

### Phase 3: AGUI (Agent GUI)
- Web dashboard for agent monitoring
- Real-time workflow visualization
- Interactive agent control
- A2A message logs

## Files Created/Modified

**New Files:**
- `src/core/langgraph_orchestrator.py` - LangGraph orchestrator
- `scripts/orchestrated_pipeline.py` - Orchestrated pipeline script
- `LANGGRAPH_A2A_IMPLEMENTATION.md` - This document

**Modified Files:**
- `src/agents/campaign_matching_engine.py` - Added A2A handler
- `src/agents/donor_affinity_profiler.py` - Added A2A handler
- `src/core/__init__.py` - Exported orchestrator
- `README.md` - Added LangGraph/A2A documentation

## Testing

Run the orchestrated pipeline to test:

```bash
python scripts/orchestrated_pipeline.py
```

Expected behavior:
1. All agents register with A2A protocol
2. Workflow executes through all nodes
3. A2A messages are exchanged (check logs)
4. Results are displayed
5. Workflow visualization is generated

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'langgraph'`
- **Solution**: `pip install langgraph==0.0.26`

**Issue**: Agents not communicating
- **Check**: Ensure agents are registered with A2A protocol (orchestrator does this automatically)
- **Check**: Look for A2A communication logs in output

**Issue**: Workflow stops early
- **Check**: Look for errors in workflow state
- **Check**: Ensure all required data is provided





