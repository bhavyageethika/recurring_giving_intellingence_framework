# Multi-Agent Collaboration Architecture

## Overview

This system is a **TRUE multi-agentic system** where all 6 agents actively collaborate with each other through the A2A (Agent-to-Agent) protocol. Agents don't work in isolation—they request help, share insights, and build on each other's work.

## Agent Collaboration Map

```
┌─────────────────────┐
│ Donor Affinity      │
│ Profiler            │
│                     │
│ Provides:           │
│ - Donor insights    │
│ - Cause affinities  │
│ - Giving patterns   │
└──────────┬──────────┘
           │
           │ A2A: get_donor_insights
           │
    ┌──────┴──────┬──────────────┬──────────────┐
    │             │              │              │
    ▼             ▼              ▼              ▼
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ Matcher │  │Community│  │Engagement│  │  Other  │
│         │  │Discovery│  │  Agent   │  │  Agents │
└────┬────┘  └────┬────┘  └────┬────┘  └─────────┘
     │            │             │
     │ A2A:       │ A2A:        │
     │ evaluate_  │ discover_   │
     │ legitimacy │ communities │
     │            │             │
     └────────────┴─────────────┘
              │
              ▼
     ┌─────────────────┐
     │ Recurring       │
     │ Curator         │
     └─────────────────┘
```

## Real Agent Collaborations

### 1. Campaign Matching Engine ↔ Donor Affinity Profiler

**When:** Matcher needs donor insights for better matching

**Flow:**
```
Matcher: "I need to match this campaign to a donor, but the profile is incomplete"
  → A2A: ask_profiler_for_insights(donor_id)
  ← Profiler: Returns cause affinities, giving patterns, motivators
  → Matcher: Uses insights to generate personalized match explanations
```

**Implementation:**
- Matcher calls `_collaborator.ask_profiler_for_insights()` when profile incomplete
- Profiler responds via A2A handler `get_donor_insights`

### 2. Recurring Curator ↔ Campaign Matching Engine

**When:** Curator needs to validate campaign legitimacy before recommending

**Flow:**
```
Curator: "Is this campaign legitimate before I recommend it for recurring giving?"
  → A2A: ask_matcher_for_legitimacy(campaign_data)
  ← Matcher: Returns legitimacy_score, is_legitimate, indicators
  → Curator: Uses legitimacy score in suitability assessment
  → Curator: Excludes campaigns with low legitimacy (< 0.3)
```

**Implementation:**
- Curator calls `_collaborator.ask_matcher_for_legitimacy()` in `_tool_assess_suitability`
- Matcher responds via A2A handler `evaluate_legitimacy`
- Legitimacy score boosts suitability score

### 3. Community Discovery ↔ Donor Affinity Profiler

**When:** Community Discovery needs donor interests to find relevant connections

**Flow:**
```
Community: "What causes does this donor care about?"
  → A2A: ask_profiler_for_insights(donor_id)
  ← Profiler: Returns top causes, giving motivators
  → Community: Uses insights to find relevant community campaigns
```

**Implementation:**
- Community calls `_collaborator.ask_profiler_for_insights()` in orchestrator
- Profiler responds via A2A handler `get_donor_insights`

### 4. Giving Circle Orchestrator ↔ Community Discovery

**When:** Giving Circle needs community connections for circle formation

**Flow:**
```
Giving Circle: "What community connections does this donor have?"
  → A2A: ask_community_for_connections(donor_id, location)
  ← Community: Returns proximities, connected campaigns
  → Giving Circle: Uses connections to suggest circle members
```

**Implementation:**
- Giving Circle calls `_collaborator.ask_community_for_connections()` in `orchestrate_circle`
- Community responds via A2A handler `discover_communities`

### 5. Engagement Agent ↔ Donor Affinity Profiler

**When:** Engagement Agent needs donor insights for personalized engagement

**Flow:**
```
Engagement: "What motivates this donor for personalized messaging?"
  → A2A: ask_profiler_for_insights(donor_id)
  ← Profiler: Returns personality, motivators, preferences
  → Engagement: Uses insights to craft personalized thank you
  → MCP: Sends personalized email using contact information
```

**Implementation:**
- Engagement calls `_collaborator.ask_profiler_for_insights()` in orchestrator
- Profiler responds via A2A handler `get_donor_insights`
- Engagement uses MCP to access contacts and send emails

## Agent Collaboration Helper

The `AgentCollaborator` class provides easy-to-use methods for agents:

```python
from src.core.agent_collaboration import get_collaborator

# In agent __init__
self._collaborator = get_collaborator(self.agent_id)

# Use in tools
insights = await self._collaborator.ask_profiler_for_insights(donor_id)
legitimacy = await self._collaborator.ask_matcher_for_legitimacy(campaign_data)
connections = await self._collaborator.ask_community_for_connections(donor_id, location)
```

## A2A Message Handlers

Each agent registers handlers for A2A requests:

**Donor Affinity Profiler:**
- `get_donor_insights` - Returns donor insights for other agents

**Campaign Matching Engine:**
- `evaluate_legitimacy` - Evaluates campaign legitimacy
- `analyze_campaign` - Analyzes campaign (with optional donor profile)
- `match_campaign` - Matches campaign to donor (may ask Profiler if needed)

**Community Discovery:**
- `discover_communities` - Discovers community connections

**Engagement Agent:**
- Uses MCP tools for contacts and email (not A2A, but external integration)

## LangGraph Workflow with Collaboration

The LangGraph orchestrator shows all collaborations:

1. **Profile Donor** (Profiler)
   - No collaboration needed (entry point)

2. **Analyze Campaigns** (Matcher)
   - **A2A:** Asks Profiler for donor insights if profile incomplete
   - Uses insights for better matching

3. **Discover Community** (Community Discovery)
   - **A2A:** Asks Profiler for donor interests
   - Uses interests to find relevant connections

4. **Curate Opportunities** (Recurring Curator)
   - **A2A:** Asks Matcher for legitimacy check (for each campaign)
   - Uses legitimacy score in suitability assessment
   - Excludes low-legitimacy campaigns

5. **Suggest Giving Circles** (Giving Circle)
   - **A2A:** Asks Community Discovery for connections
   - Uses connections for circle member suggestions

6. **Plan Engagement** (Engagement Agent)
   - **A2A:** Asks Profiler for donor insights
   - **MCP:** Accesses contacts for personalization
   - **MCP:** Sends personalized thank you email

## Benefits of Multi-Agent Collaboration

1. **Specialization**: Each agent focuses on its expertise
2. **Reusability**: Agents share capabilities via A2A
3. **Intelligence**: Decisions based on collective insights
4. **Efficiency**: Agents don't duplicate work
5. **Flexibility**: Easy to add new agents and collaborations

## Example: Real Collaboration Flow

**Scenario:** Curator assessing a campaign for recurring giving

```
1. Curator receives campaign data
2. Curator calls: ask_matcher_for_legitimacy(campaign)
   → A2A message sent to Matcher
3. Matcher evaluates legitimacy:
   - Checks verified identity
   - Reviews organizer history
   - Analyzes documentation mentions
   - Returns: {legitimacy_score: 0.85, is_legitimate: true}
4. Curator receives response
5. Curator uses legitimacy_score to boost suitability_score
6. Curator includes legitimacy info in reasons
7. Campaign recommended with: "High legitimacy (0.85) from Campaign Matching Engine"
```

This is **real collaboration**—agents working together, not in isolation!





