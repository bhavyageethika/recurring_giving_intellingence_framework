# AG-UI Frontend for Community Giving Intelligence Platform

React frontend with AG-UI integration for real-time agent communication.

## Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend will run on http://localhost:3000

## Backend Setup

Start the AG-UI server:

```bash
python src/api/ag_ui_server.py
```

Or use uvicorn:

```bash
uvicorn src.api.ag_ui_server:app --reload --port 8000
```

## Features

- **Live Campaign Intelligence**: Analyze GoFundMe campaigns with AI
- **Donor Journey Simulation**: Build giving identity from donation history
- **Agent Thoughts**: Real-time streaming of agent reasoning
- **Agent Status**: Monitor all agents in the system

## AG-UI Integration

The frontend connects to the AG-UI WebSocket server at `ws://localhost:8000/ag-ui/stream` to receive:
- Agent thoughts and reasoning steps
- Agent status updates
- Real-time analysis progress





