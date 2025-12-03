import React from 'react'
import './AgentStatus.css'

function AgentStatus({ status }) {
  const agents = [
    // Campaign Analysis Agents
    { id: 'campaign_data_agent', name: 'Campaign Data Agent' },
    { id: 'campaign_intelligence_agent', name: 'Campaign Intelligence Agent' },
    { id: 'tone_checker_agent', name: 'Tone Checker Agent' },
    { id: 'ab_testing_agent', name: 'A/B Testing Agent' },
    // Donor Intelligence Agents
    { id: 'donor_affinity_profiler', name: 'Donor Affinity Profiler' },
    { id: 'campaign_matching_engine', name: 'Campaign Matching Engine' },
    { id: 'community_discovery', name: 'Community Discovery Agent' },
    { id: 'recurring_curator', name: 'Recurring Curator Agent' },
    { id: 'giving_circle_orchestrator', name: 'Giving Circle Orchestrator' },
    { id: 'engagement_agent', name: 'Engagement Agent' },
  ]

  return (
    <div className="agent-status">
      <div className="status-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2>Agent Status</h2>
            <p>Real-time status of all agents in the system</p>
          </div>
          {Object.keys(status).length > 0 && (
            <button
              onClick={() => {
                localStorage.removeItem('agent_status')
                window.location.reload()
              }}
              style={{
                background: '#f44336',
                color: 'white',
                border: 'none',
                padding: '0.5rem 1rem',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '0.85rem'
              }}
            >
              Clear
            </button>
          )}
        </div>
      </div>

      <div className="status-grid">
        {agents.map((agent) => {
          const agentStatus = status[agent.id] || { status: 'idle' }
          return (
            <div key={agent.id} className="status-card">
              <div className="status-card-header">
                <h3>{agent.name}</h3>
                <span className={`status-badge status-${agentStatus.status}`}>
                  {agentStatus.status || 'idle'}
                </span>
              </div>
              {agentStatus.current_task && (
                <div className="status-task">
                  <strong>Current Task:</strong> {agentStatus.current_task}
                </div>
              )}
              {agentStatus.progress && (
                <div className="status-progress">
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ width: `${agentStatus.progress}%` }}
                    />
                  </div>
                  <span>{agentStatus.progress}%</span>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default AgentStatus

