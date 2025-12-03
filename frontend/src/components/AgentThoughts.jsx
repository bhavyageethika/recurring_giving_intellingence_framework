import React, { useEffect, useRef } from 'react'
import './AgentThoughts.css'

function AgentThoughts({ thoughts }) {
  const thoughtsEndRef = useRef(null)

  const scrollToBottom = () => {
    thoughtsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [thoughts])

  return (
    <div className="agent-thoughts">
      <div className="thoughts-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2>Agent Thoughts (Real-Time)</h2>
            <p>Watch agents reason and plan in real-time</p>
          </div>
          {thoughts.length > 0 && (
            <button
              onClick={() => {
                localStorage.removeItem('agent_thoughts')
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

      <div className="thoughts-container">
        {thoughts.length === 0 ? (
          <div className="no-thoughts">
            <p>No agent thoughts yet. Start an analysis to see agents think in real-time.</p>
            <div style={{ marginTop: '2rem', padding: '1rem', background: '#fff3cd', borderRadius: '8px' }}>
              <strong>💡 Debug Info:</strong>
              <ul style={{ marginTop: '0.5rem', paddingLeft: '1.5rem' }}>
                <li>Make sure backend is running: <code>uvicorn src.api.ag_ui_server:app --reload --port 8000</code></li>
                <li>Check connection status in header (should show "connected")</li>
                <li>Open browser console (F12) to see WebSocket connection logs</li>
                <li>Try clicking "Analyze Campaign" or "Build Giving Identity" to generate thoughts</li>
              </ul>
            </div>
          </div>
        ) : (
          thoughts.map((thought, index) => (
            <div key={index} className="thought-item">
              <div className="thought-header">
                <span className="thought-agent">{thought.agent}</span>
                <span className="thought-time">
                  {new Date(thought.timestamp).toLocaleTimeString()}
                </span>
              </div>
              {thought.step && (
                <span className="thought-step">{thought.step}</span>
              )}
              <div className="thought-content">{thought.thought}</div>
              {thought.data && (
                <div className="thought-data">
                  {Object.entries(thought.data).map(([key, value]) => (
                    <div key={key} className="data-item">
                      <strong>{key}:</strong> {String(value)}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))
        )}
        <div ref={thoughtsEndRef} />
      </div>
    </div>
  )
}

export default AgentThoughts

