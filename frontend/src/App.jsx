import React, { useState, useEffect } from 'react'
import './App.css'
import AgentThoughts from './components/AgentThoughts'
import CampaignIntelligence from './components/CampaignIntelligence'
import DonorJourney from './components/DonorJourney'
import AgentStatus from './components/AgentStatus'
import AgentChat from './components/AgentChat'
import { aguiClient } from './utils/aguiClient'

const THOUGHTS_STORAGE_KEY = 'agent_thoughts'
const STATUS_STORAGE_KEY = 'agent_status'
const MAX_STORED_THOUGHTS = 200 // Limit stored thoughts to prevent localStorage bloat

function App() {
  const [activeTab, setActiveTab] = useState(() => {
    const saved = localStorage.getItem('active_tab')
    return saved || 'intelligence'
  })
  const [agentThoughts, setAgentThoughts] = useState(() => {
    const saved = localStorage.getItem(THOUGHTS_STORAGE_KEY)
    if (saved) {
      try {
        return JSON.parse(saved)
      } catch (e) {
        console.error('Error loading saved thoughts:', e)
      }
    }
    return []
  })
  const [agentStatus, setAgentStatus] = useState(() => {
    const saved = localStorage.getItem(STATUS_STORAGE_KEY)
    if (saved) {
      try {
        return JSON.parse(saved)
      } catch (e) {
        console.error('Error loading saved status:', e)
      }
    }
    return {}
  })
  const [connectionStatus, setConnectionStatus] = useState('disconnected')

  // Save active tab to localStorage
  useEffect(() => {
    localStorage.setItem('active_tab', activeTab)
  }, [activeTab])

  // Save agent thoughts to localStorage (limit to last N thoughts)
  useEffect(() => {
    if (agentThoughts.length > 0) {
      const toStore = agentThoughts.slice(-MAX_STORED_THOUGHTS)
      localStorage.setItem(THOUGHTS_STORAGE_KEY, JSON.stringify(toStore))
    }
  }, [agentThoughts])

  // Save agent status to localStorage
  useEffect(() => {
    if (Object.keys(agentStatus).length > 0) {
      localStorage.setItem(STATUS_STORAGE_KEY, JSON.stringify(agentStatus))
    }
  }, [agentStatus])

  useEffect(() => {
    console.log('App: Setting up AG-UI connection...')
    
    // Connect to AG-UI WebSocket for real-time agent thoughts
    aguiClient.connect()
    
    // Set up event listeners
    const handleThought = (data) => {
      console.log('App: Agent thought received:', data)
      setAgentThoughts(prev => {
        const updated = [...prev, data]
        // Keep only last MAX_STORED_THOUGHTS in memory
        const trimmed = updated.slice(-MAX_STORED_THOUGHTS)
        console.log('App: Total thoughts:', trimmed.length)
        return trimmed
      })
    }
    
    const handleStatus = (data) => {
      console.log('App: Agent status received:', data)
      setAgentStatus(prev => ({ ...prev, [data.agent_id]: data }))
    }
    
    const handleConnected = () => {
      console.log('App: ✅ AG-UI connected!')
      setConnectionStatus('connected')
    }
    
    const handleDisconnected = () => {
      console.log('App: ❌ AG-UI disconnected!')
      setConnectionStatus('disconnected')
    }
    
    const handleError = (data) => {
      console.error('App: AG-UI error:', data)
    }
    
    aguiClient.on('agent_thought', handleThought)
    aguiClient.on('agent_status', handleStatus)
    aguiClient.on('connected', handleConnected)
    aguiClient.on('disconnected', handleDisconnected)
    aguiClient.on('error', handleError)
    
    return () => {
      console.log('App: Cleaning up AG-UI connection...')
      aguiClient.off('agent_thought', handleThought)
      aguiClient.off('agent_status', handleStatus)
      aguiClient.off('connected', handleConnected)
      aguiClient.off('disconnected', handleDisconnected)
      aguiClient.off('error', handleError)
      aguiClient.disconnect()
    }
  }, [])

  return (
    <div className="app">
      <header className="app-header">
        <h1>Community Giving Intelligence Platform</h1>
        <p>Multi-Agent System for Donor Engagement</p>
        <div className="connection-status">
          <span className={`status-indicator ${connectionStatus}`}></span>
          <span>AG-UI: {connectionStatus}</span>
        </div>
      </header>

      <nav className="app-nav">
        <button
          className={activeTab === 'intelligence' ? 'active' : ''}
          onClick={() => setActiveTab('intelligence')}
        >
          Live Campaign Intelligence
        </button>
        <button
          className={activeTab === 'journey' ? 'active' : ''}
          onClick={() => setActiveTab('journey')}
        >
          Donor Journey Simulation
        </button>
      </nav>

      <main className="app-main">
        <div className="main-content">
          <div className="left-panel">
            <AgentChat client={aguiClient} />
            {activeTab === 'intelligence' && (
              <div className="content-section">
                <CampaignIntelligence />
              </div>
            )}
            {activeTab === 'journey' && (
              <div className="content-section">
                <DonorJourney />
              </div>
            )}
          </div>
          
          <div className="right-panel">
            <div className="agent-panel-section">
              <AgentThoughts thoughts={agentThoughts} />
            </div>
            <div className="agent-panel-section">
              <AgentStatus status={agentStatus} />
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App

