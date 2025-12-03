import React, { useState, useEffect, useRef } from 'react'
import './AgentChat.css'

function AgentChat({ client }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef(null)

  // Load messages from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('agent_chat_messages')
    if (saved) {
      try {
        setMessages(JSON.parse(saved))
      } catch (e) {
        console.error('Error loading chat messages:', e)
      }
    }
  }, [])

  // Save messages to localStorage
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem('agent_chat_messages', JSON.stringify(messages))
    }
  }, [messages])

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Listen for agent responses via WebSocket
  useEffect(() => {
    if (!client) return

    const handleMessage = (event) => {
      if (event.type === 'agent_response') {
        setMessages(prev => [...prev, {
          id: Date.now(),
          type: 'agent',
          agent: event.agent || 'Agent',
          text: event.response || event.message,
          timestamp: new Date().toISOString(),
        }])
        setLoading(false)
      }
    }

    client.on('message', handleMessage)
    return () => client.off('message', handleMessage)
  }, [client])

  const handleSend = async (e) => {
    e.preventDefault()
    if (!input.trim() || loading) {
      console.log('Cannot send: input empty or loading', { input: input.trim(), loading })
      return
    }

    const userMessage = {
      id: Date.now(),
      type: 'user',
      text: input.trim(),
      timestamp: new Date().toISOString(),
    }

    console.log('Adding user message to chat:', userMessage)
    setMessages(prev => [...prev, userMessage])
    const messageText = input.trim()
    setInput('')
    setLoading(true)

    try {
      console.log('Sending chat message to /api/chat:', messageText)
      
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 60000) // 60 second timeout (LLM calls can take time)
      
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: messageText,
        }),
        signal: controller.signal,
      })

      clearTimeout(timeoutId)
      console.log('Chat response received:', response.status, response.statusText)

      // Read response body once
      const responseText = await response.text()
      
      if (!response.ok) {
        let errorMessage = `Server error: ${response.status} ${response.statusText}`
        try {
          const errorData = JSON.parse(responseText)
          errorMessage = errorData.detail || errorData.error || errorMessage
          console.error('Server error data:', errorData)
        } catch {
          // Not JSON, use text as error message
          errorMessage = responseText || errorMessage
          console.error('Server error text:', responseText)
        }
        throw new Error(errorMessage)
      }

      // Parse JSON response
      let data
      try {
        data = JSON.parse(responseText)
      } catch (e) {
        throw new Error('Invalid JSON response from server')
      }
      console.log('Chat response data:', data)
      
      if (!data || !data.response) {
        console.error('Invalid response data:', data)
        throw new Error('No response from server. Response: ' + JSON.stringify(data))
      }
      
      console.log('Adding agent response to chat:', data.response.substring(0, 50))
      setMessages(prev => [...prev, {
        id: Date.now(),
        type: 'agent',
        agent: data.agent_name || 'Assistant',
        text: data.response,
        timestamp: new Date().toISOString(),
      }])
    } catch (error) {
      console.error('Error sending message:', error)
      if (error.name === 'AbortError') {
        setMessages(prev => [...prev, {
          id: Date.now(),
          type: 'error',
          text: 'Request timed out. The server is taking too long to respond. Please try again.',
          timestamp: new Date().toISOString(),
        }])
      } else if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        setMessages(prev => [...prev, {
          id: Date.now(),
          type: 'error',
          text: 'Cannot connect to server. Please make sure the backend is running on port 8000.',
          timestamp: new Date().toISOString(),
        }])
      } else {
        setMessages(prev => [...prev, {
          id: Date.now(),
          type: 'error',
          text: `Error: ${error.message || 'Failed to send message. Please check if the backend is running.'}`,
          timestamp: new Date().toISOString(),
        }])
      }
    } finally {
      setLoading(false)
      console.log('Chat request completed, loading set to false')
    }
  }

  const handleClear = () => {
    setMessages([])
    localStorage.removeItem('agent_chat_messages')
  }

  return (
    <div className="agent-chat">
      <div className="chat-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2>💬 Chat</h2>
            <p>Ask questions about campaigns, donors, or giving strategies</p>
          </div>
          {messages.length > 0 && (
            <button
              onClick={handleClear}
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
              Clear Chat
            </button>
          )}
        </div>
      </div>

      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="chat-empty">
            <p>Start a conversation!</p>
            <p style={{ fontSize: '0.9rem', color: '#666', marginTop: '0.5rem' }}>
              Try asking: "What campaigns match my profile?" or "How do I analyze a campaign?"
            </p>
          </div>
        ) : (
          messages.map((msg) => (
            <div key={msg.id} className={`chat-message chat-message-${msg.type}`}>
              <div className="message-header">
                <span className="message-agent">
                  {msg.type === 'user' ? '👤 You' : msg.type === 'error' ? '❌ Error' : '🤖 Assistant'}
                </span>
                <span className="message-time">
                  {new Date(msg.timestamp).toLocaleTimeString()}
                </span>
              </div>
              <div className="message-text">{msg.text}</div>
            </div>
          ))
        )}
        {loading && (
          <div className="chat-message chat-message-agent">
            <div className="message-header">
              <span className="message-agent">🤖 Assistant</span>
            </div>
            <div className="message-text">
              <span className="typing-indicator">Thinking...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form className="chat-input-form" onSubmit={handleSend}>
        <input
          type="text"
          value={input}
          onChange={(e) => {
            console.log('Input changed:', e.target.value)
            setInput(e.target.value)
          }}
          onFocus={(e) => console.log('Input focused')}
          placeholder="Ask a question about campaigns, donors, or giving strategies..."
          disabled={loading}
          className="chat-input"
          autoFocus
          autoComplete="off"
        />
        <button
          type="submit"
          disabled={!input.trim() || loading}
          className="chat-send-button"
        >
          {loading ? '...' : 'Send'}
        </button>
      </form>
    </div>
  )
}

export default AgentChat

