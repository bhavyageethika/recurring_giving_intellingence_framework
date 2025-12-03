/**
 * AG-UI Client Implementation
 * 
 * Simple WebSocket client for AG-UI protocol communication.
 * Connects to the backend WebSocket server and handles agent events.
 */

class AGUIClient {
  constructor(url = null) {
    // Use relative URL if in browser, or provided URL
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsHost = window.location.host
    this.url = url || `${wsProtocol}//${wsHost}/ag-ui/stream`
    this.ws = null
    this.reconnectAttempts = 0
    this.maxReconnectAttempts = 5
    this.reconnectDelay = 1000
    this.listeners = new Map()
    this.isConnected = false
  }

  connect() {
    try {
      console.log('AG-UI: Attempting to connect to', this.url)
      this.ws = new WebSocket(this.url)
      
      // Set a connection timeout
      const connectionTimeout = setTimeout(() => {
        if (this.ws && this.ws.readyState === WebSocket.CONNECTING) {
          console.warn('AG-UI: Connection timeout, closing socket')
          this.ws.close()
          this.attemptReconnect()
        }
      }, 10000) // 10 second connection timeout
      
      this.ws.onopen = () => {
        clearTimeout(connectionTimeout)
        console.log('AG-UI: ✅ Connected to server successfully!')
        this.isConnected = true
        this.reconnectAttempts = 0
        this.emit('connected', {})
        
        // Start keepalive ping
        this.startKeepalive()
      }
      
      this.ws.onopen = () => {
        console.log('AG-UI: ✅ Connected to server successfully!')
        this.isConnected = true
        this.reconnectAttempts = 0
        this.emit('connected', {})
      }
      
      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          console.log('AG-UI: 📨 Message received:', data.type, data.agent)
          this.handleMessage(data)
        } catch (error) {
          console.error('AG-UI: Error parsing message', error, event.data)
        }
      }
      
      this.ws.onerror = (error) => {
        console.error('AG-UI: ❌ WebSocket error', error)
        this.emit('error', { error })
      }
      
      this.ws.onclose = (event) => {
        console.log('AG-UI: Disconnected from server', event.code, event.reason)
        this.isConnected = false
        this.emit('disconnected', { code: event.code, reason: event.reason })
        // Only attempt reconnect if it wasn't a manual close (code 1000)
        if (event.code !== 1000) {
          this.attemptReconnect()
        }
      }
    } catch (error) {
      console.error('AG-UI: Connection error', error)
      this.attemptReconnect()
    }
  }

  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      const delay = this.reconnectDelay * this.reconnectAttempts
      console.log(`AG-UI: Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`)
      setTimeout(() => this.connect(), delay)
    } else {
      console.error('AG-UI: Max reconnection attempts reached')
      this.emit('reconnect_failed', {})
    }
  }

  handleMessage(data) {
    // Route message to appropriate listeners
    if (data.type) {
      this.emit(data.type, data)
    }
    
    // Also emit to generic 'message' listeners
    this.emit('message', data)
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, [])
    }
    this.listeners.get(event).push(callback)
  }

  off(event, callback) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event)
      const index = callbacks.indexOf(callback)
      if (index > -1) {
        callbacks.splice(index, 1)
      }
    }
  }

  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        try {
          callback(data)
        } catch (error) {
          console.error(`AG-UI: Error in listener for ${event}`, error)
        }
      })
    }
  }

  send(data) {
    if (this.ws && this.isConnected) {
      this.ws.send(JSON.stringify(data))
    } else {
      console.warn('AG-UI: Cannot send message, not connected')
    }
  }

  startKeepalive() {
    // Clear any existing keepalive
    if (this.keepaliveInterval) {
      clearInterval(this.keepaliveInterval)
    }
    
    // Send ping every 30 seconds to keep connection alive
    this.keepaliveInterval = setInterval(() => {
      if (this.ws && this.isConnected && this.ws.readyState === WebSocket.OPEN) {
        try {
          this.ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }))
        } catch (error) {
          console.error('AG-UI: Keepalive ping failed', error)
          clearInterval(this.keepaliveInterval)
          this.attemptReconnect()
        }
      } else {
        clearInterval(this.keepaliveInterval)
      }
    }, 30000)
  }

  disconnect() {
    if (this.keepaliveInterval) {
      clearInterval(this.keepaliveInterval)
      this.keepaliveInterval = null
    }
    if (this.ws) {
      this.ws.close(1000, 'Client disconnecting') // Normal closure
      this.ws = null
    }
    this.isConnected = false
    this.listeners.clear()
  }
}

// Export singleton instance
export const aguiClient = new AGUIClient()
export default AGUIClient

