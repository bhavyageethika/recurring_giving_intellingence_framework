# Troubleshooting Guide

## Frontend "Failed to Fetch" Error

If you're getting "Failed to fetch" when clicking "Analyze Campaign":

### Step 1: Check if Backend is Running

Open a new terminal and run:
```bash
# Check if port 8000 is in use
# Windows
netstat -ano | findstr :8000

# Linux/Mac
lsof -i :8000
```

If nothing is running, start the backend:
```bash
uvicorn src.api.ag_ui_server:app --reload --port 8000
```

### Step 2: Test Backend Directly

Open your browser and go to:
- http://localhost:8000/health

You should see: `{"status": "healthy", "service": "AG-UI Server"}`

If this doesn't work, the backend isn't running or there's a configuration issue.

### Step 3: Check Browser Console

1. Open browser DevTools (F12)
2. Go to Console tab
3. Click "Analyze Campaign" again
4. Look for error messages

Common errors:
- `CORS policy`: Backend CORS not configured correctly
- `NetworkError`: Backend not running or wrong port
- `500 Internal Server Error`: Backend error (check backend terminal)

### Step 4: Check Backend Terminal

Look for error messages in the terminal where you ran `uvicorn`. Common issues:
- Import errors
- Missing dependencies
- API key not configured

### Step 5: Verify Frontend Proxy

The frontend uses Vite proxy. Make sure `frontend/vite.config.js` has:
```js
proxy: {
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
  },
}
```

### Step 6: Try Direct URL

If proxy isn't working, the frontend should fall back to direct URLs. Check browser Network tab to see what URL is being called.

### Quick Fix Checklist

- [ ] Backend server is running on port 8000
- [ ] http://localhost:8000/health returns success
- [ ] Frontend is running on port 3000
- [ ] Browser console shows no CORS errors
- [ ] .env file has API keys configured
- [ ] All Python dependencies installed (`pip install -r requirements.txt`)

### Still Not Working?

1. Check backend logs for detailed error messages
2. Check browser Network tab to see the actual request/response
3. Try the command-line version: `python scripts/live_campaign_intelligence.py`





