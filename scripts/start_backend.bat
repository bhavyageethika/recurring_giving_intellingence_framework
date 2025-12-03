@echo off
echo Starting AG-UI Backend Server...
echo.
echo Make sure you have:
echo 1. Python dependencies installed (pip install -r requirements.txt)
echo 2. .env file configured with API keys
echo.
uvicorn src.api.ag_ui_server:app --reload --port 8000 --host 0.0.0.0
pause





