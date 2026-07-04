@echo off
echo.
echo  ============================================
echo   SignAvatar -- ISL to 3D Avatar
echo  ============================================
echo.
echo  Starting server at http://localhost:8000
echo  Press CTRL+C to stop.
echo.

cd /d "%~dp0"
.\venv\Scripts\python.exe backend\app.py
