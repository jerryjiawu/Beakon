@echo off
title Acoustic Detection System
echo.
echo ========================================
echo   Acoustic Detection System Launcher
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

REM Install requirements if needed
if exist requirements.txt (
    echo Installing/updating requirements...
    pip install -r requirements.txt
    echo.
)

REM Launch the system
echo Starting Acoustic Detection System...
echo.
python launch.py

echo.
echo System stopped.
pause
