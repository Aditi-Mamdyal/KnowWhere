@echo off
:: SearchIQ Desktop Launcher
:: Double-click this file to start the application

cd /d "%~dp0"

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Launch the dark mode GUI (change to gui_light.py if preferred)
start /B pythonw gui.py

:: Exit this window immediately
exit