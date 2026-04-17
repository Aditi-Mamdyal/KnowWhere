@echo off
:: KnowWhere — Desktop Launcher
:: Double-click this file OR create a shortcut on the desktop

:: Step 1: Go to the folder where this .bat file lives
:: This is critical — without this, watchdog looks in the wrong folder
:: and incremental indexing doesn't work when launched from desktop
cd /d "%~dp0"

:: Step 2: Activate virtual environment
call venv\Scripts\activate.bat

:: Step 3: Launch GUI silently (no terminal window shown to user)
:: Change gui_dark.py to gui_light.py if you prefer the light theme
start "" pythonw gui.py

:: Step 4: Exit this launcher window
exit