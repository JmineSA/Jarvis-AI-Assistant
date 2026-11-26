@echo off
rem Starts JARVIS (dashboard + voice) with no console window. Double-click me, or pin me to the taskbar.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Could not find .venv\Scripts\pythonw.exe. Create the virtual environment first, see README.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m jarvis app
