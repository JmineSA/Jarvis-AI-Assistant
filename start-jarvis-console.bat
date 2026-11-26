@echo off
rem Same as start-jarvis.bat but keeps a console open so you can read errors.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m jarvis app
pause
