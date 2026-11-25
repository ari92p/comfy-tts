@echo off
rem Starts the voice service on port 8190 and keeps it running (restarts it if it ends). Run setup.cmd once first.
cd /d "%~dp0"
.venvs\qwen\Scripts\python.exe easy-tts.py --serve --host 0.0.0.0 --port 8190 -y
