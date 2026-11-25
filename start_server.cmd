@echo off
rem Starts the voice service on port 8190 and keeps it running (restarts it if it ends). Needs Python 3.9+ and uv on the PATH.
cd /d "%~dp0"
python tts.py --serve --host 0.0.0.0 --port 8190 -y
