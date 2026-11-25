@echo off
rem Short start for Windows: easy-tts  (runs easy-tts.py with the Python that setup.cmd made)
%~dp0.venvs\qwen\Scripts\python.exe "%~dp0src\easy-tts.py" %*
