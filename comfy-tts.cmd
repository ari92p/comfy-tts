@echo off
rem Short start for Windows: comfy-tts  (runs comfy-tts.py with the Python that setup.cmd made)
%~dp0.venvs\qwen\Scripts\python.exe "%~dp0src\comfy-tts.py" %*
