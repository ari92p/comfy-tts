@echo off
rem One-time setup for Windows. Needs nothing installed: it fetches uv, uv fetches Python and the packages.
rem Everything goes into .venvs\ in this folder. The models are downloaded on first use.
cd /d "%~dp0"
set "PY=.venvs\qwen\Scripts\python.exe"
if exist ".venvs\qwen\.tts-ready" (
  echo #easy-tts: Already set up. Start it with: easy-tts.cmd
  goto :end
)

set "UV="
for /f "delims=" %%i in ('where uv 2^>nul') do if not defined UV set "UV=%%i"
if not defined UV (
  set "UV=.venvs\uv\bin\uv.exe"
  if not exist ".venvs\uv\bin\uv.exe" (
    echo #easy-tts: Installing uv ^(the Python setup tool^)...
    set "UV_INSTALL_DIR=%~dp0.venvs\uv\bin"
    set "UV_NO_MODIFY_PATH=1"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" >nul
  )
)
if not exist "%UV%" (
  echo #easy-tts: Could not install uv.
  goto :end
)

echo #easy-tts: Setting up ^(a few minutes^)...
if exist ".venvs\qwen" rmdir /s /q ".venvs\qwen"
"%UV%" venv --quiet --python 3.12 .venvs\qwen || goto :end
rem on Windows the default PyPI torch is CPU-only: use the CUDA build when there is an NVIDIA card
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  "%UV%" pip install --quiet --python "%PY%" torch torchaudio --index-url https://download.pytorch.org/whl/cu128 || goto :end
)
"%UV%" pip install --quiet --python "%PY%" -r src\requirements.txt || goto :end
echo ok> .venvs\qwen\.tts-ready
echo #easy-tts: Done. Start it with: easy-tts.cmd

:end
pause
