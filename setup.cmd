@echo off
rem One-time setup: installs everything easy-tts needs (inside this folder) and downloads the models.
rem Works without Python on the computer: it fetches uv, and uv fetches Python.
cd /d "%~dp0"
where python >nul 2>nul
if not errorlevel 1 (
  python easy-tts.py --setup
  goto :eof
)
if not exist ".venvs\uv\bin\uv.exe" (
  echo #easy-tts: Installing uv ^(the Python setup tool^)...
  set "UV_INSTALL_DIR=%~dp0.venvs\uv\bin"
  set "UV_NO_MODIFY_PATH=1"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" >nul
)
".venvs\uv\bin\uv.exe" run --python 3.12 easy-tts.py --setup || goto :eof
echo #easy-tts: Start it with: .venvs\uv\bin\uv.exe run --python 3.12 easy-tts.py
