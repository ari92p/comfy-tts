#!/bin/sh
# One-time setup: installs everything easy-tts needs (inside this folder) and downloads the models.
# Works without Python on the computer: it fetches uv, and uv fetches Python.
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then
  exec python3 easy-tts.py --setup
fi
UV=.venvs/uv/bin/uv
if [ ! -x "$UV" ]; then
  echo "#easy-tts: Installing uv (the Python setup tool)..."
  curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$PWD/.venvs/uv/bin" UV_NO_MODIFY_PATH=1 sh >/dev/null || exit 1
fi
"$UV" run --python 3.12 easy-tts.py --setup || exit 1
echo "#easy-tts: Start it with: $UV run --python 3.12 easy-tts.py"
