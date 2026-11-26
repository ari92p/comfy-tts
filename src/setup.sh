#!/bin/sh
# One-time setup for Mac and Linux. Needs nothing installed: it fetches uv, uv fetches Python and the packages.
# Everything goes into .venvs/ in this folder. The models are downloaded on first use.
cd "$(dirname "$0")/.." || exit 1
say() { echo "#comfy-tts: $*"; }
PY=.venvs/qwen/bin/python
if [ -f .venvs/qwen/.tts-ready ]; then say "Already set up. Start it with: ./comfy-tts"; exit 0; fi

UV=$(command -v uv)
if [ -z "$UV" ]; then
  UV=.venvs/uv/bin/uv
  if [ ! -x "$UV" ]; then
    say "Installing uv (the Python setup tool)..."
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$PWD/.venvs/uv/bin" UV_NO_MODIFY_PATH=1 sh >/dev/null || { say "Could not install uv."; exit 1; }
  fi
fi

say "Setting up (a few minutes)..."
rm -rf .venvs/qwen
"$UV" venv --quiet --python 3.12 .venvs/qwen || exit 1
"$UV" pip install --quiet --python "$PY" -r src/requirements.txt || exit 1
touch .venvs/qwen/.tts-ready
say "Done. Start it with: ./comfy-tts"
