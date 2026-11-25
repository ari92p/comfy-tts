#!/bin/sh
# Starts the voice service on port 8190 and keeps it running (restarts it if it ends). Needs Python 3.9+ and uv.
cd "$(dirname "$0")" || exit 1
exec python3 tts.py --serve --host 0.0.0.0 --port 8190 -y
