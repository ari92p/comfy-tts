#!/bin/sh
# Starts the voice service on port 8190 and keeps it running (restarts it if it ends). Run setup.sh once first.
cd "$(dirname "$0")/.." || exit 1
exec ./easy-tts --serve --host 0.0.0.0 --port 8190 -y
