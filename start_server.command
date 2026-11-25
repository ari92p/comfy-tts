#!/bin/sh
# Double-click this file on a Mac to start the voice service (http://127.0.0.1:8190). Close the window or press Ctrl+C to stop it.
# To make it reachable from other computers add: --host 0.0.0.0   (there is no login, use a firewall)
cd "$(dirname "$0")" || exit 1
exec python3 tts.py --serve --port 8190 -y
