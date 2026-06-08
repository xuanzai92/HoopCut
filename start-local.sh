#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

find_available_port() {
  python3 - "$1" <<'PY'
import socket
import sys

start = int(sys.argv[1])

for port in range(start, start + 20):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        raise SystemExit(0)

raise SystemExit(1)
PY
}

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required."
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required."
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required."
  exit 1
fi

BACKEND_PORT="${BACKEND_PORT:-$(find_available_port 5050)}"
if [[ -z "${BACKEND_PORT:-}" ]]; then
  echo "Failed to find an available backend port."
  exit 1
fi

cd "$BACKEND_DIR"
if [[ ! -d venv ]]; then
  python3 -m venv venv
fi

source venv/bin/activate
python -m pip install -r requirements.txt
export BACKEND_PORT
export FRONTEND_PORT
venv/bin/python app.py &
BACKEND_PID=$!

cd "$FRONTEND_DIR"
if [[ ! -d node_modules ]]; then
  npm install
fi

echo "Frontend: http://127.0.0.1:${FRONTEND_PORT}"
echo "Backend:  http://127.0.0.1:${BACKEND_PORT}"
VITE_PROXY_TARGET="http://127.0.0.1:${BACKEND_PORT}" \
VITE_API_BASE_URL="" \
VITE_SOCKET_URL="" \
npm run dev -- --host 127.0.0.1 --port "${FRONTEND_PORT}"
