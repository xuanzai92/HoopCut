#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"

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

is_port_available() {
  python3 - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        raise SystemExit(1)

raise SystemExit(0)
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

if [[ -n "${BACKEND_PORT:-}" ]]; then
  if ! is_port_available "$BACKEND_PORT"; then
    echo "Backend port ${BACKEND_PORT} is already in use."
    exit 1
  fi
else
  BACKEND_PORT="$(find_available_port 5050)"
fi

if [[ -z "${BACKEND_PORT:-}" ]]; then
  echo "Failed to find an available backend port."
  exit 1
fi

if [[ -n "${FRONTEND_PORT:-}" ]]; then
  if ! is_port_available "$FRONTEND_PORT"; then
    echo "Frontend port ${FRONTEND_PORT} is already in use."
    exit 1
  fi
else
  FRONTEND_PORT="$(find_available_port 5173)"
fi

if [[ -z "${FRONTEND_PORT:-}" ]]; then
  echo "Failed to find an available frontend port."
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

BACKEND_URL="http://127.0.0.1:${BACKEND_PORT}"

cd "$FRONTEND_DIR"
if [[ ! -d node_modules ]]; then
  npm install
fi

echo "Frontend: http://127.0.0.1:${FRONTEND_PORT}"
echo "Backend:  ${BACKEND_URL}"
VITE_PROXY_TARGET="${BACKEND_URL}" \
VITE_API_BASE_URL="${BACKEND_URL}" \
VITE_SOCKET_URL="${BACKEND_URL}" \
npm run dev -- --host 127.0.0.1 --port "${FRONTEND_PORT}"
