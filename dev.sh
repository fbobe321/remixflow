#!/usr/bin/env bash
# Dev convenience: run the FastAPI backend (:8000) and the Vite dev server
# (:5173, proxying /api -> :8000) together. Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")"

echo "▸ starting backend on http://127.0.0.1:8770 (auto-reload)"
( cd backend && remixflow serve --reload ) &
BACK=$!

echo "▸ starting frontend on http://127.0.0.1:5173"
( cd frontend && npm run dev ) &
FRONT=$!

trap 'kill $BACK $FRONT 2>/dev/null || true' EXIT INT TERM
wait
