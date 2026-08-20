#!/usr/bin/env bash
set -euo pipefail

ENGINE_PATH="${ACCELSERVE_ENGINE_PATH:-/app/inference/accelserve_mlp_fp16.engine}"
MANIFEST_PATH="${ENGINE_PATH}.json"

echo "AccelServe GPU startup"
echo "TensorRT engine: ${ENGINE_PATH}"

if [ ! -f "${ENGINE_PATH}" ] || [ ! -f "${MANIFEST_PATH}" ]; then
    echo "TensorRT engine or manifest not found."
    echo "Building engine on this NVIDIA host..."

    cd /app

    ACCELSERVE_ENGINE_PATH="${ENGINE_PATH}" \
    python3 inference/build_engine.py

    echo "TensorRT engine build complete."
else
    echo "Existing TensorRT engine found."
fi

echo "Starting AccelServe..."

exec python3 -m uvicorn \
    api.server:app \
    --host 0.0.0.0 \
    --port 8000
