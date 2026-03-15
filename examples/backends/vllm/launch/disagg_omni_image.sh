#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 2-stage disaggregated omni image generation.
# Stage 0: AR model (GPU 0), Stage 1: DiT/Diffusion (GPU 1)
# Router: orchestrates the pipeline and registers as backend for the frontend.
#
# Supported models: GLM-Image, BAGEL, MammothModa2

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="${MODEL:-THUDM/GLM-4.1V-9B-Thinking}"
STAGE_CONFIG="${STAGE_CONFIG:-glm_image.yaml}"

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --stage-config)
            STAGE_CONFIG="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Disaggregated Omni Image (2-stage, 2 GPUs)" "$MODEL" "$HTTP_PORT"

# Frontend (discovers router as backend)
python -m dynamo.frontend &
FRONTEND_PID=$!
sleep 2

# Stage 0: AR model (GPU 0)
echo "Starting Stage 0 (AR)..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm \
    --model "$MODEL" \
    --omni \
    --stage-configs-path "$STAGE_CONFIG" \
    --stage-id 0 \
    "${EXTRA_ARGS[@]}" &
sleep 5

# Stage 1: Diffusion (GPU 1)
echo "Starting Stage 1 (DiT)..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT_2:-8082} \
    python -m dynamo.vllm \
    --model "$MODEL" \
    --omni \
    --stage-configs-path "$STAGE_CONFIG" \
    --stage-id 1 \
    "${EXTRA_ARGS[@]}" &
sleep 5

# Router (discovers stages, registers as backend)
echo "Starting OmniStageRouter..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT_3:-8083} \
    python -m dynamo.vllm \
    --model "$MODEL" \
    --omni \
    --omni-router \
    --stage-configs-path "$STAGE_CONFIG" \
    --output-modalities image \
    --media-output-fs-url file:///tmp/dynamo_media \
    "${EXTRA_ARGS[@]}" &

wait_any_exit
