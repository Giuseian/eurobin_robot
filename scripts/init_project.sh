#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$PROJECT_ROOT/shared_data/realsense/rgb"
mkdir -p "$PROJECT_ROOT/shared_data/realsense/depth"
mkdir -p "$PROJECT_ROOT/shared_data/realsense/camera"
mkdir -p "$PROJECT_ROOT/shared_data/realsense/masks"
mkdir -p "$PROJECT_ROOT/shared_data/realsense/outputs"
mkdir -p "$PROJECT_ROOT/shared_data/meshes"

chmod -R 777 shared_data

echo "Project folders initialized in: $PROJECT_ROOT"