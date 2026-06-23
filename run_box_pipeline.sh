#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Automated manipulation pipeline:
#   1. robot_docker      -> RGB-D acquisition
#   2. perception_docker -> SAM3 segmentation
#   3. perception_docker -> FoundationPose estimation
#   4. robot_docker      -> object_pose.txt update
#   5. robot_docker      -> Centauro grasp execution
#
# The script includes visual previews and user confirmation after:
#   - RGB-D acquisition
#   - SAM3 segmentation
#   - FoundationPose estimation
# ============================================================

# === Docker Compose directories on the host ===
ROBOT_DOCKER_DIR="$HOME/eurobin_robot/robot_docker"
PERCEPTION_DOCKER_DIR="$HOME/eurobin_robot/perception_docker"

# === Docker Compose service names ===
ROBOT_SERVICE="dev"
PERCEPTION_SERVICE="perception"

# === Shared data path inside perception_docker ===
DATA_ROOT="/workspace/shared_data/realsense"

# === Manipulation paths inside robot_docker ===
ROBOT_MANIPULATION_DIR="/home/user/eurobin/manipulation_code/centauro"
ROBOT_OBJECT_POSE_FILE="${ROBOT_MANIPULATION_DIR}/object_pose.txt"
ROBOT_GRASP_SCRIPT="${ROBOT_MANIPULATION_DIR}/grasp_centauro_test_world.py"

# === Pipeline parameters ===
PROMPT="box"
IMAGE_ID="000000"

MESH_FILE="${DATA_ROOT}/meshes/box_amazon_model/meshes/box_amazon_model.obj"

CAPTURE_SECONDS=10
CAPTURE_HZ=30
RESIZE_WIDTH=640
RESIZE_HEIGHT=360

# === Preview directory on the host ===
PREVIEW_DIR="/tmp/box_pipeline_preview"
mkdir -p "$PREVIEW_DIR"

# === Global pose variables ===
FP_X=""
FP_Y=""
FP_Z=""

ROBOT_POS_1=""
ROBOT_POS_2=""
ROBOT_POS_3=""

preview_image_from_perception() {
  local container_image_path="$1"
  local host_image_path="$2"
  local description="$3"

  echo ""
  echo "[preview] ${description}"
  echo "[preview] Copying image from perception_docker:"
  echo "          ${container_image_path}"
  echo "[preview] Temporary host path:"
  echo "          ${host_image_path}"

  cd "$PERCEPTION_DOCKER_DIR"

  docker compose exec -T "$PERCEPTION_SERVICE" bash -lc "
    set -e
    test -f '${container_image_path}' || {
      echo 'Error: image not found: ${container_image_path}' >&2
      exit 1
    }
    cat '${container_image_path}'
  " > "$host_image_path"

  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$host_image_path" >/dev/null 2>&1 &
  elif command -v eog >/dev/null 2>&1; then
    eog "$host_image_path" >/dev/null 2>&1 &
  elif command -v display >/dev/null 2>&1; then
    display "$host_image_path" >/dev/null 2>&1 &
  else
    echo "[preview] No image viewer found. Open this file manually:"
    echo "          ${host_image_path}"
  fi
}

copy_text_from_perception() {
  local container_text_path="$1"
  local host_text_path="$2"

  cd "$PERCEPTION_DOCKER_DIR"

  docker compose exec -T "$PERCEPTION_SERVICE" bash -lc "
    set -e
    test -f '${container_text_path}' || {
      echo 'Error: file not found: ${container_text_path}' >&2
      exit 1
    }
    cat '${container_text_path}'
  " > "$host_text_path"
}

read_and_print_object_position_from_pose_file() {
  local container_pose_path="$1"
  local host_pose_path="$2"

  echo ""
  echo "[pose] Reading FoundationPose output:"
  echo "       ${container_pose_path}"

  copy_text_from_perception "$container_pose_path" "$host_pose_path"

  local values
  values=$(
    awk '
      NR==1 {x=$4}
      NR==2 {y=$4}
      NR==3 {z=$4}
      END {
        printf "%.6f %.6f %.6f %.6f %.6f %.6f", x, y, z, z, x, -y
      }
    ' "$host_pose_path"
  )

  read -r FP_X FP_Y FP_Z ROBOT_POS_1 ROBOT_POS_2 ROBOT_POS_3 <<< "$values"

  echo ""
  echo "=========================================="
  echo " Object position in camera frame"
  echo "=========================================="
  echo " x = ${FP_X} m   right"
  echo " y = ${FP_Y} m   down"
  echo " z = ${FP_Z} m   depth"
  echo "=========================================="
  echo ""
  echo "=========================================="
  echo " Position to write for manipulation"
  echo "=========================================="
  echo " object.position: ${ROBOT_POS_1} ${ROBOT_POS_2} ${ROBOT_POS_3}"
  echo ""
  echo " mapping:"
  echo "   first term  = z"
  echo "   second term = x"
  echo "   third term  = -y"
  echo "=========================================="
  echo ""
  echo "[pose] Pose file copied to host:"
  echo "       ${host_pose_path}"
}

write_object_pose_to_robot() {
  local p1="$1"
  local p2="$2"
  local p3="$3"

  echo ""
  echo "[robot] Writing object.position inside robot_docker..."
  echo "[robot] Target file:"
  echo "        ${ROBOT_OBJECT_POSE_FILE}"
  echo "[robot] New content:"
  echo "        object.position: ${p1} ${p2} ${p3}"

  cd "$ROBOT_DOCKER_DIR"

  docker compose exec -T "$ROBOT_SERVICE" bash -lc "
    set -e

    target_file='${ROBOT_OBJECT_POSE_FILE}'
    target_dir=\$(dirname \"\$target_file\")

    test -d \"\$target_dir\" || {
      echo 'Error: directory not found:' \"\$target_dir\" >&2
      exit 1
    }

    printf 'object.position: %s %s %s\n' '${p1}' '${p2}' '${p3}' > \"\$target_file\"

    echo ''
    echo 'Updated object_pose.txt content:'
    cat \"\$target_file\"
  "
}

run_grasp_centauro() {
  echo ""
  echo "[robot] Launching grasp_centauro_test_world.py inside robot_docker..."
  echo "[robot] Working directory:"
  echo "        ${ROBOT_MANIPULATION_DIR}"
  echo "[robot] Script:"
  echo "        ${ROBOT_GRASP_SCRIPT}"

  cd "$ROBOT_DOCKER_DIR"

  # Use an interactive shell to reproduce the environment obtained with:
  # docker compose exec dev bash
  docker compose exec "$ROBOT_SERVICE" bash -ic "
    set -e

    cd '${ROBOT_MANIPULATION_DIR}'

    test -f '${ROBOT_OBJECT_POSE_FILE}' || {
      echo 'Error: object_pose.txt not found: ${ROBOT_OBJECT_POSE_FILE}' >&2
      exit 1
    }

    test -f '${ROBOT_GRASP_SCRIPT}' || {
      echo 'Error: grasp script not found: ${ROBOT_GRASP_SCRIPT}' >&2
      exit 1
    }

    echo ''
    echo 'Environment used for grasp execution:'
    echo \"USER=\$(whoami)\"
    echo \"PWD=\$(pwd)\"
    echo \"CONDA_DEFAULT_ENV=\${CONDA_DEFAULT_ENV:-}\"
    echo \"VIRTUAL_ENV=\${VIRTUAL_ENV:-}\"
    echo \"PYTHONPATH=\${PYTHONPATH:-}\"
    echo \"AMENT_PREFIX_PATH=\${AMENT_PREFIX_PATH:-}\"
    echo ''
    echo 'Python interpreter:'
    which python3
    python3 -c 'import sys; print(sys.executable)'

    echo ''
    echo 'Checking cartesian_interface_ros import...'
    python3 -c 'import cartesian_interface_ros; print(\"cartesian_interface_ros OK\")'

    echo ''
    echo 'object_pose.txt used for grasp execution:'
    cat '${ROBOT_OBJECT_POSE_FILE}'
    echo ''

    python3 '${ROBOT_GRASP_SCRIPT}' --ros-args \
      -p object_pose_file:=object_pose.txt \
      -p base_frame:=world \
      -p cartesian_world_frame:=ci/world \
      -p cartesian_robot_base_frame:=ci/pelvis \
      -p robot_base_frame:=pelvis \
      -p camera_frame:=D435_head_camera_link \
      -p approach_mode:=yz \
      -p grasp_offset_x:=0.12 \
      -p grasp_offset_y:=-0.05 \
      -p grasp_offset_z:=0.0 \
      -p phase1_split_fraction:=0.5 \
      -p constrain_orientation:=true
  "
}

ask_continue_or_stop() {
  local message="$1"
  local answer=""

  echo ""
  read -r -p "${message} [y/N]: " answer

  case "$answer" in
    y|Y|yes|Yes|YES|s|S|si|Si|SI|sì|Sì|SÌ)
      echo "Continuing..."
      ;;
    *)
      echo "Pipeline stopped by user."
      exit 0
      ;;
  esac
}

# === ROS setup inside robot_docker ===
ROBOT_SETUP='
set -e

source /opt/ros/jazzy/setup.bash

if [ -f /home/user/xbot2_ws/install/setup.bash ]; then
  source /home/user/xbot2_ws/install/setup.bash
fi

if [ -f /home/user/env/bin/activate ]; then
  source /home/user/env/bin/activate
fi
'

echo "[1/6] Setting align_depth.enable..."

cd "$ROBOT_DOCKER_DIR"

docker compose exec -T "$ROBOT_SERVICE" bash -lc "
  ${ROBOT_SETUP}

  echo 'Waiting for node /head_cam to become available...'

  for i in \$(seq 1 30); do
    if ros2 node list | grep -qx '/head_cam'; then
      echo 'Node /head_cam found.'
      break
    fi

    if [ \$i -eq 30 ]; then
      echo 'Error: node /head_cam not found after 30 seconds.'
      echo 'Available ROS nodes:'
      ros2 node list || true
      exit 1
    fi

    sleep 1
  done

  ros2 param set /head_cam align_depth.enable true
"

echo "[2/6] Capturing RGB-D frames for ${CAPTURE_SECONDS} seconds..."

docker compose exec -T "$ROBOT_SERVICE" bash -lc "
  ${ROBOT_SETUP}

  set +e

  timeout --foreground --signal=SIGINT ${CAPTURE_SECONDS}s \
    python3 ~/PoseEstimation/pipeline/save_realsense_rgbd.py \
      --capture_hz ${CAPTURE_HZ} \
      --resize_width ${RESIZE_WIDTH} \
      --resize_height ${RESIZE_HEIGHT}

  code=\$?

  # timeout returns 124 when it interrupts the process.
  # This is expected here because it acts as an automatic Ctrl+C.
  if [ \$code -eq 124 ]; then
    exit 0
  else
    exit \$code
  fi
"

echo "[3/6] Retrieving the latest timestamp from rgb/ ..."

cd "$PERCEPTION_DOCKER_DIR"

TIMESTAMP=$(
  docker compose exec -T "$PERCEPTION_SERVICE" bash -lc "
    set -e

    find '${DATA_ROOT}/rgb' \
      -mindepth 1 \
      -maxdepth 1 \
      -type d \
      -printf '%f\n' \
    | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}$' \
    | sort \
    | tail -n 1
  " | tr -d '\r'
)

if [ -z "$TIMESTAMP" ]; then
  echo "Error: no timestamp found in ${DATA_ROOT}/rgb"
  exit 1
fi

echo "Selected timestamp: ${TIMESTAMP}"

echo "[check] Verifying RGB, depth, camera, and mesh data..."

docker compose exec -T "$PERCEPTION_SERVICE" bash -lc "
  set -e

  test -d '${DATA_ROOT}/rgb/${TIMESTAMP}' || {
    echo 'Error: missing ${DATA_ROOT}/rgb/${TIMESTAMP}'
    exit 1
  }

  test -d '${DATA_ROOT}/depth/${TIMESTAMP}' || {
    echo 'Error: missing ${DATA_ROOT}/depth/${TIMESTAMP}'
    exit 1
  }

  test -d '${DATA_ROOT}/camera/${TIMESTAMP}' || {
    echo 'Error: missing ${DATA_ROOT}/camera/${TIMESTAMP}'
    exit 1
  }

  test -f '${DATA_ROOT}/rgb/${TIMESTAMP}/${IMAGE_ID}.png' || {
    echo 'Error: missing RGB image ${DATA_ROOT}/rgb/${TIMESTAMP}/${IMAGE_ID}.png'
    exit 1
  }

  test -f '${MESH_FILE}' || {
    echo 'Error: mesh file not found: ${MESH_FILE}'
    exit 1
  }

  echo 'RGB directory:'
  ls -lah '${DATA_ROOT}/rgb/${TIMESTAMP}' | head

  echo 'Depth directory:'
  ls -lah '${DATA_ROOT}/depth/${TIMESTAMP}' | head

  echo 'Camera directory:'
  ls -lah '${DATA_ROOT}/camera/${TIMESTAMP}' | head
"

# === RGB preview after acquisition ===
RGB_IMAGE_CONTAINER="${DATA_ROOT}/rgb/${TIMESTAMP}/${IMAGE_ID}.png"
RGB_IMAGE_HOST="${PREVIEW_DIR}/${TIMESTAMP}_${IMAGE_ID}_rgb.png"

preview_image_from_perception \
  "$RGB_IMAGE_CONTAINER" \
  "$RGB_IMAGE_HOST" \
  "First captured RGB image"

ask_continue_or_stop "Continue with SAM3 segmentation?"

echo "[4/6] Running SAM3 segmentation..."

docker compose exec -T "$PERCEPTION_SERVICE" bash -lc "
  set -e

  if command -v conda >/dev/null 2>&1; then
    source \"\$(conda info --base)/etc/profile.d/conda.sh\"
  elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
  elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
  elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
  else
    echo 'Error: conda.sh not found'
    exit 1
  fi

  conda activate sam3

  python /workspace/PoseEstimation/pipeline/sam_script_fp.py \
    --data_root '${DATA_ROOT}' \
    --timestamp '${TIMESTAMP}' \
    --prompt '${PROMPT}' \
    --image_id '${IMAGE_ID}'

  conda deactivate
"

# === SAM3 mask preview ===
MASK_IMAGE_CONTAINER="${DATA_ROOT}/masks/${TIMESTAMP}/${IMAGE_ID}.png"
MASK_IMAGE_HOST="${PREVIEW_DIR}/${TIMESTAMP}_${IMAGE_ID}_mask.png"

preview_image_from_perception \
  "$MASK_IMAGE_CONTAINER" \
  "$MASK_IMAGE_HOST" \
  "Generated SAM3 mask"

ask_continue_or_stop "Continue with FoundationPose?"

echo "[5/6] Running FoundationPose..."

docker compose exec -T "$PERCEPTION_SERVICE" bash -lc "
  set -e

  if command -v conda >/dev/null 2>&1; then
    source \"\$(conda info --base)/etc/profile.d/conda.sh\"
  elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
  else
    echo 'Error: conda.sh not found'
    exit 1
  fi

  conda activate my

  echo 'Active environment for FoundationPose:'
  echo \"CONDA_DEFAULT_ENV=\$CONDA_DEFAULT_ENV\"

  echo 'Python interpreter for FoundationPose:'
  which python

  echo 'Checking cv2 import...'
  python -c 'import cv2; print(\"cv2 OK\", cv2.__version__)'

  python /workspace/PoseEstimation/pipeline/run_fp_single_frame.py \
    --data_root '${DATA_ROOT}' \
    --timestamp '${TIMESTAMP}' \
    --mesh_file '${MESH_FILE}' \
    --start_frame '${IMAGE_ID}' \
    --max_frames 20 \
    --est_refine_iter 5 \
    --track_refine_iter 2 \
    --debug 1
"

# === FoundationPose output preview ===
FP_VIS_CONTAINER="${DATA_ROOT}/outputs/${TIMESTAMP}/vis/${IMAGE_ID}.png"
FP_VIS_HOST="${PREVIEW_DIR}/${TIMESTAMP}_${IMAGE_ID}_foundationpose_vis.png"

FP_POSE_CONTAINER="${DATA_ROOT}/outputs/${TIMESTAMP}/ob_in_cam/${IMAGE_ID}.txt"
FP_POSE_HOST="${PREVIEW_DIR}/${TIMESTAMP}_${IMAGE_ID}_ob_in_cam.txt"

preview_image_from_perception \
  "$FP_VIS_CONTAINER" \
  "$FP_VIS_HOST" \
  "FoundationPose output with bounding box"

read_and_print_object_position_from_pose_file \
  "$FP_POSE_CONTAINER" \
  "$FP_POSE_HOST"

ask_continue_or_stop "Is the FoundationPose output acceptable and should object_pose.txt be updated?"

write_object_pose_to_robot \
  "$ROBOT_POS_1" \
  "$ROBOT_POS_2" \
  "$ROBOT_POS_3"

ask_continue_or_stop "Execute the Centauro grasp now?"

echo "[6/6] Executing Centauro grasp..."

run_grasp_centauro

echo ""
echo "Pipeline completed."
echo "Timestamp used: ${TIMESTAMP}"
echo ""
echo "RGB preview:              ${RGB_IMAGE_HOST}"
echo "SAM3 mask preview:        ${MASK_IMAGE_HOST}"
echo "FoundationPose preview:   ${FP_VIS_HOST}"
echo "Copied pose file:         ${FP_POSE_HOST}"
echo ""
echo "Mask directory:           ${DATA_ROOT}/masks/${TIMESTAMP}"
echo "FoundationPose output:    ${DATA_ROOT}/outputs/${TIMESTAMP}"
echo ""
echo "Robot object pose file:   ${ROBOT_OBJECT_POSE_FILE}"
echo "Written value:"
echo "object.position: ${ROBOT_POS_1} ${ROBOT_POS_2} ${ROBOT_POS_3}"
echo ""
echo "Executed grasp script:"
echo "${ROBOT_GRASP_SCRIPT}"