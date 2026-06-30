#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class Config:
    # Docker Compose directories on the host
    robot_docker_dir: Path = Path.home() / "eurobin_robot" / "robot_docker"
    perception_docker_dir: Path = Path.home() / "eurobin_robot" / "perception_docker"

    # Docker Compose service names
    robot_service: str = "dev"
    perception_service: str = "perception"

    # Shared data path inside perception_docker
    data_root: str = "/workspace/shared_data/realsense"

    # Manipulation paths inside robot_docker
    robot_manipulation_dir: str = "/home/user/eurobin/manipulation_code/centauro"
    robot_object_pose_file: str = "/home/user/eurobin/manipulation_code/centauro/object_pose.txt"
    robot_grasp_script: str = "/home/user/eurobin/manipulation_code/centauro/grasp_centauro_test_world.py"
    robot_push_script: str = "/home/user/eurobin/manipulation_code/centauro/push_box_xyz.py"

    # Pipeline parameters
    image_id: str = "000000"
    capture_hz: int = 30
    resize_width: int = 640
    resize_height: int = 360
    max_frames: int = 1
    max_pipeline_cycles: int = 10

    # VLM scene perception parameters
    vlm_scenario: str = "centauro"
    vlm_scene_v: str = "v6"
    #vlm_scene_model: str = "gpt-5.2"
    vlm_scene_model: str = "gemini-robotics-er-1.6-preview"

    # VLM task planning / simulation / validation parameters
    vlm_task: str = "Grasp the brown box. The red box can also fall on the ground."
    vlm_plan_v: str = "v5"
    vlm_sim_v: str = "v5"
    vlm_validator_v: str = "v5"
    vlm_plan_model: str = "gemini-robotics-er-1.6-preview"
    vlm_sim_model: str = "gemini-robotics-er-1.6-preview"
    vlm_validator_model: str = "gemini-robotics-er-1.6-preview"

    # live2 new-frame waiting parameters
    vlm_new_frame_timeout: float = 3600.0
    vlm_new_frame_poll_seconds: float = 1.0

    # Scene enrichment debug
    scene_enrichment_debug: bool = True

    # FoundationPose parameters
    foundationpose_max_frames: int = 1
    foundationpose_debug: int = 1
    est_refine_iter: int = 5
    track_refine_iter: int = 2

    # Manipulation defaults
    default_grasp_box_width_extra_m: float = 0.120

    # Preview directory on the host
    preview_dir: Path = Path("/tmp/box_pipeline_preview")

    @property
    def mesh_file(self) -> str:
        return f"{self.data_root}/meshes/box_amazon_model/meshes/box_amazon_model.obj"

    @property
    def mesh_metadata_file(self) -> str:
        return f"{self.data_root}/meshes/mesh_metadata.json"

    @property
    def objects_root(self) -> str:
        return f"{self.data_root}/objects"

    @property
    def masks_by_object_root(self) -> str:
        return f"{self.data_root}/masks_by_object"

    @property
    def masks_by_objects_root(self) -> str:
        return f"{self.data_root}/masks_by_objects"

    @property
    def outputs_by_object_root(self) -> str:
        return f"{self.data_root}/outputs_by_object"

    @property
    def outputs_by_objects_root(self) -> str:
        return f"{self.data_root}/outputs_by_objects"


@dataclass
class PoseResult:
    fp_x: float
    fp_y: float
    fp_z: float
    robot_pos_1: float
    robot_pos_2: float
    robot_pos_3: float

    @property
    def robot_object_position_line(self) -> str:
        return (
            f"object.position: "
            f"{self.robot_pos_1:.6f} "
            f"{self.robot_pos_2:.6f} "
            f"{self.robot_pos_3:.6f}"
        )


@dataclass
class VLMTaskValidationResult:
    task_completed: bool
    outcome: str
    saw_non_matching: bool
    should_restart: bool


ROBOT_SETUP = r"""
set -e

source /opt/ros/jazzy/setup.bash

if [ -f /home/user/xbot2_ws/install/setup.bash ]; then
  source /home/user/xbot2_ws/install/setup.bash
fi

if [ -f /home/user/env/bin/activate ]; then
  source /home/user/env/bin/activate
fi

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
"""


def _print_captured_failure(exc: subprocess.CalledProcessError) -> None:
    if exc.stdout:
        print()
        print("[error] Captured stdout:")
        if isinstance(exc.stdout, bytes):
            print(exc.stdout.decode(errors="replace"))
        else:
            print(exc.stdout)

    if exc.stderr:
        print()
        print("[error] Captured stderr:")
        if isinstance(exc.stderr, bytes):
            print(exc.stderr.decode(errors="replace"))
        else:
            print(exc.stderr)


def run_cmd(
    cmd: list[str],
    *,
    cwd: Optional[Path] = None,
    check: bool = True,
    capture_output: bool = False,
    text: bool = True,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=check,
            capture_output=capture_output,
            text=text,
            bufsize=1 if text else -1,
        )
    except subprocess.CalledProcessError as exc:
        if capture_output:
            _print_captured_failure(exc)
        raise


def docker_exec_bash(
    compose_dir: Path,
    service: str,
    script: str,
    *,
    interactive: bool = False,
    allocate_tty: bool = False,
    capture_output: bool = False,
    text: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess:
    cmd = ["docker", "compose", "exec"]

    if not allocate_tty:
        cmd.append("-T")

    streaming_setup = """
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
"""

    full_script = f"""
{streaming_setup}
{script}
"""

    cmd.extend([service, "bash", "-ic" if interactive else "-lc", full_script])

    return run_cmd(
        cmd,
        cwd=compose_dir,
        check=check,
        capture_output=capture_output,
        text=text,
    )


def docker_exec_bash_streaming(
    compose_dir: Path,
    service: str,
    script: str,
    *,
    on_stdout_line: Optional[Callable[[str], None]] = None,
    interactive: bool = False,
    allocate_tty: bool = False,
) -> None:
    cmd = ["docker", "compose", "exec"]

    if not allocate_tty:
        cmd.append("-T")

    streaming_setup = """
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
"""

    full_script = f"""
{streaming_setup}
{script}
"""

    cmd.extend([service, "bash", "-ic" if interactive else "-lc", full_script])

    process = subprocess.Popen(
        cmd,
        cwd=str(compose_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None

    try:
        for line in process.stdout:
            print(line, end="", flush=True)

            if on_stdout_line is not None:
                on_stdout_line(line)

        return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)

    except KeyboardInterrupt:
        process.terminate()

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

        raise


def ask_continue_or_stop(message: str) -> None:
    answer = input(f"\n{message} [y/N]: ").strip()

    if is_yes(answer):
        print("Continuing...")
        return

    print("Pipeline stopped by user.")
    raise SystemExit(0)


def is_yes(answer: str) -> bool:
    return answer.strip() in {
        "y",
        "Y",
        "yes",
        "Yes",
        "YES",
        "s",
        "S",
        "si",
        "Si",
        "SI",
        "sì",
        "Sì",
        "SÌ",
    }


def open_image(path: Path) -> None:
    viewers = ["xdg-open", "eog", "display"]

    for viewer in viewers:
        if shutil.which(viewer):
            subprocess.Popen(
                [viewer, str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

    print("[preview] No image viewer found. Open this file manually:")
    print(f"          {path}")


def copy_binary_from_perception(cfg: Config, container_path: str, host_path: Path) -> None:
    script = f"""
set -e
test -f {shlex.quote(container_path)} || {{
  echo 'Error: file not found: {container_path}' >&2
  exit 1
}}
cat {shlex.quote(container_path)}
"""

    result = docker_exec_bash(
        cfg.perception_docker_dir,
        cfg.perception_service,
        script,
        capture_output=True,
        text=False,
    )

    host_path.parent.mkdir(parents=True, exist_ok=True)
    host_path.write_bytes(result.stdout)


def copy_text_from_perception(cfg: Config, container_path: str, host_path: Path) -> None:
    script = f"""
set -e
test -f {shlex.quote(container_path)} || {{
  echo 'Error: file not found: {container_path}' >&2
  exit 1
}}
cat {shlex.quote(container_path)}
"""

    result = docker_exec_bash(
        cfg.perception_docker_dir,
        cfg.perception_service,
        script,
        capture_output=True,
        text=True,
    )

    host_path.parent.mkdir(parents=True, exist_ok=True)
    host_path.write_text(result.stdout)


def read_json_from_perception(
    cfg: Config,
    container_path: str,
    *,
    required: bool = True,
) -> dict[str, Any]:
    script = f"""
set -e

if [ ! -f {shlex.quote(container_path)} ]; then
  if [ {str(required).lower()} = true ]; then
    echo 'Error: JSON file not found: {container_path}' >&2
    exit 1
  fi
  exit 0
fi

cat {shlex.quote(container_path)}
"""

    result = docker_exec_bash(
        cfg.perception_docker_dir,
        cfg.perception_service,
        script,
        capture_output=True,
        text=True,
    )

    text = result.stdout.strip()

    if not text:
        return {}

    data = json.loads(text)

    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object in {container_path}")

    return data


def preview_image_from_perception(
    cfg: Config,
    container_image_path: str,
    host_image_path: Path,
    description: str,
) -> None:
    print()
    print(f"[preview] {description}")
    print("[preview] Copying image from perception_docker:")
    print(f"          {container_image_path}")
    print("[preview] Temporary host path:")
    print(f"          {host_image_path}")

    copy_binary_from_perception(cfg, container_image_path, host_image_path)
    open_image(host_image_path)


def parse_pose_file(host_pose_path: Path) -> PoseResult:
    lines = host_pose_path.read_text().strip().splitlines()

    if len(lines) < 3:
        raise RuntimeError(f"Invalid pose file: expected at least 3 rows, got {len(lines)}")

    matrix_rows: list[list[float]] = []

    for row_idx, line in enumerate(lines[:3], start=1):
        values = line.split()

        if len(values) < 4:
            raise RuntimeError(
                f"Invalid pose file row {row_idx}: expected at least 4 columns, got {len(values)}"
            )

        matrix_rows.append([float(v) for v in values[:4]])

    fp_x = matrix_rows[0][3]
    fp_y = matrix_rows[1][3]
    fp_z = matrix_rows[2][3]

    robot_pos_1 = fp_z
    robot_pos_2 = -fp_x
    robot_pos_3 = -fp_y

    return PoseResult(
        fp_x=fp_x,
        fp_y=fp_y,
        fp_z=fp_z,
        robot_pos_1=robot_pos_1,
        robot_pos_2=robot_pos_2,
        robot_pos_3=robot_pos_3,
    )


def read_and_print_object_position_from_pose_file(
    cfg: Config,
    container_pose_path: str,
    host_pose_path: Path,
) -> PoseResult:
    print()
    print("[pose] Reading FoundationPose output:")
    print(f"       {container_pose_path}")

    copy_text_from_perception(cfg, container_pose_path, host_pose_path)
    pose = parse_pose_file(host_pose_path)

    print()
    print("==========================================")
    print(" Object position in camera frame")
    print("==========================================")
    print(f" x = {pose.fp_x:.6f} m   right")
    print(f" y = {pose.fp_y:.6f} m   down")
    print(f" z = {pose.fp_z:.6f} m   depth")
    print("==========================================")
    print()
    print("==========================================")
    print(" Position to write for manipulation")
    print("==========================================")
    print(f" {pose.robot_object_position_line}")
    print()
    print(" mapping:")
    print("   first term  = z")
    print("   second term = -x")
    print("   third term  = -y")
    print("==========================================")
    print()
    print("[pose] Pose file copied to host:")
    print(f"       {host_pose_path}")

    return pose


def extract_strings_from_json_like_data(data: Any) -> list[str]:
    if isinstance(data, list):
        objects: list[str] = []

        for item in data:
            if isinstance(item, str) and item.strip():
                objects.append(item.strip())
            elif isinstance(item, dict):
                objects.extend(extract_strings_from_json_like_data(item))

        return objects

    if isinstance(data, dict):
        preferred_keys = [
            "objects",
            "detected_objects",
            "prompts",
            "items",
            "classes",
            "labels",
        ]

        for key in preferred_keys:
            if key in data:
                extracted = extract_strings_from_json_like_data(data[key])
                if extracted:
                    return extracted

        objects: list[str] = []

        for value in data.values():
            objects.extend(extract_strings_from_json_like_data(value))

        return objects

    return []


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []

    for value in values:
        clean = value.strip()
        key = clean.lower()

        if not clean or key in seen:
            continue

        seen.add(key)
        output.append(clean)

    return output


def parse_detected_objects_text(text: str) -> list[str]:
    raw = text.strip()

    if not raw:
        return []

    candidates = [raw]

    match = re.search(r"\[[\s\S]*\]", raw)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            objects = extract_strings_from_json_like_data(data)
            if objects:
                return unique_preserve_order(objects)
        except Exception:
            pass

        try:
            data = ast.literal_eval(candidate)
            objects = extract_strings_from_json_like_data(data)
            if objects:
                return unique_preserve_order(objects)
        except Exception:
            pass

    line_objects: list[str] = []

    for line in raw.splitlines():
        clean = line.strip().strip("-*•").strip().strip('"').strip("'")

        if clean and not clean.startswith("{") and not clean.startswith("["):
            line_objects.append(clean)

    return unique_preserve_order(line_objects)


def resolve_existing_timestamp_root(
    cfg: Config,
    candidate_roots: list[str],
    timestamp: str,
) -> str:
    candidates = [f"{root}/{timestamp}" for root in candidate_roots]
    candidate_lines = "\n".join(candidate for candidate in candidates)

    script = f"""
set -e

while IFS= read -r candidate; do
  if [ -d "$candidate" ]; then
    echo "$candidate"
    exit 0
  fi
done <<'EOF'
{candidate_lines}
EOF

echo 'Error: none of the expected timestamp directories exists:' >&2
while IFS= read -r candidate; do
  echo "  $candidate" >&2
done <<'EOF'
{candidate_lines}
EOF

exit 1
"""

    result = docker_exec_bash(
        cfg.perception_docker_dir,
        cfg.perception_service,
        script,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def resolve_masks_timestamp_root(cfg: Config, timestamp: str) -> str:
    return resolve_existing_timestamp_root(
        cfg,
        [
            cfg.masks_by_object_root,
            cfg.masks_by_objects_root,
        ],
        timestamp,
    )


def resolve_outputs_timestamp_root(cfg: Config, timestamp: str) -> str:
    return resolve_existing_timestamp_root(
        cfg,
        [
            cfg.outputs_by_object_root,
            cfg.outputs_by_objects_root,
        ],
        timestamp,
    )


def list_object_names_from_timestamp_root(
    cfg: Config,
    timestamp_root: str,
) -> list[str]:
    script = f"""
set -e

root={shlex.quote(timestamp_root)}

find "$root" \\
  -mindepth 1 \\
  -maxdepth 1 \\
  -type d \\
  -printf '%f\\n' \\
| sort
"""

    result = docker_exec_bash(
        cfg.perception_docker_dir,
        cfg.perception_service,
        script,
        capture_output=True,
        text=True,
    )

    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def get_sam3_object_names(cfg: Config, timestamp: str) -> list[str]:
    masks_root = resolve_masks_timestamp_root(cfg, timestamp)
    return list_object_names_from_timestamp_root(cfg, masks_root)


def get_foundationpose_object_names(cfg: Config, timestamp: str) -> list[str]:
    outputs_root = resolve_outputs_timestamp_root(cfg, timestamp)
    return list_object_names_from_timestamp_root(cfg, outputs_root)


def get_sam3_mask_path(cfg: Config, timestamp: str, object_name: str) -> str:
    masks_root = resolve_masks_timestamp_root(cfg, timestamp)
    return f"{masks_root}/{object_name}/{cfg.image_id}.png"


def get_foundationpose_output_paths(
    cfg: Config,
    timestamp: str,
    object_name: str,
) -> tuple[str, str]:
    outputs_root = resolve_outputs_timestamp_root(cfg, timestamp)

    fp_vis_container = f"{outputs_root}/{object_name}/vis/{cfg.image_id}.png"
    fp_pose_container = f"{outputs_root}/{object_name}/ob_in_cam/{cfg.image_id}.txt"

    return fp_vis_container, fp_pose_container


def read_prompt_for_object(cfg: Config, timestamp: str, object_name: str) -> str:
    candidates = [
        f"{cfg.masks_by_object_root}/{timestamp}/{object_name}/prompt.txt",
        f"{cfg.masks_by_objects_root}/{timestamp}/{object_name}/prompt.txt",
        f"{cfg.outputs_by_object_root}/{timestamp}/{object_name}/prompt.txt",
        f"{cfg.outputs_by_objects_root}/{timestamp}/{object_name}/prompt.txt",
    ]

    candidate_lines = "\n".join(candidate for candidate in candidates)

    script = f"""
set -e

while IFS= read -r candidate; do
  if [ -f "$candidate" ]; then
    cat "$candidate"
    exit 0
  fi
done <<'EOF'
{candidate_lines}
EOF

exit 0
"""

    result = docker_exec_bash(
        cfg.perception_docker_dir,
        cfg.perception_service,
        script,
        capture_output=True,
        text=True,
    )

    prompt = result.stdout.strip()

    if prompt:
        return prompt

    return object_name.replace("_", " ")


def load_mesh_assignments(cfg: Config, timestamp: str) -> dict[str, Any]:
    outputs_root = resolve_outputs_timestamp_root(cfg, timestamp)
    assignments_path = f"{outputs_root}/mesh_assignments.json"

    data = read_json_from_perception(
        cfg,
        assignments_path,
        required=False,
    )

    if not data:
        print()
        print("[mesh] Warning: mesh_assignments.json not found.")
        print(f"[mesh] Expected: {assignments_path}")
        print("[mesh] Object actions can still be selected, but grasp box_width may be unavailable.")

    return data


def load_mesh_metadata(cfg: Config) -> dict[str, Any]:
    data = read_json_from_perception(
        cfg,
        cfg.mesh_metadata_file,
        required=False,
    )

    if not data:
        print()
        print("[mesh] Warning: mesh_metadata.json not found.")
        print(f"[mesh] Expected: {cfg.mesh_metadata_file}")

    return data


def mesh_metadata_for_id(
    mesh_metadata: dict[str, Any],
    mesh_id: Optional[str],
) -> dict[str, Any]:
    if not mesh_id:
        return {}

    meshes = mesh_metadata.get("meshes", {})

    if not isinstance(meshes, dict):
        return {}

    item = meshes.get(mesh_id, {})

    if not isinstance(item, dict):
        return {}

    return item


def compute_grasp_box_width(
    cfg: Config,
    mesh_id: Optional[str],
    metadata: dict[str, Any],
) -> float:
    robot_params = metadata.get("robot_params", {})

    if not isinstance(robot_params, dict):
        robot_params = {}

    dimensions = metadata.get("dimensions_m", {})

    if not isinstance(dimensions, dict):
        dimensions = {}

    grasp = metadata.get("grasp", {})

    if not isinstance(grasp, dict):
        grasp = {}

    direct_value = robot_params.get("grasp_box_width")

    if direct_value is None:
        direct_value = robot_params.get("box_width")

    if direct_value is not None:
        return float(direct_value)

    width = dimensions.get("width")

    if width is None:
        raise RuntimeError(
            "Cannot compute grasp box_width because mesh metadata does not contain "
            f"dimensions_m.width for mesh_id={mesh_id!r}."
        )

    extra = robot_params.get("grasp_box_width_extra")

    if extra is None:
        extra = robot_params.get("box_width_extra_m")

    if extra is None:
        extra = grasp.get("box_width_extra_m")

    if extra is None:
        extra = cfg.default_grasp_box_width_extra_m

    return float(width) + float(extra)


def enrich_object_results_with_mesh_info(
    cfg: Config,
    timestamp: str,
    object_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    assignments = load_mesh_assignments(cfg, timestamp)
    metadata = load_mesh_metadata(cfg)

    assignment_objects = assignments.get("objects", {})

    if not isinstance(assignment_objects, dict):
        assignment_objects = {}

    print()
    print("============================================================")
    print("[mesh] Object -> mesh assignments")
    print("============================================================")

    for result in object_results:
        object_name = result["object_name"]

        assignment = assignment_objects.get(object_name, {})

        if not isinstance(assignment, dict):
            assignment = {}

        mesh_id = assignment.get("mesh_id")
        mesh_relative_path = assignment.get("mesh_relative_path")
        mesh_path = assignment.get("mesh_path")
        mesh_item_metadata = mesh_metadata_for_id(metadata, mesh_id)

        result["mesh_assignment"] = assignment
        result["mesh_id"] = mesh_id
        result["mesh_relative_path"] = mesh_relative_path
        result["mesh_path"] = mesh_path
        result["mesh_metadata"] = mesh_item_metadata

        try:
            result["grasp_box_width"] = compute_grasp_box_width(
                cfg,
                mesh_id,
                mesh_item_metadata,
            )
        except Exception:
            result["grasp_box_width"] = None

        pose = result["pose"]
        mesh_display = mesh_id or "unknown_mesh"
        width_display = (
            f"{result['grasp_box_width']:.3f}"
            if result["grasp_box_width"] is not None
            else "n/a"
        )

        print(
            f"  - {object_name} -> mesh {mesh_display} "
            f"-> grasp_box_width={width_display} "
            f"-> {pose.robot_object_position_line}"
        )

        if mesh_relative_path:
            print(f"    mesh path: {mesh_relative_path}")

    print("============================================================")
    print()

    return object_results


def _conda_setup_block() -> str:
    return r"""
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
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

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
"""


def run_vlm_scene_perception(cfg: Config) -> None:
    print("[3/8] Running VLM scene perception...")

    script = f"""
set -e

{_conda_setup_block()}

conda activate vlm

python -u /workspace/eurobin/src/run_validation_live1.py \\
  --scenario {shlex.quote(cfg.vlm_scenario)} \\
  --scene-v {shlex.quote(cfg.vlm_scene_v)} \\
  --scene-model {shlex.quote(cfg.vlm_scene_model)}

conda deactivate
"""

    docker_exec_bash(cfg.perception_docker_dir, cfg.perception_service, script)


def resolve_vlm_objects_file(cfg: Config, timestamp: str) -> tuple[str, str]:
    script = f"""
set -e

base={shlex.quote(cfg.objects_root)}
preferred="$base/{timestamp}"

if [ -d "$preferred" ]; then
  objects_timestamp="{timestamp}"
  objects_dir="$preferred"
elif [ -f "$preferred" ]; then
  echo "{timestamp}"
  echo "$preferred"
  exit 0
else
  objects_timestamp=$(
    find "$base" \\
      -mindepth 1 \\
      -maxdepth 1 \\
      -type d \\
      -printf '%f\\n' \\
    | grep -E '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}_[0-9]{{2}}-[0-9]{{2}}-[0-9]{{2}}$' \\
    | sort \\
    | tail -n 1
  )

  if [ -z "$objects_timestamp" ]; then
    echo 'Error: no VLM objects timestamp found in:' "$base" >&2
    exit 1
  fi

  objects_dir="$base/$objects_timestamp"
fi

objects_file=$(
  find "$objects_dir" \\
    -maxdepth 2 \\
    -type f \\
    | sort \\
    | head -n 1
)

if [ -z "$objects_file" ]; then
  echo 'Error: no objects file found in:' "$objects_dir" >&2
  exit 1
fi

echo "$objects_timestamp"
echo "$objects_file"
"""

    result = docker_exec_bash(
        cfg.perception_docker_dir,
        cfg.perception_service,
        script,
        capture_output=True,
        text=True,
    )

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    if len(lines) < 2:
        raise RuntimeError(f"Could not resolve VLM objects file. Output was:\n{result.stdout}")

    return lines[0], lines[1]


def read_detected_objects_from_vlm(cfg: Config, timestamp: str) -> list[str]:
    objects_timestamp, objects_container_path = resolve_vlm_objects_file(cfg, timestamp)

    if objects_timestamp != timestamp:
        print()
        print("[warning] VLM objects timestamp differs from RGB-D timestamp.")
        print(f"          RGB-D timestamp: {timestamp}")
        print(f"          VLM timestamp:   {objects_timestamp}")

    objects_host_path = cfg.preview_dir / f"{timestamp}_vlm_objects.txt"

    copy_text_from_perception(cfg, objects_container_path, objects_host_path)

    objects = parse_detected_objects_text(objects_host_path.read_text())

    print()
    print("[vlm] Objects file:")
    print(f"      {objects_container_path}")
    print("[vlm] Copied objects file:")
    print(f"      {objects_host_path}")
    print("[vlm] Parsed objects:")

    for idx, obj in enumerate(objects, start=1):
        print(f"      [{idx}] {obj}")

    return objects


def set_align_depth_parameter(cfg: Config) -> None:
    print("[1/8] Setting align_depth.enable...")

    script = f"""
{ROBOT_SETUP}

echo 'Waiting for node /head_cam to become available...'

for i in $(seq 1 30); do
  if ros2 node list | grep -qx '/head_cam'; then
    echo 'Node /head_cam found.'
    break
  fi

  if [ "$i" -eq 30 ]; then
    echo 'Error: node /head_cam not found after 30 seconds.'
    echo 'Available ROS nodes:'
    ros2 node list || true
    exit 1
  fi

  sleep 1
done

ros2 param set /head_cam align_depth.enable true
"""

    docker_exec_bash(cfg.robot_docker_dir, cfg.robot_service, script)


def capture_rgbd_frames(cfg: Config) -> None:
    print(f"[2/8] Capturing {cfg.max_frames} RGB-D frame(s)...")

    script = f"""
{ROBOT_SETUP}

set -e

python3 -u ~/PoseEstimation/pipeline/save_realsense_rgbd.py \\
  --save_dir ~/shared_data \\
  --capture_hz {cfg.capture_hz} \\
  --resize_width {cfg.resize_width} \\
  --resize_height {cfg.resize_height} \\
  --max_frames {cfg.max_frames}
"""

    docker_exec_bash(cfg.robot_docker_dir, cfg.robot_service, script)


def get_latest_timestamp(cfg: Config) -> str:
    print("[timestamp] Retrieving the latest timestamp from rgb/ ...")

    script = f"""
set -e

find {shlex.quote(cfg.data_root)}/rgb \\
  -mindepth 1 \\
  -maxdepth 1 \\
  -type d \\
  -printf '%f\\n' \\
| grep -E '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}_[0-9]{{2}}-[0-9]{{2}}-[0-9]{{2}}$' \\
| sort \\
| tail -n 1
"""

    result = docker_exec_bash(
        cfg.perception_docker_dir,
        cfg.perception_service,
        script,
        capture_output=True,
        text=True,
    )

    timestamp = result.stdout.strip().replace("\r", "")

    if not timestamp:
        raise RuntimeError(f"Error: no timestamp found in {cfg.data_root}/rgb")

    if not re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", timestamp):
        raise RuntimeError(f"Invalid timestamp returned: {timestamp}")

    print(f"Selected timestamp: {timestamp}")
    return timestamp


def verify_input_data(cfg: Config, timestamp: str) -> None:
    print("[check] Verifying RGB, depth, and camera data...")

    script = f"""
set -e

test -d {shlex.quote(f"{cfg.data_root}/rgb/{timestamp}")} || {{
  echo 'Error: missing {cfg.data_root}/rgb/{timestamp}'
  exit 1
}}

test -d {shlex.quote(f"{cfg.data_root}/depth/{timestamp}")} || {{
  echo 'Error: missing {cfg.data_root}/depth/{timestamp}'
  exit 1
}}

test -d {shlex.quote(f"{cfg.data_root}/camera/{timestamp}")} || {{
  echo 'Error: missing {cfg.data_root}/camera/{timestamp}'
  exit 1
}}

test -f {shlex.quote(f"{cfg.data_root}/rgb/{timestamp}/{cfg.image_id}.png")} || {{
  echo 'Error: missing RGB image {cfg.data_root}/rgb/{timestamp}/{cfg.image_id}.png'
  exit 1
}}

echo 'RGB directory:'
ls -lah {shlex.quote(f"{cfg.data_root}/rgb/{timestamp}")} | head

echo 'Depth directory:'
ls -lah {shlex.quote(f"{cfg.data_root}/depth/{timestamp}")} | head

echo 'Camera directory:'
ls -lah {shlex.quote(f"{cfg.data_root}/camera/{timestamp}")} | head
"""

    docker_exec_bash(cfg.perception_docker_dir, cfg.perception_service, script)


def run_sam3_all_objects(cfg: Config, timestamp: str) -> None:
    print("[4/8] Running SAM3 multi-object segmentation...")

    script = f"""
set -e

{_conda_setup_block()}

conda activate sam3

python -u /workspace/PoseEstimation/pipeline/sam_script_fp.py \\
  --data_root {shlex.quote(cfg.data_root)} \\
  --timestamp {shlex.quote(timestamp)} \\
  --image_id {shlex.quote(cfg.image_id)}

conda deactivate
"""

    docker_exec_bash(cfg.perception_docker_dir, cfg.perception_service, script)


def run_foundationpose_all_objects(cfg: Config, timestamp: str) -> None:
    print("[5/8] Running FoundationPose multi-object pose estimation...")

    script = f"""
set -e

{_conda_setup_block()}

conda activate my

echo 'Active environment for FoundationPose:'
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"

echo 'Python interpreter for FoundationPose:'
which python

echo 'Checking cv2 import...'
python -u -c 'import cv2; print("cv2 OK", cv2.__version__)'

python -u /workspace/PoseEstimation/pipeline/run_fp_single_frame.py \\
  --data_root {shlex.quote(cfg.data_root)} \\
  --timestamp {shlex.quote(timestamp)} \\
  --start_frame {shlex.quote(cfg.image_id)} \\
  --max_frames {cfg.foundationpose_max_frames} \\
  --est_refine_iter {cfg.est_refine_iter} \\
  --track_refine_iter {cfg.track_refine_iter} \\
  --debug {cfg.foundationpose_debug}

conda deactivate
"""

    docker_exec_bash(cfg.perception_docker_dir, cfg.perception_service, script)


def write_object_pose_to_robot(cfg: Config, pose: PoseResult) -> None:
    p1 = f"{pose.robot_pos_1:.6f}"
    p2 = f"{pose.robot_pos_2:.6f}"
    p3 = f"{pose.robot_pos_3:.6f}"

    print()
    print("[robot] Writing object.position inside robot_docker...")
    print("[robot] Target file:")
    print(f"        {cfg.robot_object_pose_file}")
    print("[robot] New content:")
    print(f"        object.position: {p1} {p2} {p3}")

    script = f"""
set -e

target_file={shlex.quote(cfg.robot_object_pose_file)}
target_dir=$(dirname "$target_file")

test -d "$target_dir" || {{
  echo 'Error: directory not found:' "$target_dir" >&2
  exit 1
}}

printf 'object.position: %s %s %s\\n' {shlex.quote(p1)} {shlex.quote(p2)} {shlex.quote(p3)} > "$target_file"

echo ''
echo 'Updated object_pose.txt content:'
cat "$target_file"
"""

    docker_exec_bash(cfg.robot_docker_dir, cfg.robot_service, script)


def run_grasp_centauro(cfg: Config, box_width: float) -> None:
    print()
    print("[robot] Launching grasp_centauro_test_world.py inside robot_docker...")
    print("[robot] Working directory:")
    print(f"        {cfg.robot_manipulation_dir}")
    print("[robot] Script:")
    print(f"        {cfg.robot_grasp_script}")
    print("[robot] box_width:")
    print(f"        {box_width:.6f}")

    script = f"""
set -e

{ROBOT_SETUP}

cd {shlex.quote(cfg.robot_manipulation_dir)}

test -f {shlex.quote(cfg.robot_object_pose_file)} || {{
  echo 'Error: object_pose.txt not found: {cfg.robot_object_pose_file}' >&2
  exit 1
}}

test -f {shlex.quote(cfg.robot_grasp_script)} || {{
  echo 'Error: grasp script not found: {cfg.robot_grasp_script}' >&2
  exit 1
}}

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

echo ''
echo 'Environment used for grasp execution:'
echo "USER=$(whoami)"
echo "PWD=$(pwd)"
echo "CONDA_DEFAULT_ENV=${{CONDA_DEFAULT_ENV:-}}"
echo "VIRTUAL_ENV=${{VIRTUAL_ENV:-}}"
echo "PYTHONPATH=${{PYTHONPATH:-}}"
echo "AMENT_PREFIX_PATH=${{AMENT_PREFIX_PATH:-}}"
echo ''
echo 'Python interpreter:'
which python3
python3 -u -c 'import sys; print(sys.executable)'

echo ''
echo 'Checking cartesian_interface_ros import...'
python3 -u -c 'import cartesian_interface_ros; print("cartesian_interface_ros OK")'

echo ''
echo 'object_pose.txt used for grasp execution:'
cat {shlex.quote(cfg.robot_object_pose_file)}
echo ''

python3 -u {shlex.quote(cfg.robot_grasp_script)} --ros-args \\
  -p object_pose_file:=object_pose.txt \\
  -p base_frame:=world \\
  -p cartesian_world_frame:=ci/world \\
  -p cartesian_robot_base_frame:=ci/pelvis \\
  -p robot_base_frame:=pelvis \\
  -p camera_frame:=D435_head_camera_link \\
  -p approach_mode:=yz \\
  -p grasp_offset_x:=0.12 \\
  -p grasp_offset_y:=-0.03 \\
  -p grasp_offset_z:=0.02 \\
  -p phase1_split_fraction:=0.5 \\
  -p constrain_orientation:=true \\
  -p box_width:={box_width:.6f}
"""

    docker_exec_bash(
        cfg.robot_docker_dir,
        cfg.robot_service,
        script,
        interactive=True,
        allocate_tty=True,
    )


def run_push_box_xyz(cfg: Config) -> None:
    print()
    print("[robot] Launching push_box_xyz.py inside robot_docker...")
    print("[robot] Working directory:")
    print(f"        {cfg.robot_manipulation_dir}")
    print("[robot] Script:")
    print(f"        {cfg.robot_push_script}")

    script = f"""
set -e

{ROBOT_SETUP}

cd {shlex.quote(cfg.robot_manipulation_dir)}

test -f {shlex.quote(cfg.robot_object_pose_file)} || {{
  echo 'Error: object_pose.txt not found: {cfg.robot_object_pose_file}' >&2
  exit 1
}}

test -f {shlex.quote(cfg.robot_push_script)} || {{
  echo 'Error: push script not found: {cfg.robot_push_script}' >&2
  exit 1
}}

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

echo ''
echo 'Environment used for push execution:'
echo "USER=$(whoami)"
echo "PWD=$(pwd)"
echo "CONDA_DEFAULT_ENV=${{CONDA_DEFAULT_ENV:-}}"
echo "VIRTUAL_ENV=${{VIRTUAL_ENV:-}}"
echo "PYTHONPATH=${{PYTHONPATH:-}}"
echo "AMENT_PREFIX_PATH=${{AMENT_PREFIX_PATH:-}}"
echo ''
echo 'Python interpreter:'
which python3
python3 -u -c 'import sys; print(sys.executable)'

echo ''
echo 'Checking cartesian_interface_ros import...'
python3 -u -c 'import cartesian_interface_ros; print("cartesian_interface_ros OK")'

echo ''
echo 'object_pose.txt used for push execution:'
cat {shlex.quote(cfg.robot_object_pose_file)}
echo ''

python3 -u {shlex.quote(cfg.robot_push_script)} --ros-args \\
  -p object_pose_file:=object_pose.txt \\
  -p base_frame:=world \\
  -p cartesian_world_frame:=ci/world \\
  -p cartesian_robot_base_frame:=ci/pelvis \\
  -p robot_base_frame:=pelvis \\
  -p camera_frame:=D435_head_camera_link \\
  -p push_offset_x:=0.0 \\
  -p push_offset_y:=-0.0 \\
  -p push_offset_z:=0.02 \\
  -p push_direction_x:=1.0 \\
  -p push_direction_y:=0.0 \\
  -p push_direction_z:=0.0 \\
  -p pre_push_distance:=0.08 \\
  -p push_distance:=0.15 \\
  -p retreat_distance:=0.0 \\
  -p constrain_orientation:=true \\
  -p push_dagana:=2
"""

    docker_exec_bash(
        cfg.robot_docker_dir,
        cfg.robot_service,
        script,
        interactive=True,
        allocate_tty=True,
    )


def print_available_object_actions(object_results: list[dict[str, Any]]) -> None:
    print()
    print("============================================================")
    print("Available object poses:")
    print("============================================================")

    for idx, result in enumerate(object_results, start=1):
        pose = result["pose"]
        object_name = result["object_name"]
        prompt = result.get("prompt", object_name)
        mesh_id = result.get("mesh_id") or "unknown_mesh"
        grasp_box_width = result.get("grasp_box_width")

        if grasp_box_width is None:
            width_text = "grasp_box_width=n/a"
        else:
            width_text = f"grasp_box_width={grasp_box_width:.3f}"

        print(
            f"  [{idx}] {object_name} -> mesh {mesh_id} -> "
            f"{pose.robot_object_position_line} -> {width_text}"
        )
        print(f"      prompt: {prompt}")

    print("============================================================")


def wait_user_before_manual_capture() -> None:
    print()
    print("============================================================")
    print("[manual-action] Execute the manual robot manipulation now.")
    print("[manual-action] When the action is finished, type 'y' and press Enter.")
    print("============================================================")

    while True:
        answer = input("\nCapture a new RGB-D frame now and continue live2? [y/N]: ").strip()

        if is_yes(answer):
            print("[manual-action] Continuing with new frame capture...")
            return

        print("[manual-action] Not capturing yet. Finish the manual action, then type 'y'.")


def select_object_result(
    object_results: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not object_results:
        print()
        print("[action] No FoundationPose object results available.")
        return None

    print_available_object_actions(object_results)

    answer = input(
        "\nSelect object index for robot action "
        "(Enter for manual action without automatic robot command): "
    ).strip()

    if not answer:
        return None

    try:
        selected_idx = int(answer)
    except ValueError:
        raise RuntimeError(f"Invalid object selection: {answer}")

    if selected_idx < 1 or selected_idx > len(object_results):
        raise RuntimeError(f"Object selection out of range: {selected_idx}")

    selected = object_results[selected_idx - 1]

    print()
    print(f"[action] Selected object: {selected['object_name']}")
    print(f"[action] Mesh:            {selected.get('mesh_id') or 'unknown_mesh'}")
    print(f"[action] Pose:            {selected['pose'].robot_object_position_line}")

    return selected


def select_action_for_object() -> str:
    print()
    print("Select action:")
    print("  [1] grasp")
    print("  [2] push")
    print("  [Enter] manual / no automatic robot command")

    answer = input("Action: ").strip()

    if not answer:
        return "manual"

    if answer == "1" or answer.lower() == "grasp":
        return "grasp"

    if answer == "2" or answer.lower() == "push":
        return "push"

    raise RuntimeError(f"Invalid action selection: {answer}")


def execute_selected_robot_action(
    cfg: Config,
    object_results: list[dict[str, Any]],
) -> None:
    print()
    print("============================================================")
    print("[action] PRE condition has been evaluated as matching.")
    print("[action] live2 is waiting for a new Realsense frame.")
    print("[action] Select an automatic robot action, or use manual mode.")
    print("============================================================")

    selected = select_object_result(object_results)

    if selected is None:
        wait_user_before_manual_capture()
        return

    action = select_action_for_object()

    if action == "manual":
        wait_user_before_manual_capture()
        return

    pose = selected["pose"]

    write_object_pose_to_robot(cfg, pose)

    if action == "grasp":
        grasp_box_width = selected.get("grasp_box_width")

        if grasp_box_width is None:
            raise RuntimeError(
                f"Cannot execute grasp for {selected['object_name']}: "
                "grasp_box_width is not available. Check mesh_metadata.json."
            )

        run_grasp_centauro(cfg, box_width=float(grasp_box_width))

        print()
        print("============================================================")
        print("[action] Grasp action completed.")
        print("[action] The pipeline will NOT capture the next RGB-D frame automatically.")
        print("[action] Inspect the robot/scene, then decide when to continue.")
        print("============================================================")

        wait_user_before_manual_capture()
        return

    if action == "push":
        run_push_box_xyz(cfg)

        print()
        print("============================================================")
        print("[action] Push action completed.")
        print("[action] The pipeline will NOT capture the next RGB-D frame automatically.")
        print("[action] Inspect the robot/scene, then decide when to continue.")
        print("============================================================")

        wait_user_before_manual_capture()
        return

    raise RuntimeError(f"Unsupported action: {action}")


def run_vlm_task_validation_with_auto_capture(
    cfg: Config,
    object_results: list[dict[str, Any]],
) -> VLMTaskValidationResult:
    print("[6/8] Running VLM task planning, simulation, and validation...")
    print("[6/8] Interactive robot action execution is enabled.")
    print("[6/8] When live2 waits for a new Realsense frame, this pipeline will pause.")
    print("[6/8] You can select grasp, push, or manual mode.")

    scene_enrichment_debug_flag = (
        "  --grounding-debug-mapping \\\n"
        if cfg.scene_enrichment_debug
        else ""
    )

    script = f"""
set -e

{_conda_setup_block()}

conda activate vlm

python -u /workspace/eurobin/src/run_validation_live2.py \\
  --scenario {shlex.quote(cfg.vlm_scenario)} \\
  --task {shlex.quote(cfg.vlm_task)} \\
  --scene-v {shlex.quote(cfg.vlm_scene_v)} \\
  --plan-v {shlex.quote(cfg.vlm_plan_v)} \\
  --sim-v {shlex.quote(cfg.vlm_sim_v)} \\
  --validator-v {shlex.quote(cfg.vlm_validator_v)} \\
  --scene-model {shlex.quote(cfg.vlm_scene_model)} \\
  --plan-model {shlex.quote(cfg.vlm_plan_model)} \\
  --sim-model {shlex.quote(cfg.vlm_sim_model)} \\
  --validator-model {shlex.quote(cfg.vlm_validator_model)} \\
{scene_enrichment_debug_flag}  --new-frame-timeout {cfg.vlm_new_frame_timeout} \\
  --new-frame-poll-seconds {cfg.vlm_new_frame_poll_seconds}

conda deactivate
"""

    capture_trigger_pattern = "Waiting for a new Realsense frame newer than"
    capture_in_progress = False
    capture_count = 0

    task_completed: Optional[bool] = None
    outcome: Optional[str] = None
    saw_non_matching = False

    def on_live2_line(line: str) -> None:
        nonlocal capture_in_progress
        nonlocal capture_count
        nonlocal task_completed
        nonlocal outcome
        nonlocal saw_non_matching

        stripped = line.strip()

        if (
            "[LOOP] PRE result:" in stripped
            or "[LOOP] POST result:" in stripped
        ) and "non_matching" in stripped:
            saw_non_matching = True

        if stripped.startswith("Task completed:"):
            value = stripped.split(":", 1)[1].strip().lower()
            task_completed = value == "true"

        if stripped.startswith("Outcome:"):
            outcome = stripped.split(":", 1)[1].strip()

        if capture_trigger_pattern not in line:
            return

        if capture_in_progress:
            return

        capture_in_progress = True
        capture_count += 1

        print()
        print("============================================================")
        print(f"[action] live2 requested a new Realsense frame #{capture_count}.")
        print("[action] The pipeline is paused before capturing.")
        print("============================================================")

        try:
            execute_selected_robot_action(
                cfg=cfg,
                object_results=object_results,
            )

            print()
            print("============================================================")
            print(f"[auto-capture] Capturing RGB-D frame #{capture_count} from robot_docker...")
            print("============================================================")

            capture_rgbd_frames(cfg)

            latest_timestamp = get_latest_timestamp(cfg)

            print()
            print("============================================================")
            print("[auto-capture] New RGB-D frame captured.")
            print(f"[auto-capture] Latest Realsense timestamp: {latest_timestamp}")
            print("[auto-capture] Opening captured RGB image preview...")
            print("============================================================")
            print()

            captured_rgb_container = (
                f"{cfg.data_root}/rgb/{latest_timestamp}/{cfg.image_id}.png"
            )
            captured_rgb_host = (
                cfg.preview_dir
                / f"{latest_timestamp}_{cfg.image_id}_auto_capture_after_robot_action.png"
            )

            preview_image_from_perception(
                cfg,
                captured_rgb_container,
                captured_rgb_host,
                "RGB image captured after robot action",
            )

            print()
            print("============================================================")
            print("[auto-capture] Preview opened.")
            print("[auto-capture] live2 should detect the new frame and continue.")
            print("============================================================")
            print()

        finally:
            capture_in_progress = False

    docker_exec_bash_streaming(
        cfg.perception_docker_dir,
        cfg.perception_service,
        script,
        on_stdout_line=on_live2_line,
    )

    final_task_completed = bool(task_completed)
    final_outcome = outcome or "unknown"

    restartable_outcome = (
        final_outcome.startswith("stopped_replan_on_pre_stage_")
        or final_outcome.startswith("stopped_replan_on_post_stage_")
    )

    cycle_error = final_outcome.startswith("cycle_error:")

    should_restart = (
        not cycle_error
        and not final_task_completed
        and (
            saw_non_matching
            or restartable_outcome
        )
    )

    return VLMTaskValidationResult(
        task_completed=final_task_completed,
        outcome=final_outcome,
        saw_non_matching=saw_non_matching,
        should_restart=should_restart,
    )