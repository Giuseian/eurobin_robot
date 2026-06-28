#!/usr/bin/env python3

from __future__ import annotations

from run_box_pipeline_utils import (
    Config,
    ask_continue_or_stop,
    capture_rgbd_frames,
    enrich_object_results_with_mesh_info,
    get_foundationpose_object_names,
    get_foundationpose_output_paths,
    get_latest_timestamp,
    get_sam3_mask_path,
    get_sam3_object_names,
    preview_image_from_perception,
    read_and_print_object_position_from_pose_file,
    read_detected_objects_from_vlm,
    read_prompt_for_object,
    run_foundationpose_all_objects,
    run_sam3_all_objects,
    run_vlm_scene_perception,
    run_vlm_task_validation_with_auto_capture,
    set_align_depth_parameter,
    verify_input_data,
)


def run_single_perception_validation_cycle(cfg: Config, cycle_idx: int) -> bool:
    print()
    print("############################################################")
    print(f"# PIPELINE CYCLE {cycle_idx}/{cfg.max_pipeline_cycles}")
    print("############################################################")
    print()

    # ------------------------------------------------------------
    # 1. RGB-D acquisition
    # ------------------------------------------------------------
    capture_rgbd_frames(cfg)

    timestamp = get_latest_timestamp(cfg)
    verify_input_data(cfg, timestamp)

    rgb_image_container = f"{cfg.data_root}/rgb/{timestamp}/{cfg.image_id}.png"
    rgb_image_host = cfg.preview_dir / f"{timestamp}_{cfg.image_id}_cycle_{cycle_idx}_rgb.png"

    preview_image_from_perception(
        cfg,
        rgb_image_container,
        rgb_image_host,
        f"Cycle {cycle_idx} - captured RGB image",
    )

    # ------------------------------------------------------------
    # 2. VLM scene perception
    # ------------------------------------------------------------
    ask_continue_or_stop(f"[cycle {cycle_idx}] Continue with VLM scene perception?")

    run_vlm_scene_perception(cfg)

    detected_objects = read_detected_objects_from_vlm(cfg, timestamp)

    if not detected_objects:
        raise RuntimeError("No objects were detected by the VLM scene perception step.")

    print()
    print(f"[cycle {cycle_idx}] Detected objects:")
    for idx, obj in enumerate(detected_objects, start=1):
        print(f"  [{idx}] {obj}")

    # ------------------------------------------------------------
    # 3. SAM3 multi-object segmentation
    # ------------------------------------------------------------
    ask_continue_or_stop(f"[cycle {cycle_idx}] Continue with SAM3 multi-object segmentation?")

    run_sam3_all_objects(cfg, timestamp)

    sam3_object_names = get_sam3_object_names(cfg, timestamp)

    if not sam3_object_names:
        raise RuntimeError("SAM3 completed, but no object masks were found.")

    print()
    print(f"[cycle {cycle_idx}] SAM3 masks generated for:")
    for idx, object_name in enumerate(sam3_object_names, start=1):
        prompt = read_prompt_for_object(cfg, timestamp, object_name)
        print(f"  [{idx}] {prompt}  ({object_name})")

    for object_name in sam3_object_names:
        prompt = read_prompt_for_object(cfg, timestamp, object_name)
        mask_container = get_sam3_mask_path(cfg, timestamp, object_name)
        mask_host = (
            cfg.preview_dir
            / f"{timestamp}_{object_name}_{cfg.image_id}_cycle_{cycle_idx}_mask.png"
        )

        preview_image_from_perception(
            cfg,
            mask_container,
            mask_host,
            f"Cycle {cycle_idx} - SAM3 mask for object: {prompt}",
        )

    ask_continue_or_stop(f"[cycle {cycle_idx}] Are the SAM3 masks acceptable?")

    # ------------------------------------------------------------
    # 4. FoundationPose multi-object pose estimation
    # ------------------------------------------------------------
    run_foundationpose_all_objects(cfg, timestamp)

    fp_object_names = get_foundationpose_object_names(cfg, timestamp)

    if not fp_object_names:
        raise RuntimeError("FoundationPose completed, but no object outputs were found.")

    print()
    print(f"[cycle {cycle_idx}] FoundationPose outputs generated for:")
    for idx, object_name in enumerate(fp_object_names, start=1):
        prompt = read_prompt_for_object(cfg, timestamp, object_name)
        print(f"  [{idx}] {prompt}  ({object_name})")

    # ------------------------------------------------------------
    # 5. Preview and parse all FoundationPose outputs
    # ------------------------------------------------------------
    object_results = []

    for idx, object_name in enumerate(fp_object_names, start=1):
        prompt = read_prompt_for_object(cfg, timestamp, object_name)

        print()
        print("============================================================")
        print(f" Cycle {cycle_idx} - FoundationPose result {idx}/{len(fp_object_names)}: {prompt}")
        print("============================================================")

        fp_vis_container, fp_pose_container = get_foundationpose_output_paths(
            cfg,
            timestamp,
            object_name,
        )

        fp_vis_host = (
            cfg.preview_dir
            / f"{timestamp}_{object_name}_{cfg.image_id}_cycle_{cycle_idx}_foundationpose_vis.png"
        )
        fp_pose_host = (
            cfg.preview_dir
            / f"{timestamp}_{object_name}_{cfg.image_id}_cycle_{cycle_idx}_ob_in_cam.txt"
        )

        preview_image_from_perception(
            cfg,
            fp_vis_container,
            fp_vis_host,
            f"Cycle {cycle_idx} - FoundationPose output for object: {prompt}",
        )

        pose = read_and_print_object_position_from_pose_file(
            cfg,
            fp_pose_container,
            fp_pose_host,
        )

        object_results.append(
            {
                "prompt": prompt,
                "object_name": object_name,
                "fp_vis_container": fp_vis_container,
                "fp_vis_host": fp_vis_host,
                "fp_pose_container": fp_pose_container,
                "fp_pose_host": fp_pose_host,
                "pose": pose,
            }
        )

    object_results = enrich_object_results_with_mesh_info(
        cfg=cfg,
        timestamp=timestamp,
        object_results=object_results,
    )

    ask_continue_or_stop(f"[cycle {cycle_idx}] Are the FoundationPose outputs acceptable?")

    # ------------------------------------------------------------
    # 6. VLM task planning / simulation / validation
    # ------------------------------------------------------------
    ask_continue_or_stop(
        f"[cycle {cycle_idx}] Continue with VLM task planning and validation "
        "with interactive robot action execution?"
    )

    validation_result = run_vlm_task_validation_with_auto_capture(
        cfg=cfg,
        object_results=object_results,
    )

    print()
    print("============================================================")
    print(f"[cycle {cycle_idx}] live2 result")
    print("============================================================")
    print(f"task_completed:   {validation_result.task_completed}")
    print(f"outcome:          {validation_result.outcome}")
    print(f"saw_non_matching: {validation_result.saw_non_matching}")
    print(f"should_restart:   {validation_result.should_restart}")
    print("============================================================")
    print()

    if validation_result.should_restart:
        print()
        print("============================================================")
        print("[pipeline] live2 returned non_matching / stopped_replan.")
        print("[pipeline] Restarting the full pipeline from RGB-D acquisition.")
        print("============================================================")
        print()
        return False

    if validation_result.task_completed:
        print()
        print("============================================================")
        print("[pipeline] live2 completed the task.")
        print("[pipeline] Pipeline finished successfully.")
        print("============================================================")
        print()
        return True

    raise RuntimeError(
        "live2 finished without task completion and without a restartable non_matching outcome. "
        f"Outcome: {validation_result.outcome}"
    )


def run_pipeline() -> int:
    cfg = Config()
    cfg.preview_dir.mkdir(parents=True, exist_ok=True)

    # align_depth is a robot/camera setup step, so we do it once before the loop.
    set_align_depth_parameter(cfg)

    for cycle_idx in range(1, cfg.max_pipeline_cycles + 1):
        completed = run_single_perception_validation_cycle(cfg, cycle_idx)

        if completed:
            return 0

    raise RuntimeError(
        f"Maximum number of pipeline cycles reached: {cfg.max_pipeline_cycles}. "
        "The task was not completed."
    )


if __name__ == "__main__":
    try:
        raise SystemExit(run_pipeline())
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        raise SystemExit(130)