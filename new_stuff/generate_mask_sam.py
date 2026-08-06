#!/usr/bin/env python3
# Genera la maschera di segmentazione con SAM (ViT-H) a partire dalla foto
# salvata da save_zed_rgbd.py, usando un punto al centro dell'immagine come
# prompt (di default). Va lanciato DENTRO al container SAM
# (./run_sam_container), non dentro al container Isaac ROS o ZED.
#
# Legge:  /docker_shared/<camera_name>/rgb/<run_timestamp>/<frame_id>.png
# Scrive: /docker_shared/<camera_name>/masks/<run_timestamp>/mask.png
#
# La maschera prodotta va poi passata a photo_mask_publisher.py con:
#   --mask_path /docker_shared/<camera_name>/masks/<run_timestamp>/mask.png

import argparse
import os

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="/docker_shared")
    parser.add_argument("--camera_name", type=str, default="zed")
    parser.add_argument(
        "--run_timestamp", type=str, required=True,
        help="Es. 2026-08-05_14-32-10. Deve essere lo stesso della foto da segmentare.",
    )
    parser.add_argument("--frame_id", type=int, default=0)
    parser.add_argument(
        "--checkpoint", type=str,
        default="/docker_shared/eurobin/sam/sam_vit_h_4b8939.pth",
    )
    parser.add_argument("--model_type", type=str, default="vit_h")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--point_x", type=int, default=None,
        help="Coordinata X del punto prompt. Default: centro immagine.",
    )
    parser.add_argument(
        "--point_y", type=int, default=None,
        help="Coordinata Y del punto prompt. Default: centro immagine.",
    )
    args = parser.parse_args()

    camera_dir = os.path.join(args.save_dir, args.camera_name)
    rgb_path = os.path.join(camera_dir, "rgb", args.run_timestamp, f"{args.frame_id:06d}.png")
    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(f"Foto non trovata: {rgb_path}")

    out_dir = os.path.join(camera_dir, "masks", args.run_timestamp)
    os.makedirs(out_dir, exist_ok=True)

    print("CUDA available:", torch.cuda.is_available())

    image = cv2.imread(rgb_path)
    if image is None:
        raise RuntimeError(f"Impossibile leggere {rgb_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    print(f"Image loaded: {w}x{h}")

    point_x = args.point_x if args.point_x is not None else w // 2
    point_y = args.point_y if args.point_y is not None else h // 2

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)
    print("Predictor initialized. Setting image...")
    predictor.set_image(image)

    input_point = np.array([[point_x, point_y]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    print("masks shape:", masks.shape)
    print("scores:", scores)

    for i, (mask, score) in enumerate(zip(masks, scores)):
        # overlay a colori (per controllo visivo delle 3 ipotesi generate da SAM)
        overlay = image.copy()
        color_mask = np.zeros_like(overlay)
        color_mask[mask] = [30, 144, 255]
        overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.6, 0)
        overlay_path = os.path.join(out_dir, f"overlay_{i}_score_{score:.3f}.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print("saved:", overlay_path)

    best_idx = int(np.argmax(scores))
    best_binary = np.where(masks[best_idx], 255, 0).astype(np.uint8)
    mask_path = os.path.join(out_dir, "mask.png")
    cv2.imwrite(mask_path, best_binary)
    print(f"saved mask (score {scores[best_idx]:.3f}):", mask_path)


if __name__ == "__main__":
    main()
