#!/usr/bin/env python3
# Disegna il box 3D stimato da FoundationPose (letto dal topic /output) sopra
# la foto RGB, leggendo foto e intrinseci direttamente dagli stessi file usati
# da photo_mask_publisher.py (non serve riprenderli via ROS). Salva il
# risultato su file (nessun bisogno di RViz2/X11).

import argparse
import os
from datetime import datetime

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray


def find_latest_run(camera_dir):
    rgb_root = os.path.join(camera_dir, "rgb")
    runs = sorted(os.listdir(rgb_root))
    if not runs:
        raise RuntimeError(f"Nessuna cartella trovata in {rgb_root}")
    return runs[-1]


def quat_to_rot(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


class PoseVisualizer(Node):
    def __init__(self, save_dir, camera_name, run_timestamp, frame_id,
                 output_topic, output_dir):
        super().__init__("pose_visualizer")

        camera_dir = os.path.join(save_dir, camera_name)
        if run_timestamp is None:
            run_timestamp = find_latest_run(camera_dir)
            self.get_logger().info(
                f"Nessun --run_timestamp specificato, uso l'ultimo trovato: {run_timestamp}"
            )

        rgb_path = os.path.join(camera_dir, "rgb", run_timestamp, f"{frame_id:06d}.png")
        intrinsics_path = os.path.join(camera_dir, "camera", run_timestamp, "intrinsics.yaml")

        for p in (rgb_path, intrinsics_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"File non trovato: {p}")

        self.img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if self.img is None:
            raise RuntimeError(f"Impossibile leggere {rgb_path}")

        with open(intrinsics_path) as f:
            intr = yaml.safe_load(f)
        K = np.array(intr["K"], dtype=np.float64)
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.done = False

        self.create_subscription(Detection3DArray, output_topic, self.detection_cb, 10)
        self.get_logger().info(f"Foto: {rgb_path} -- in attesa di una posa su '{output_topic}'...")

    def detection_cb(self, msg):
        if self.done or not msg.detections:
            return

        img = self.img.copy()
        det = msg.detections[0]
        pos = det.bbox.center.position
        ori = det.bbox.center.orientation
        size = det.bbox.size

        R = quat_to_rot(ori.x, ori.y, ori.z, ori.w)
        t = np.array([pos.x, pos.y, pos.z])
        hx, hy, hz = size.x / 2.0, size.y / 2.0, size.z / 2.0

        corners_local = np.array([
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
        ])

        corners_cam = (R @ corners_local.T).T + t

        points_2d = []
        for p in corners_cam:
            if p[2] <= 0.01:
                points_2d.append(None)
                continue
            u = self.fx * p[0] / p[2] + self.cx
            v = self.fy * p[1] / p[2] + self.cy
            points_2d.append((int(round(u)), int(round(v))))

        for i, j in CUBE_EDGES:
            if points_2d[i] is not None and points_2d[j] is not None:
                cv2.line(img, points_2d[i], points_2d[j], (0, 255, 0), 2)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = os.path.join(self.output_dir, f"{timestamp}.png")
        cv2.imwrite(out_path, img)
        self.get_logger().info(f"Salvato: {out_path}")
        self.done = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="/docker_shared")
    parser.add_argument("--camera_name", type=str, default="zed")
    parser.add_argument("--run_timestamp", type=str, default=None)
    parser.add_argument("--frame_id", type=int, default=0)
    parser.add_argument("--output_topic", type=str, default="output")
    parser.add_argument("--output_dir", type=str, default="/docker_shared/zed/debug/pose_viz")
    args = parser.parse_args()

    rclpy.init()
    node = PoseVisualizer(
        save_dir=os.path.expanduser(args.save_dir),
        camera_name=args.camera_name,
        run_timestamp=args.run_timestamp,
        frame_id=args.frame_id,
        output_topic=args.output_topic,
        output_dir=args.output_dir,
    )
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
