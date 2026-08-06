#!/usr/bin/env python3
# Ponte: legge una foto RGB-D gia' salvata (save_zed_rgbd.py / save_realsense_rgbd.py)
# e la ripubblica in loop sui topic che si aspetta il launch "foundationpose"
# di Isaac ROS (drop_node -> RT-DETR): image_rect, depth, camera_info_rect.
#
# Usa questo script per il percorso CON RT-DETR (rilevamento automatico).
# Per il percorso SENZA RT-DETR (maschera fornita a mano) usa invece
# photo_mask_publisher.py insieme a foundationpose_manual_mask.launch.py.

import argparse
import os

import cv2
import numpy as np
import yaml

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


def find_latest_run(camera_dir):
    rgb_root = os.path.join(camera_dir, "rgb")
    runs = sorted(os.listdir(rgb_root))
    if not runs:
        raise RuntimeError(f"Nessuna cartella trovata in {rgb_root}")
    return runs[-1]


class PhotoPublisher(Node):
    def __init__(self, save_dir, camera_name, run_timestamp, frame_id, rate_hz,
                 image_topic, depth_topic, camera_info_topic, camera_frame):
        super().__init__("photo_publisher")

        self.bridge = CvBridge()
        camera_dir = os.path.join(save_dir, camera_name)

        if run_timestamp is None:
            run_timestamp = find_latest_run(camera_dir)
            self.get_logger().info(
                f"Nessun --run_timestamp specificato, uso l'ultimo trovato: {run_timestamp}"
            )

        rgb_path = os.path.join(camera_dir, "rgb", run_timestamp, f"{frame_id:06d}.png")
        depth_path = os.path.join(camera_dir, "depth", run_timestamp, "png", f"{frame_id:06d}.png")
        intrinsics_path = os.path.join(camera_dir, "camera", run_timestamp, "intrinsics.yaml")

        for p in (rgb_path, depth_path, intrinsics_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"File non trovato: {p}")

        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if rgb_bgr is None:
            raise RuntimeError(f"Impossibile leggere {rgb_path}")
        if depth_mm is None:
            raise RuntimeError(f"Impossibile leggere {depth_path}")
        if depth_mm.dtype != np.uint16:
            raise RuntimeError(f"Depth attesa a 16 bit (mm), trovata {depth_mm.dtype}")

        with open(intrinsics_path) as f:
            intr = yaml.safe_load(f)

        self.rgb_msg = self.bridge.cv2_to_imgmsg(rgb_bgr, encoding="bgr8")
        self.depth_msg = self.bridge.cv2_to_imgmsg(depth_mm, encoding="16UC1")

        self.camera_info_msg = CameraInfo()
        self.camera_info_msg.width = int(intr["width"])
        self.camera_info_msg.height = int(intr["height"])
        K = [float(v) for row in intr["K"] for v in row]
        self.camera_info_msg.k = K
        self.camera_info_msg.distortion_model = "plumb_bob"
        self.camera_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.camera_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]
        self.camera_info_msg.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]

        for msg in (self.rgb_msg, self.depth_msg, self.camera_info_msg):
            msg.header.frame_id = camera_frame

        self.image_pub = self.create_publisher(Image, image_topic, 10)
        self.depth_pub = self.create_publisher(Image, depth_topic, 10)
        self.info_pub = self.create_publisher(CameraInfo, camera_info_topic, 10)

        self.get_logger().info(
            f"Pubblico in loop {rgb_path} / {depth_path} su "
            f"'{image_topic}', '{depth_topic}', '{camera_info_topic}' a {rate_hz} Hz"
        )

        self.timer = self.create_timer(1.0 / rate_hz, self.publish_once)

    def publish_once(self):
        now = self.get_clock().now().to_msg()
        self.rgb_msg.header.stamp = now
        self.depth_msg.header.stamp = now
        self.camera_info_msg.header.stamp = now

        self.image_pub.publish(self.rgb_msg)
        self.depth_pub.publish(self.depth_msg)
        self.info_pub.publish(self.camera_info_msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="/docker_shared")
    parser.add_argument("--camera_name", type=str, default="zed",
                         help="Sottocartella dentro save_dir (zed oppure realsense).")
    parser.add_argument(
        "--run_timestamp", type=str, default=None,
        help="Es. 2026-08-04_10-31-01. Se omesso, usa la cartella piu' recente.",
    )
    parser.add_argument("--frame_id", type=int, default=0)
    parser.add_argument("--rate_hz", type=float, default=10.0)
    parser.add_argument("--image_topic", type=str, default="image_rect")
    parser.add_argument("--depth_topic", type=str, default="depth")
    parser.add_argument("--camera_info_topic", type=str, default="camera_info_rect")
    parser.add_argument("--camera_frame", type=str, default="camera_color_optical_frame")
    args = parser.parse_args()

    rclpy.init()
    node = PhotoPublisher(
        save_dir=os.path.expanduser(args.save_dir),
        camera_name=args.camera_name,
        run_timestamp=args.run_timestamp,
        frame_id=args.frame_id,
        rate_hz=args.rate_hz,
        image_topic=args.image_topic,
        depth_topic=args.depth_topic,
        camera_info_topic=args.camera_info_topic,
        camera_frame=args.camera_frame,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
