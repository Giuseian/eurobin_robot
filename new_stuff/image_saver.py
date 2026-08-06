#!/usr/bin/env python3
# Salva su file (con nome basato sul timestamp) il primo messaggio ricevuto
# su un topic immagine ROS2. Usato per esempio per salvare l'immagine
# annotata da isaac_ros_rtdetr_visualizer.py (/rtdetr_processed_image)
# senza bisogno di RViz2/X11.

import argparse
import os
from datetime import datetime

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class ImageSaver(Node):
    def __init__(self, topic, output_dir):
        super().__init__("image_saver")
        self.bridge = CvBridge()
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_path = os.path.join(output_dir, f"{timestamp}.png")
        self.saved = False
        self.create_subscription(Image, topic, self.callback, 10)
        self.get_logger().info(f"In attesa di un messaggio su '{topic}'...")

    def callback(self, msg):
        if self.saved:
            return
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imwrite(self.output_path, img)
        self.get_logger().info(f"Salvato: {self.output_path}")
        self.saved = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default="/rtdetr_processed_image")
    parser.add_argument("--output_dir", type=str, default="/docker_shared/zed/debug/annotated")
    args = parser.parse_args()

    rclpy.init()
    node = ImageSaver(args.topic, args.output_dir)
    try:
        while rclpy.ok() and not node.saved:
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
