"""
Player detection module using YOLOv8 segmentation.
Handles loading the model and running inference on frames.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch


class PlayerDetector:
    """
    Detect players in frames using YOLOv8 segmentation.
    Returns bounding boxes and segmentation masks aligned with the frame.
    """

    def __init__(self, model_name="yolov8m-seg.pt", classes=[0], device=None):
        """
        Args:
            model_name: YOLO model path
            classes: COCO classes to detect (0 = person)
            device: 'cuda' or 'cpu'
        """

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading YOLO model on {self.device}...")
        self.model = YOLO(model_name)

        self.classes = classes

        print("Model ready")

    def detect(self, frame):
        """
        Run detection on frame.

        Returns
        boxes : numpy array (N,4)
        masks : numpy array (N,H,W) or None
        """

        if frame is None:
            return np.array([]), None

        results = self.model(
            frame,
            device=self.device,
            classes=self.classes,
            verbose=False
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return np.array([]), None

        boxes = results.boxes.xyxy.detach().cpu().numpy()

        masks = None

        if results.masks is not None:

            raw_masks = results.masks.data.detach().cpu().numpy()
            frame_h, frame_w = frame.shape[:2]

            resized_masks = []

            for m in raw_masks:

                mask = cv2.resize(
                    m,
                    (frame_w, frame_h),
                    interpolation=cv2.INTER_NEAREST
                )

                mask = (mask > 0.5).astype(np.uint8)
                resized_masks.append(mask)

            masks = np.array(resized_masks)

        return boxes, masks

    def detect_resized(self, frame, size=640):
        """
        Faster detection using resized inference.
        """

        h, w = frame.shape[:2]

        resized = cv2.resize(frame, (size, size))

        boxes, masks = self.detect(resized)

        if len(boxes) == 0:
            return boxes, None

        scale_x = w / size
        scale_y = h / size

        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        if masks is not None:

            restored = []

            for m in masks:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                restored.append(m)

            masks = np.array(restored)

        return boxes, masks

    def get_model_info(self):

        return {
            "device": self.device,
            "classes": self.classes
        }