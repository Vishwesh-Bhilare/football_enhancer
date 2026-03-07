"""
AI / classical inpainting module.

Used to fill regions where players are removed.
Supports fast OpenCV methods now and can later integrate
deep learning models (LaMa, Stable Diffusion).
"""

import cv2
import numpy as np
from config import INPAINTING_METHOD, INPAINTING_RADIUS


class Inpainter:

    def __init__(self, method=INPAINTING_METHOD):

        self.method = method

        if method == "telea":
            self.flag = cv2.INPAINT_TELEA

        elif method == "ns":
            self.flag = cv2.INPAINT_NS

        else:
            raise ValueError("Unsupported inpainting method")

    def inpaint(self, frame, mask):

        """
        Perform inpainting.

        frame : original frame
        mask : binary mask where pixels should be filled
        """

        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        mask = mask * 255

        result = cv2.inpaint(
            frame,
            mask,
            INPAINTING_RADIUS,
            self.flag
        )

        return result


def create_combined_mask(masks, selected_players, frame_shape):

    """
    Combine multiple player masks into one mask.
    """

    if masks is None or len(masks) == 0:
        return None

    h, w = frame_shape

    combined = np.zeros((h, w), dtype=np.uint8)

    for idx in selected_players:

        if idx >= len(masks):
            continue

        mask = masks[idx]

        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        combined = np.maximum(combined, mask)

    return combined