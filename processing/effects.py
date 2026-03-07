"""
Visual effects module.
Handles translucency and prepares masks for future inpainting.
"""

import cv2
import numpy as np
from config import SELECTED_PLAYER_OPACITY


def apply_translucency(frame, boxes, masks, selected_players, frame_shape):
    """
    Apply translucency to selected players.

    Args:
        frame
        boxes
        masks
        selected_players
        frame_shape

    Returns
        processed frame
    """

    if boxes is None or len(boxes) == 0:
        return frame.copy()

    if not selected_players:
        return frame.copy()

    h, w = frame_shape

    result = frame.copy()

    if masks is None:
        return result

    for idx in selected_players:

        if idx >= len(masks):
            continue

        mask = masks[idx]

        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        mask = mask.astype(np.float32)

        alpha = SELECTED_PLAYER_OPACITY

        mask_3 = np.repeat(mask[:, :, None], 3, axis=2)

        result = result * (1 - mask_3) + (result * alpha) * mask_3

    result = result.astype(np.uint8)

    return result


def create_debug_frame(frame, boxes, masks, opacities, selected_players):

    debug = frame.copy()

    if boxes is None or len(boxes) == 0:
        cv2.putText(
            debug,
            "No players detected",
            (40, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        return debug

    for i, box in enumerate(boxes):

        x1, y1, x2, y2 = map(int, box[:4])

        if i in selected_players:
            color = (0, 0, 255)
            label = "SELECTED"
        else:
            color = (0, 255, 0)
            label = f"{i}"

        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            debug,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

        if opacities is not None and i < len(opacities):

            cv2.putText(
                debug,
                f"{opacities[i]:.2f}",
                (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

    return debug