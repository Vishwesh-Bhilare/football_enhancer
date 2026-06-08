from pathlib import Path

import cv2
import numpy as np


class SAMRefiner:

    def __init__(self, model_type="vit_b", checkpoint="sam_vit_b_01ec64.pth"):
        self.predictor = None
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(__file__).resolve().parents[1] / checkpoint_path

        if not checkpoint_path.exists():
            print(f"SAM disabled: checkpoint not found ({checkpoint_path})")
            return

        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            print(f"SAM disabled: optional dependency unavailable ({exc})")
            return

        print("Loading SAM (CPU mode)...")

        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        sam.to("cpu")
        self.predictor = SamPredictor(sam)

        print("SAM ready (CPU, optimized)")

    @property
    def is_enabled(self):
        return self.predictor is not None

    def refine(self, frame, boxes):
        """
        frame: BGR image
        boxes: [N,4]
        returns: masks [N, H, W], or None when SAM is unavailable
        """

        if not self.is_enabled or boxes is None or len(boxes) == 0:
            return None

        orig_h, orig_w = frame.shape[:2]

        max_dim = 512
        scale = min(1.0, max_dim / max(orig_h, orig_w))

        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))

        frame_small = cv2.resize(frame, (new_w, new_h))
        frame_small_rgb = frame_small[:, :, ::-1]

        self.predictor.set_image(frame_small_rgb)

        masks = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            input_box = np.array([
                int(x1 * scale),
                int(y1 * scale),
                int(x2 * scale),
                int(y2 * scale),
            ])

            mask, _, _ = self.predictor.predict(
                box=input_box,
                multimask_output=False,
            )

            mask_full = cv2.resize(
                mask[0].astype(np.uint8),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )
            masks.append(mask_full)

        return np.array(masks) if masks else None
