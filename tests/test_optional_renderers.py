import sys
from types import SimpleNamespace

import numpy as np

# The CI image lacks libGL; these tests exercise disabled paths that do not use OpenCV.
sys.modules.setdefault("cv2", SimpleNamespace())

from processing.sam_refiner import SAMRefiner
from processing.sd_inpainting import SDInpainter


def test_sam_disables_itself_without_checkpoint(tmp_path):
    refiner = SAMRefiner(checkpoint=tmp_path / "missing.pth")

    assert not refiner.is_enabled
    assert refiner.refine(np.zeros((4, 4, 3), dtype=np.uint8), []) is None


def test_sd_returns_frame_unchanged_when_disabled():
    inpainter = SDInpainter.__new__(SDInpainter)
    inpainter.pipe = None
    inpainter.generator = None
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)

    assert not inpainter.is_enabled
    assert inpainter.inpaint(frame, mask) is frame
