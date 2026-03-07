"""
Opacity calculation module.
Maps player distance proxies to opacity values.
Distance is approximated using bounding box area or optional ball distance.
"""

import numpy as np
from config import *


def calculate_opacity_from_bbox(bbox, frame_area):
    """
    Estimate opacity from bounding box size.

    Larger bbox → player closer → more opaque.
    Smaller bbox → player farther → more translucent.
    """

    x1, y1, x2, y2 = bbox[:4]

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    bbox_area = width * height
    area_ratio = bbox_area / frame_area

    min_ratio = AREA_RATIO_FOR_MIN_OPACITY
    max_ratio = AREA_RATIO_FOR_MAX_OPACITY

    if area_ratio <= min_ratio:
        return OPACITY_MIN

    if area_ratio >= max_ratio:
        return OPACITY_MAX

    t = (area_ratio - min_ratio) / (max_ratio - min_ratio)

    opacity = OPACITY_MIN + t * (OPACITY_MAX - OPACITY_MIN)

    return float(opacity)


def calculate_batch_opacity(bboxes, frame_area):
    """
    Compute opacity for all players in a frame.
    """

    if bboxes is None or len(bboxes) == 0:
        return np.array([])

    opacities = []

    for bbox in bboxes:
        opacities.append(calculate_opacity_from_bbox(bbox, frame_area))

    return np.array(opacities)


def bbox_center(bbox):
    """
    Compute bbox center.
    """

    x1, y1, x2, y2 = bbox[:4]

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    return cx, cy


def distance(p1, p2):

    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_opacity_from_ball_distance(boxes, ball_center, frame_diag):
    """
    Optional advanced opacity mapping based on distance to ball.

    Closer to ball → more visible.
    Farther from ball → more translucent.
    """

    if boxes is None or len(boxes) == 0 or ball_center is None:
        return None

    opacities = []

    for box in boxes:

        player_center = bbox_center(box)

        d = distance(player_center, ball_center)

        norm_d = d / frame_diag

        opacity = OPACITY_MAX - norm_d * (OPACITY_MAX - OPACITY_MIN)

        opacity = np.clip(opacity, OPACITY_MIN, OPACITY_MAX)

        opacities.append(opacity)

    return np.array(opacities)