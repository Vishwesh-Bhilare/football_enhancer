"""
Main entry point for the football video enhancer.
"""

import cv2
import argparse
import numpy as np

from config import *
from models.detector import PlayerDetector
from processing.effects import apply_translucency, create_debug_frame
from processing.tracker import PlayerTracker
from processing.opacity import calculate_batch_opacity
from utils.visualization import draw_selection_overlay, draw_instructions, FPSCounter


class AppState:

    def __init__(self):

        self.effect_enabled = True

        self.selected_players = set()
        self.selected_tracked_ids = set()

        self.current_boxes = []
        self.current_masks = []

        self.frame_shape = None

        self.tracker = None

        self.show_debug = False

        self.fps_counter = FPSCounter()


def mouse_callback(event, x, y, flags, param):

    state = param

    if event == cv2.EVENT_LBUTTONDOWN:

        for i, box in enumerate(state.current_boxes):

            x1, y1, x2, y2 = map(int, box[:4])

            if x1 <= x <= x2 and y1 <= y <= y2:

                if state.tracker and i in state.tracker.id_mapping:

                    tracked_id = state.tracker.id_mapping[i]

                    state.selected_tracked_ids.add(tracked_id)

                    state.selected_players = state.tracker.get_detection_indices(
                        state.selected_tracked_ids
                    )

                else:

                    state.selected_players.add(i)

                break

    elif event == cv2.EVENT_RBUTTONDOWN:

        for i, box in enumerate(state.current_boxes):

            x1, y1, x2, y2 = map(int, box[:4])

            if x1 <= x <= x2 and y1 <= y <= y2:

                if state.tracker and i in state.tracker.id_mapping:

                    tracked_id = state.tracker.id_mapping[i]

                    if tracked_id in state.selected_tracked_ids:

                        state.selected_tracked_ids.remove(tracked_id)

                        state.selected_players = state.tracker.get_detection_indices(
                            state.selected_tracked_ids
                        )

                else:

                    if i in state.selected_players:
                        state.selected_players.remove(i)

                break


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--no-track", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print("Cannot open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video loaded: {fps:.2f} fps | {total_frames} frames")

    detector = PlayerDetector(YOLO_MODEL_NAME, DETECTION_CLASSES)

    tracker = PlayerTracker() if not args.no_track else None

    state = AppState()

    state.tracker = tracker
    state.show_debug = args.debug

    cv2.namedWindow(WINDOW_NAME)

    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, state)

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1

        state.frame_shape = frame.shape[:2]

        boxes, masks = detector.detect(frame)

        state.fps_counter.update()

        if tracker and boxes is not None and len(boxes) > 0:

            id_mapping, tracked_boxes = tracker.update(boxes)

            state.current_boxes = tracked_boxes

            state.selected_players = tracker.get_detection_indices(
                state.selected_tracked_ids
            )

        else:

            state.current_boxes = boxes if boxes is not None else []

        state.current_masks = masks

        if state.effect_enabled and boxes is not None and len(boxes) > 0:

            output_frame = apply_translucency(
                frame,
                boxes,
                masks,
                state.selected_players,
                state.frame_shape,
            )

        else:

            output_frame = frame.copy()

        output_frame = draw_selection_overlay(
            output_frame,
            state.current_boxes,
            state.selected_players,
        )

        if state.show_debug and boxes is not None:

            frame_area = state.frame_shape[0] * state.frame_shape[1]

            opacities = calculate_batch_opacity(boxes, frame_area)

            output_frame = create_debug_frame(
                output_frame,
                boxes,
                masks,
                opacities,
                state.selected_players,
            )

        output_frame = state.fps_counter.draw(output_frame)

        output_frame = draw_instructions(
            output_frame,
            state.effect_enabled,
            len(state.selected_players),
        )

        cv2.putText(
            output_frame,
            f"Frame {frame_count}/{total_frames}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        cv2.imshow(WINDOW_NAME, output_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == KEY_QUIT:
            break

        elif key == KEY_TOGGLE_EFFECT:

            state.effect_enabled = not state.effect_enabled

        elif key == KEY_DESELECT_ALL:

            state.selected_players.clear()
            state.selected_tracked_ids.clear()

        elif key == KEY_DEBUG_TOGGLE:

            state.show_debug = not state.show_debug

        elif key == KEY_SAVE_FRAME:

            cv2.imwrite(f"frame_{frame_count}.jpg", output_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()