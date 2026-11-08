
import argparse
from pathlib import Path
import sys
import time

import cv2
from ultralytics import YOLO


NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_WRIST = 9
RIGHT_WRIST = 10


def hands_above_head(
    keypoints,
    required_hands: int = 2,
    min_keypoint_confidence: float = 0.35,
) -> bool:

    required_indices = (
        NOSE,
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_WRIST,
        RIGHT_WRIST,
    )
    if keypoints is None or len(keypoints) <= max(required_indices):
        return False

    joints = keypoints
    if hasattr(joints, "cpu"):
        joints = joints.cpu().numpy()

    if joints.shape[1] < 3:
        return False
    if any(joints[index, 2] < min_keypoint_confidence for index in required_indices):
        return False

    nose_y = float(joints[NOSE, 1])
    shoulder_y = (float(joints[LEFT_SHOULDER, 1]) + float(joints[RIGHT_SHOULDER, 1])) / 2
    shoulder_to_nose = shoulder_y - nose_y
    if shoulder_to_nose <= 0:
        return False

    head_top_y = nose_y - 0.6 * shoulder_to_nose
    raised_wrists = sum(
        float(joints[index, 1]) < head_top_y
        for index in (LEFT_WRIST, RIGHT_WRIST)
    )
    return raised_wrists >= required_hands


def parse_source(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Camera index, video path, or stream URL (default: 0)")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="Ultralytics pose model weights")
    parser.add_argument(
        "--hands",
        choices=("one", "both"),
        default="both",
        help="How many hands must be above the head (default: both)",
    )
    parser.add_argument(
        "--keypoint-confidence",
        type=float,
        default=0.35,
        help="Minimum confidence for required pose landmarks (default: 0.35)",
    )
    args = parser.parse_args()

    if not 0 <= args.keypoint_confidence <= 1:
        parser.error("--keypoint-confidence must be between 0 and 1")

    model_path = Path(args.model)
    try:
        model = YOLO(str(model_path) if model_path.exists() else args.model)
    except Exception as exc:
        print(f"Could not load pose model '{args.model}': {exc}", file=sys.stderr)
        return 1

    capture = cv2.VideoCapture(parse_source(args.source))
    if not capture.isOpened():
        print(f"Could not open video source: {args.source}", file=sys.stderr)
        return 1

    required_hands = 1 if args.hands == "one" else 2
    was_detected = False
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            results = model(frame, verbose=False)
            detected = False
            for result in results:
                if result.keypoints is None:
                    continue
                poses = result.keypoints.data
                for pose in poses:
                    if hands_above_head(
                        pose,
                        required_hands=required_hands,
                        min_keypoint_confidence=args.keypoint_confidence,
                    ):
                        detected = True
                        break
                if detected:
                    break

            label = "HANDS RAISED ABOVE HEAD" if detected else "No hand raise"
            color = (0, 220, 0) if detected else (40, 40, 230)
            cv2.putText(
                frame,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("YOLO hand-raise detection", frame)

            if detected and not was_detected:
                print("Hands raised above head")
            was_detected = detected

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
