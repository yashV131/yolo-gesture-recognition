import argparse
from pathlib import Path
import sys

import cv2
import depthai as dai
from ultralytics import YOLO


# YOLO pose landmark indices
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_WRIST = 9
RIGHT_WRIST = 10


def hands_above_head(
    keypoints,
    required_hands=2,
    min_keypoint_confidence=0.35,
):
    required_indices = (
        NOSE,
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_WRIST,
        RIGHT_WRIST,
    )

    if keypoints is None:
        return False

    joints = keypoints

    if hasattr(joints, "cpu"):
        joints = joints.cpu().numpy()

    if len(joints) <= max(required_indices):
        return False

    if joints.shape[1] < 3:
        return False

    for index in required_indices:
        if joints[index, 2] < min_keypoint_confidence:
            return False

    nose_y = float(joints[NOSE, 1])

    shoulder_y = (
        float(joints[LEFT_SHOULDER, 1])
        + float(joints[RIGHT_SHOULDER, 1])
    ) / 2

    shoulder_to_nose = shoulder_y - nose_y

    if shoulder_to_nose <= 0:
        return False

    head_top_y = nose_y - 0.6 * shoulder_to_nose

    raised_wrists = sum(
        float(joints[index, 1]) < head_top_y
        for index in (LEFT_WRIST, RIGHT_WRIST)
    )

    return raised_wrists >= required_hands


def detect_hand_raise(
    frame,
    model,
    required_hands,
    min_keypoint_confidence,
):
    results = model(frame, verbose=False)

    for result in results:
        if result.keypoints is None:
            continue

        for pose in result.keypoints.data:
            if hands_above_head(
                pose,
                required_hands,
                min_keypoint_confidence,
            ):
                return True

    return False


def annotate_frame(frame, detected):
    label = (
        "HANDS RAISED ABOVE HEAD"
        if detected
        else "No hand raise"
    )

    color = (
        (0, 220, 0)
        if detected
        else (40, 40, 230)
    )

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

    return frame


def run_oak_camera(
    model,
    required_hands,
    min_keypoint_confidence,
    camera_socket,
):
    print("\nSearching for Luxonis OAK cameras...")

    # Detect connected OAK devices
    devices = dai.Device.getAllAvailableDevices()

    if len(devices) == 0:
        print("ERROR: No OAK camera detected by DepthAI.")
        print("Check USB connection and close OAK Viewer.")
        return 1

    print(f"Found {len(devices)} OAK camera(s)!")

    for device in devices:
        print("Device ID:", device.getDeviceId())

    # Create DepthAI pipeline
    pipeline = dai.Pipeline()

    # Connect to the selected physical camera socket
    camera = pipeline.create(dai.node.Camera).build(
        camera_socket
    )

    # Request image output
    stream = camera.requestOutput(
        size=(1280, 720),
        type=dai.ImgFrame.Type.NV12,
    )

    # Create image queue
    frame_queue = stream.createOutputQueue(
        maxSize=4,
        blocking=False,
    )

    was_detected = False

    try:
        print("\nStarting OAK camera pipeline...")

        pipeline.start()

        print("Camera started successfully!")
        print("Running YOLO hand-raise detection.")
        print("Press Q to quit.\n")

        while pipeline.isRunning():

            message = frame_queue.tryGet()

            if message is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # Convert DepthAI image to OpenCV frame
            frame = message.getCvFrame()

            # Run YOLO pose detection
            detected = detect_hand_raise(
                frame,
                model,
                required_hands,
                min_keypoint_confidence,
            )

            # Annotate frame
            annotate_frame(frame, detected)

            # Print event only when detection starts
            if detected and not was_detected:
                print("HANDS RAISED ABOVE HEAD!")

            was_detected = detected

            # Display live camera
            cv2.imshow(
                "OAK-D-S2-FF | YOLO Hand Raise Detection",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as exc:
        print(f"\nCamera error: {exc}", file=sys.stderr)
        return 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stopped.")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="OAK-D hand raise detection using YOLO pose."
    )

    parser.add_argument(
        "--model",
        default="yolo11n-pose.pt",
        help="YOLO pose model weights",
    )

    parser.add_argument(
        "--hands",
        choices=("one", "both"),
        default="both",
        help="Number of hands required above head",
    )

    parser.add_argument(
        "--keypoint-confidence",
        type=float,
        default=0.35,
        help="Minimum YOLO keypoint confidence",
    )

    parser.add_argument(
        "--socket",
        choices=("CAM_B", "CAM_C"),
        default="CAM_B",
        help="OAK mono camera socket (default: left camera CAM_B)",
    )

    args = parser.parse_args()

    if not 0 <= args.keypoint_confidence <= 1:
        parser.error(
            "--keypoint-confidence must be between 0 and 1"
        )

    # Load YOLO model
    model_path = Path(args.model)

    try:
        model = YOLO(
            str(model_path)
            if model_path.exists()
            else args.model
        )

    except Exception as exc:
        print(
            f"Could not load YOLO model: {exc}",
            file=sys.stderr,
        )
        return 1

    required_hands = (
        1 if args.hands == "one" else 2
    )

    # Select physical camera socket
    camera_socket = (
        dai.CameraBoardSocket.CAM_B
        if args.socket == "CAM_B"
        else dai.CameraBoardSocket.CAM_C
    )

    # Run external OAK camera
    return run_oak_camera(
        model,
        required_hands,
        args.keypoint_confidence,
        camera_socket,
    )


if __name__ == "__main__":
    raise SystemExit(main())