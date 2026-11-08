# YOLO hand-raise detection

Detect when one or both wrists are above an estimated head-top using an
Ultralytics YOLO pose model and OpenCV.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python hand_raise.py
```

The default uses webcam `0` and the `yolo11n-pose.pt` model. Ultralytics
downloads the model weights on first use. Press `q` to quit.

Use `--hands one` to detect either hand, or provide a video file, camera index,
or stream URL with `--source`:

```powershell
python hand_raise.py --hands one --source 1
python hand_raise.py --source "C:\videos\gesture.mp4"
```

The app displays a status label and prints a message when a raise starts. Since
the COCO pose model does not provide a head-top landmark, the detector estimates
that point from the nose and shoulder positions.
