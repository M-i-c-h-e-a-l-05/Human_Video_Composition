# Human Video Composition

A real-time computer vision project that places a segmented human from a webcam into a playing video background and allows the user to interact with the video using hand gestures.

The project combines **human segmentation, hand gesture recognition, real-time video compositing, and interactive video-region manipulation** using Python, OpenCV, and MediaPipe.

## Overview

The system uses a webcam to detect the user's hand and body while a video plays in the background.

The user's real background is removed using human segmentation, leaving only the person. The person is then composited over the playing video.

The system also allows the user to select a rectangular portion of the video using a hand gesture, cut that region out, and move it to another location using a fist gesture.

The cut region is not a frozen image. It continues to display the corresponding portion of the **currently playing video**, allowing the video content inside the moved region to continue changing normally.

### Layer Priority

The final composition follows this order:

```text
Human
  ↓
Movable Video Region
  ↓
Video Background
```

The human always has the highest visual priority.

---

## Features

### Human Segmentation

* Captures the user through a webcam.
* Detects and separates the human from the real webcam background.
* Uses MediaPipe Image Segmentation.
* Places the segmented human over the background video.

### Hand Gesture Control

The system uses MediaPipe Hand Landmarker to recognize hand gestures.

Current gestures:

| Gesture        | Action                            |
| -------------- | --------------------------------- |
| ✌️ Two fingers | Create a rectangular selection    |
| ✊ Fist         | Grab and move the selected region |
| Open hand      | Release the selected region       |
| `R`            | Reset the current cut             |
| `ESC`          | Exit the application              |

### Interactive Video Cutting

The user can:

1. Make a ✌️ gesture.
2. Move the index finger to draw a rectangle.
3. Release the gesture to select the region.
4. Place the hand inside the selected region.
5. Make a fist to grab it.
6. Move the fist to move the selected video region.
7. Open the hand to release it.

The original location becomes black while the selected video region can be moved elsewhere.

### Live Video Region

The selected region is taken from the **current video frame on every iteration**.

Therefore, the cut region continues playing the video instead of becoming a static screenshot.

For example, if a moving object is present in the selected area, its motion continues while the region is being moved.

---

## System Architecture

```text
                    Webcam
                       │
                       ▼
              ┌─────────────────┐
              │ Hand Detection  │
              │ MediaPipe       │
              └────────┬────────┘
                       │
                       │ Gestures
                       ▼
              ┌─────────────────┐
              │ Gesture Control │
              └────────┬────────┘
                       │
                       │
Webcam ────────────────┼──────────────► Human Segmentation
                       │                         │
                       │                         ▼
                       │                  Human Foreground
                       │
Video File ────────────────────────────► Video Background
                                                 │
                                                 ▼
                                      Movable Video Region
                                                 │
                                                 ▼
                                         Final Composition
                                                 │
                                                 ▼
                                      ┌──────────────────┐
                                      │ Human            │
                                      │ Cut Video Region │
                                      │ Video Background │
                                      └──────────────────┘
```

---

## Technologies Used

* **Python 3.12**
* **OpenCV**
* **MediaPipe Tasks**

  * Image Segmenter
  * Hand Landmarker
* **NumPy**
* **OpenCV VideoCapture**
* **TensorFlow Lite models through MediaPipe**

---

## Project Structure

```text
human_video_composition/
│
├── main.py
├── human_segmenter.py
├── gesture_controller.py
├── video_background.py
│
├── models/
│   ├── selfie_segmenter.tflite
│   └── hand_landmarker.task
│
├── output_trimmed.mp4
│
├── HVC/
│   └── Python virtual environment
│
└── README.md
```

### File Description

#### `main.py`

Main application loop.

Responsible for:

* Opening the webcam.
* Reading video frames.
* Running hand detection.
* Running human segmentation.
* Getting background video frames.
* Creating the final composition.
* Displaying the output.

#### `human_segmenter.py`

Handles human segmentation using MediaPipe Image Segmenter.

It generates a mask identifying the human and uses that mask to separate the person from the webcam background.

#### `gesture_controller.py`

Handles:

* Hand landmark detection.
* ✌️ gesture detection.
* ✊ fist detection.
* Rectangle creation.
* Rectangle selection.
* Cut-region movement.
* Live video-region extraction.
* Glowing rectangle visualization.

#### `video_background.py`

Handles the background video.

Responsibilities include:

* Opening the video.
* Reading frames.
* Resizing frames to webcam resolution.
* Restarting the video when it reaches the end.

---

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd human_video_composition
```

### 2. Create Virtual Environment

```bash
python3.12 -m venv HVC
```

Activate it:

```bash
source HVC/bin/activate
```

### 3. Install Dependencies

```bash
pip install opencv-python numpy mediapipe
```

---

## MediaPipe Models

The project requires two MediaPipe model files.

### Human Segmentation Model

Place the segmentation model inside:

```text
models/selfie_segmenter.tflite
```

### Hand Landmarker Model

Place the hand model inside:

```text
models/hand_landmarker.task
```

The `models` directory should therefore contain:

```text
models/
├── selfie_segmenter.tflite
└── hand_landmarker.task
```

---

## Background Video

Place the background video in the project directory:

```text
output_trimmed.mp4
```

The current application expects:

```python
background = VideoBackground(
    "output_trimmed.mp4"
)
```

To use another video, change the path in `main.py`.

---

## Running the Project

Activate the virtual environment:

```bash
source HVC/bin/activate
```

Run:

```bash
python main.py
```

A window named:

```text
Human in Video
```

will open.

---

## How to Use

### Step 1 — Enter the Scene

Stand in front of the webcam.

Your real background will be removed and replaced with the selected video.

```text
Webcam
   ↓
Human segmentation
   ↓
Your body + video background
```

### Step 2 — Draw a Cut Region

Make a:

```text
✌️
```

gesture.

Move your index finger to define the rectangular region.

The system converts the freehand movement into a straight-edged rectangle.

### Step 3 — Select the Region

Stop making the ✌️ gesture.

The rectangle becomes selected.

### Step 4 — Grab the Region

Move your hand inside the selected rectangle.

Make a:

```text
✊
```

fist.

The selected video region is now grabbed.

### Step 5 — Move the Region

Move your fist.

The selected region follows your hand.

The original location becomes black.

### Step 6 — Release

Open your hand.

The video region is released at its current location.

The video inside the region continues playing.

---

## Important Rendering Rule

The project intentionally maintains this layer priority:

```text
┌─────────────────────────────┐
│        HUMAN                │  ← Highest
├─────────────────────────────┤
│    MOVABLE VIDEO CUT        │
├─────────────────────────────┤
│    VIDEO BACKGROUND         │  ← Lowest
└─────────────────────────────┘
```

This means that if the movable video region passes over the user's body, the human remains visible.

The cut region cannot visually replace the segmented human.

---

## Interaction Logic

### Rectangle Creation

```text
✌️ detected
     ↓
Start point = index fingertip
     ↓
Track index fingertip
     ↓
Create straight rectangle
     ↓
Release ✌️
     ↓
Rectangle selected
```

### Cut and Move

```text
Hand inside rectangle
        ↓
      ✊ Fist
        ↓
    Grab region
        ↓
 Move hand/fist
        ↓
Move video region
        ↓
 Open hand
        ↓
 Release region
```

### Invalid Grab

A fist outside the selected rectangle does nothing.

```text
✊ outside rectangle
        ↓
     No action
```

This prevents accidental movement of the selected region.

---

## Real-Time Video Cutting

The project does not permanently extract a static image from the video.

Instead, the source rectangle is remembered:

```text
Source rectangle
       ↓
Current video frame
       ↓
Extract region
       ↓
Place at current destination
```

This process happens continuously.

Therefore:

```text
Frame 1 → extract video region
Frame 2 → extract new video region
Frame 3 → extract new video region
Frame 4 → extract new video region
...
```

The content inside the movable region continues to update with the video.

---

## Controls

| Input     | Function                  |
| --------- | ------------------------- |
| ✌️        | Draw/select video region  |
| ✊         | Grab/move selected region |
| Open hand | Release region            |
| `R`       | Reset cut                 |
| `ESC`     | Exit                      |

---

## Current Limitations

* Human segmentation quality depends on lighting and camera conditions.
* Hand gesture detection can become less reliable when the hand is heavily rotated or occluded.
* The current finger-state detection uses landmark positions and is optimized for the intended gestures.
* Only one hand is currently tracked.
* The current implementation supports one active movable video region.
* The selected source region remains associated with its original location in the background video.
* The original cut location is currently represented as a black region.

---

## Future Improvements

Possible future extensions include:

* Multiple simultaneous cut regions.
* More accurate hand gesture recognition.
* Smooth gesture tracking.
* Edge feathering around the cut region.
* Adjustable cut-region size.
* Rotation and scaling of cut regions.
* Multiple movable video objects.
* Gesture-controlled video playback.
* Play/pause using hand gestures.
* Video seeking using gestures.
* Volume control using gestures.
* Switching between multiple background videos.
* Improved human segmentation around hair and hands.
* Temporal smoothing for more stable segmentation.
* GPU acceleration where available.
* More advanced compositing effects.

---

## Project Goal

The goal of **Human Video Composition** is to create an interactive real-time video environment where a person can become part of a digital scene and manipulate portions of that scene using natural hand gestures.

The project combines:

**Computer Vision + Human Segmentation + Gesture Recognition + Real-Time Video Processing + Interactive Compositing**

into a single real-time application.
