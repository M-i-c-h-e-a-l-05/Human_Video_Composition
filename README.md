# Human Video Composition

A real-time computer vision project that combines **human segmentation, hand gesture recognition, and video compositing** to create an interactive video composition system. Users can manipulate selected video regions using hand gestures while keeping the segmented human in the foreground.

## Overview

Human Video Composition uses a webcam to track hand gestures and segment the person in the scene. A separate video plays as the background, while a selected region of the camera feed can be cut out, moved, resized, and positioned interactively.

The system combines these visual layers to produce a dynamic composition without requiring a green screen or traditional editing tools.

## Features

- **Real-time Human Segmentation:** Separates the person from the surrounding scene.
- **Hand Gesture Recognition:** Uses hand landmarks to control video regions.
- **Interactive Video Cutting:** Draw a rectangular region using hand gestures.
- **Move and Resize:** Reposition and adjust the selected video region.
- **Layered Compositing:** Keeps the segmented human in the foreground, the selected video region in the middle, and the background video at the bottom.
- **Grid Mode:** Supports grid-based region selection.
- **Region Locking:** Prevents accidental movement of a selected region.
- **Delete and Restore:** Provides a control for removing or restoring the selected region.
- **Video Background:** Plays a background video behind the person and the selected region.

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- TensorFlow Lite
- Computer Vision
- Image and Video Processing

## Project Architecture

```text
Human_Video_Composition/
├── main.py
├── gesture/
│   └── Hand gesture recognition modules
├── human/
│   └── Human segmentation modules
├── video/
│   └── Video background and composition modules
├── models/
│   ├── hand_landmarker.task
│   └── selfie_segmenter.tflite
├── assets/
│   └── videos/
│       └── output_trimmed.mp4
├── .gitignore
└── README.md
```

*Note: The directory descriptions are illustrative. Keep the file names and folder structure consistent with your actual repository.*

## How It Works

The system processes the webcam feed and background video through several stages:

1. **Camera Input:** Captures live frames from the webcam.
2. **Hand Tracking:** Detects hand landmarks and identifies gestures.
3. **Human Segmentation:** Separates the person from the original camera frame.
4. **Region Selection:** Allows the user to select a region of the camera feed.
5. **Region Manipulation:** Supports moving, resizing, locking, and deleting the selected region.
6. **Video Compositing:** Combines the background video, selected video region, and segmented person into the final output.

### Composition Layer Order

1. **Foreground:** Segmented human.
2. **Middle layer:** Selected video region.
3. **Background:** Looping background video.

This ordering allows the person to remain visible in front of the selected video region.

## Gesture Controls

| Gesture / Control | Action |
|---|---|
| Index and middle fingers extended (✌️) | Draw a rectangular cut region |
| Closed fist | Grab and move the region when the hand is inside it |
| Thumb and index finger | Resize the region while grabbed |
| Open hand | Release the region |
| Top-left fix control | Lock the region |
| Top-right trash control | Delete or restore the region |
| Top-center Grid button | Enable grid-based selection |

Grid Mode is intended to disable freehand cutting and free scaling while grid selection is active.

## Requirements

- Python 3.10 or a compatible Python version
- A webcam
- OpenCV
- MediaPipe
- NumPy
- Required model files
- A background video asset

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/M-i-c-h-e-a-l-05/Human_Video_Composition.git
cd Human_Video_Composition
```

### 2. Create a virtual environment

```bash
python3 -m venv HVC
source HVC/bin/activate
```

### 3. Install dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies required by the current implementation. For example:

```bash
pip install opencv-python mediapipe numpy
```

Additional dependencies may be required by the installed MediaPipe version and the project's implementation.

### 4. Check the required assets

Ensure the following files are available at their expected locations:

- `models/hand_landmarker.task`
- `models/selfie_segmenter.tflite`
- `assets/videos/output_trimmed.mp4`

Model files and video assets must be present locally if they are not included in the repository.

### 5. Run the project

```bash
python main.py
```

Allow camera access if prompted.

## Performance Considerations

- Performance depends on webcam resolution, CPU capability, and the segmentation model.
- Real-time processing may be slower on systems without a dedicated GPU.
- Lowering the processing resolution can help improve performance.
- Accurate hand tracking and stable segmentation are important for smooth interactions.

## Future Improvements

- Improve segmentation stability and reduce flickering.
- Refine gesture recognition and region manipulation.
- Add more flexible region shapes and composition options.
- Improve grid-based selection and interaction feedback.
- Optimize processing speed for CPU-only systems.
- Support saving and exporting the final composition.

## Applications

- Interactive video editing
- Gesture-controlled visual effects
- Real-time video compositing
- Computer vision demonstrations
- Human-computer interaction

## Author

**Micheal George Leveiro**

Computer Science and Engineering — Artificial Intelligence and Machine Learning

GitHub: [M-i-c-h-e-a-l-05](https://github.com/M-i-c-h-e-a-l-05)
