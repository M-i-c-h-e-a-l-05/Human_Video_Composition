import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandDetector:

    def __init__(
        self,
        model_path="models/hand_landmarker.task"
    ):

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.detector = (
            vision.HandLandmarker.create_from_options(
                options
            )
        )

    def detect(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.detector.detect(
            mp_image
        )

        if not result.hand_landmarks:
            return None

        return result.hand_landmarks[0]

    def close(self):

        self.detector.close()