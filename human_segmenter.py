import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HumanSegmenter:

    def __init__(
        self,
        model_path="models/selfie_segmenter.tflite",
        threshold=0.5
    ):

        self.threshold = threshold

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_confidence_masks=True
        )

        self.segmenter = (
            vision.ImageSegmenter.create_from_options(
                options
            )
        )

    def get_mask(self, frame):

        # -------------------------------------------------
        # Convert BGR -> RGB
        # -------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # -------------------------------------------------
        # Create MediaPipe image
        # -------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # -------------------------------------------------
        # Run segmentation
        # -------------------------------------------------

        result = self.segmenter.segment(
            mp_image
        )

        # -------------------------------------------------
        # Get person confidence mask
        # -------------------------------------------------

        person_confidence = (
            result.confidence_masks[0].numpy_view()
        )

        # -------------------------------------------------
        # Convert confidence -> binary mask
        #
        # 255 = HUMAN
        # 0   = BACKGROUND
        # -------------------------------------------------

        mask = (
            person_confidence >= self.threshold
        ).astype(np.uint8) * 255

        return mask

    def extract_person(self, frame):

        mask = self.get_mask(
            frame
        )

        person = cv2.bitwise_and(
            frame,
            frame,
            mask=mask
        )

        return person

    def get_person_with_alpha(self, frame):

        mask = self.get_mask(
            frame
        )

        bgra = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2BGRA
        )

        bgra[:, :, 3] = mask

        return bgra

    def close(self):

        self.segmenter.close()