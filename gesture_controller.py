import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class GestureController:

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

        self.detector = vision.HandLandmarker.create_from_options(
            options
        )

        # Current interaction state
        self.mode = "NORMAL"

        # Rectangle drawing
        self.start_point = None
        self.current_point = None
        self.rectangle = None

        # Fist dragging
        self.dragging = False
        self.drag_offset_x = 0
        self.drag_offset_y = 0

        # Actual cut-out data
        self.cut_patch = None
        self.cut_source_rectangle = None
        self.cut_active = False
        self.cut_just_started = False

        # -----------------------------------------------------
        # Gesture stability
        # -----------------------------------------------------

        # Number of consecutive frames required before
        # accepting that the CUT gesture has really ended.
        self.cut_release_frames = 5
        self.cut_release_counter = 0

    # =========================================================
    # HAND DETECTION
    # =========================================================

    def detect_hand(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            return None

        return result.hand_landmarks[0]

    # =========================================================
    # FINGER STATES
    # =========================================================

    def finger_states(self, landmarks):

        # Index
        index_up = (
            landmarks[8].y <
            landmarks[6].y
        )

        # Middle
        middle_up = (
            landmarks[12].y <
            landmarks[10].y
        )

        # Ring
        ring_closed = (
            landmarks[16].y >
            landmarks[14].y
        )

        # Pinky
        pinky_closed = (
            landmarks[20].y >
            landmarks[18].y
        )

        return (
            index_up,
            middle_up,
            ring_closed,
            pinky_closed
        )

    # =========================================================
    # CUT GESTURE
    # =========================================================

    def is_cut_gesture(self, landmarks):

        (
            index_up,
            middle_up,
            ring_closed,
            pinky_closed
        ) = self.finger_states(landmarks)

        return (
            index_up
            and
            middle_up
            and
            ring_closed
            and
            pinky_closed
        )

    # =========================================================
    # FIST
    # =========================================================

    def is_fist(self, landmarks):

        index_closed = (
            landmarks[8].y >
            landmarks[6].y
        )

        middle_closed = (
            landmarks[12].y >
            landmarks[10].y
        )

        ring_closed = (
            landmarks[16].y >
            landmarks[14].y
        )

        pinky_closed = (
            landmarks[20].y >
            landmarks[18].y
        )

        return (
            index_closed
            and
            middle_closed
            and
            ring_closed
            and
            pinky_closed
        )

    # =========================================================
    # INDEX POSITION
    # =========================================================

    def get_index_position(
        self,
        landmarks,
        width,
        height
    ):

        x = int(
            landmarks[8].x * width
        )

        y = int(
            landmarks[8].y * height
        )

        return x, y

    # =========================================================
    # RECTANGLE CHECK
    # =========================================================

    def point_inside_rectangle(
        self,
        x,
        y
    ):

        if self.rectangle is None:
            return False

        x1, y1, x2, y2 = self.rectangle

        return (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        )

    # =========================================================
    # CREATE STRAIGHT RECTANGLE
    # =========================================================

    def make_rectangle(
        self,
        start,
        end
    ):

        x1, y1 = start
        x2, y2 = end

        return (
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2)
        )

    # =========================================================
    # PROCESS HAND
    #
    # IMPORTANT:
    # This function receives the ORIGINAL webcam frame.
    #
    # It does NOT draw anything on the frame.
    # =========================================================

    def process_hand(self, frame):

        height, width = frame.shape[:2]

        landmarks = self.detect_hand(frame)

        # -----------------------------------------------------
        # No hand detected
        # -----------------------------------------------------

        if landmarks is None:

            # Do NOT immediately cancel CUT mode.
            #
            # This prevents one-frame detection failures
            # from fixing the rectangle prematurely.

            if self.mode == "CUT":

                self.cut_release_counter += 1

                if (
                    self.cut_release_counter
                    >= self.cut_release_frames
                ):

                    self.finish_cut()

            return None

        # Hand detected again
        self.cut_release_counter = 0

        # -----------------------------------------------------
        # Finger position
        # -----------------------------------------------------

        index_x, index_y = self.get_index_position(
            landmarks,
            width,
            height
        )

        hand_x, hand_y = self.get_hand_center(
            landmarks,
            width,
            height
        )

        # -----------------------------------------------------
        # ✌️ CUT GESTURE
        # -----------------------------------------------------

        if self.is_cut_gesture(landmarks):

            self.mode = "CUT"

            # Start drawing
            if self.start_point is None:

                self.start_point = (
                    index_x,
                    index_y
                )

            # Continue drawing
            self.current_point = (
                index_x,
                index_y
            )

            return {
                "landmarks": landmarks,
                "gesture": "CUT",
                "position": (
                    index_x,
                    index_y
                )
            }

        # =====================================================
        # ✊ FIST
        # =====================================================

        if self.is_fist(landmarks):

            if self.rectangle is not None:

                # -------------------------------------------------
                # First fist frame
                # -------------------------------------------------

                if not self.dragging:

                    # Hand MUST be inside rectangle
                    if self.point_inside_rectangle(
                        hand_x,
                        hand_y
                    ):

                        self.dragging = True
                        self.cut_just_started = True

                        x1, y1, x2, y2 = self.rectangle

                        self.drag_offset_x = (
                            hand_x - x1
                        )

                        self.drag_offset_y = (
                            hand_y - y1
                        )

                        self.mode = "DRAGGING"

                # -------------------------------------------------
                # Already dragging
                # -------------------------------------------------

                if self.dragging:

                    x1, y1, x2, y2 = self.rectangle

                    rect_width = x2 - x1
                    rect_height = y2 - y1

                    new_x1 = (
                        hand_x -
                        self.drag_offset_x
                    )

                    new_y1 = (
                        hand_y -
                        self.drag_offset_y
                    )

                    self.rectangle = (
                        int(new_x1),
                        int(new_y1),
                        int(new_x1 + rect_width),
                        int(new_y1 + rect_height)
                    )

            return {
                "landmarks": landmarks,
                "gesture": "FIST",
                "position": (
                    hand_x,
                    hand_y
                )
            }


        # =====================================================
        # NO SPECIAL GESTURE
        # =====================================================

        # Opening the hand while dragging = DROP
        if self.dragging:

            self.dragging = False
            self.mode = "SELECTED"


        if self.mode == "CUT":

            self.cut_release_counter += 1

            if (
                self.cut_release_counter
                >= self.cut_release_frames
            ):

                self.finish_cut()


        return {
            "landmarks": landmarks,
            "gesture": "NORMAL",
            "position": (
                index_x,
                index_y
            )
        }

    # =========================================================
    # FINISH CUT
    # =========================================================

    def finish_cut(self):

        if (
            self.start_point is not None
            and
            self.current_point is not None
        ):

            rectangle = self.make_rectangle(
                self.start_point,
                self.current_point
            )

            x1, y1, x2, y2 = rectangle

            if (
                abs(x2 - x1) > 20
                and
                abs(y2 - y1) > 20
            ):

                self.rectangle = rectangle
                self.cut_source_rectangle = rectangle

                self.cut_active = True

                self.mode = "SELECTED"

        self.start_point = None
        self.current_point = None
        self.cut_release_counter = 0

    # =========================================================
    # APPLY ACTUAL CUT
    # =========================================================

    def apply_cut(self, frame):

        # -----------------------------------------------------
        # Start a new cut when fist grabs the rectangle
        # -----------------------------------------------------

        if (
            self.dragging
            and self.cut_just_started
            and self.rectangle is not None
        ):

            x1, y1, x2, y2 = self.rectangle

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # Make sure coordinates are ordered
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))

            # Clamp source rectangle to frame
            frame_h, frame_w = frame.shape[:2]

            sx1 = max(0, min(frame_w, x1))
            sy1 = max(0, min(frame_h, y1))
            sx2 = max(0, min(frame_w, x2))
            sy2 = max(0, min(frame_h, y2))

            if sx2 > sx1 and sy2 > sy1:

                # ---------------------------------------------
                # Save the selected image
                # ---------------------------------------------

                self.cut_patch = frame[
                    sy1:sy2,
                    sx1:sx2
                ].copy()

                # ---------------------------------------------
                # Remember ORIGINAL location
                # This location will stay black.
                # ---------------------------------------------

                self.cut_source_rectangle = (
                    sx1,
                    sy1,
                    sx2,
                    sy2
                )

                self.cut_active = True

            self.cut_just_started = False

        # -----------------------------------------------------
        # Nothing has been cut yet
        # -----------------------------------------------------

        if (
            not self.cut_active
            or self.cut_patch is None
            or self.cut_source_rectangle is None
            or self.rectangle is None
        ):
            return frame

        # -----------------------------------------------------
        # Create result from current frame
        # -----------------------------------------------------

        result = frame.copy()

        # -----------------------------------------------------
        # BLACK OUT ORIGINAL LOCATION
        # -----------------------------------------------------

        sx1, sy1, sx2, sy2 = self.cut_source_rectangle

        sx1 = max(0, min(frame.shape[1], sx1))
        sx2 = max(0, min(frame.shape[1], sx2))
        sy1 = max(0, min(frame.shape[0], sy1))
        sy2 = max(0, min(frame.shape[0], sy2))

        if sx2 > sx1 and sy2 > sy1:

            result[
                sy1:sy2,
                sx1:sx2
            ] = 0

        # -----------------------------------------------------
        # PASTE CUT PATCH AT CURRENT LOCATION
        # -----------------------------------------------------

        x1, y1, x2, y2 = self.rectangle

        x1 = int(x1)
        y1 = int(y1)

        patch_h, patch_w = self.cut_patch.shape[:2]

        x2 = x1 + patch_w
        y2 = y1 + patch_h

        frame_h, frame_w = result.shape[:2]

        # -----------------------------------------------------
        # Find visible portion of patch
        # -----------------------------------------------------

        dest_x1 = max(0, x1)
        dest_y1 = max(0, y1)

        dest_x2 = min(frame_w, x2)
        dest_y2 = min(frame_h, y2)

        if (
            dest_x2 > dest_x1
            and dest_y2 > dest_y1
        ):

            # Corresponding source coordinates
            src_x1 = dest_x1 - x1
            src_y1 = dest_y1 - y1

            src_x2 = src_x1 + (
                dest_x2 - dest_x1
            )

            src_y2 = src_y1 + (
                dest_y2 - dest_y1
            )

            result[
                dest_y1:dest_y2,
                dest_x1:dest_x2
            ] = self.cut_patch[
                src_y1:src_y2,
                src_x1:src_x2
            ]

        return result

    # =========================================================
    # DRAW FINAL OVERLAY
    #
    # This is called AFTER:
    #
    #     Human segmentation
    #     Video background
    #     Person + video compositing
    #
    # Therefore the rectangle is the HIGHEST layer.
    # =========================================================

    def draw_overlay(self, frame):

        # -----------------------------------------------------
        # Draw rectangle currently being created
        # -----------------------------------------------------

        if (
            self.mode == "CUT"
            and
            self.start_point is not None
            and
            self.current_point is not None
        ):

            rectangle = self.make_rectangle(
                self.start_point,
                self.current_point
            )

            self.draw_glowing_rectangle(
                frame,
                rectangle
            )

        # -----------------------------------------------------
        # Draw completed rectangle
        # -----------------------------------------------------

        elif self.rectangle is not None:

            self.draw_glowing_rectangle(
                frame,
                self.rectangle
            )

        return frame

    # =========================================================
    # GLOWING RECTANGLE
    # =========================================================

    def draw_glowing_rectangle(
        self,
        frame,
        rectangle
    ):

        x1, y1, x2, y2 = rectangle

        # Clamp coordinates
        x1 = max(
            0,
            min(frame.shape[1] - 1, int(x1))
        )

        y1 = max(
            0,
            min(frame.shape[0] - 1, int(y1))
        )

        x2 = max(
            0,
            min(frame.shape[1] - 1, int(x2))
        )

        y2 = max(
            0,
            min(frame.shape[0] - 1, int(y2))
        )

        # -----------------------------------------------------
        # Glow
        # -----------------------------------------------------

        glow = np.zeros_like(frame)

        cv2.rectangle(
            glow,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            6
        )

        glow = cv2.GaussianBlur(
            glow,
            (21, 21),
            0
        )

        frame[:] = cv2.addWeighted(
            frame,
            1.0,
            glow,
            0.8,
            0
        )

        # -----------------------------------------------------
        # Sharp border
        # -----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):

        self.mode = "NORMAL"

        self.start_point = None
        self.current_point = None

        self.rectangle = None

        self.dragging = False

        self.drag_offset_x = 0
        self.drag_offset_y = 0

        self.cut_release_counter = 0

        self.cut_source_rectangle = None
        self.cut_active = False
        self.cut_just_started = False
    # =========================================================
    # GET HAND CENTER
    # =========================================================

    def get_hand_center(
        self,
        landmarks,
        width,
        height
    ):

        # Use several palm landmarks instead of
        # the fingertip.

        points = [
            landmarks[0],   # Wrist
            landmarks[5],   # Index MCP
            landmarks[9],   # Middle MCP
            landmarks[13],  # Ring MCP
            landmarks[17]   # Pinky MCP
        ]

        x = int(
            sum(point.x for point in points)
            / len(points)
            * width
        )

        y = int(
            sum(point.y for point in points)
            / len(points)
            * height
        )

        return x, y

    # =========================================================
    # APPLY LIVE VIDEO CUT
    #
    # Layer order:
    #
    #     HUMAN
    #       ↑
    #     CUT VIDEO
    #       ↑
    #     BACKGROUND VIDEO
    #
    # The cut region is taken from the CURRENT video frame,
    # so the video continues playing.
    # =========================================================

    def apply_cut(
        self,
        video_frame,
        human_frame,
        human_mask
    ):

        result = video_frame.copy()

        # -----------------------------------------------------
        # No cut exists yet
        # -----------------------------------------------------

        if (
            not self.cut_active
            or self.cut_source_rectangle is None
            or self.rectangle is None
        ):

            # Normal composition:
            # video + human
            mask_float = (
                human_mask.astype(np.float32) / 255.0
            )

            if mask_float.ndim == 2:
                mask_float = mask_float[:, :, None]

            result = (
                human_frame.astype(np.float32)
                * mask_float
                +
                video_frame.astype(np.float32)
                * (1.0 - mask_float)
            ).astype(np.uint8)

            return result

        # =====================================================
        # 1. NORMAL VIDEO BACKGROUND
        # =====================================================

        result = video_frame.copy()

        # =====================================================
        # 2. BLACK OUT ORIGINAL CUT LOCATION
        # =====================================================

        sx1, sy1, sx2, sy2 = (
            self.cut_source_rectangle
        )

        sx1 = int(sx1)
        sy1 = int(sy1)
        sx2 = int(sx2)
        sy2 = int(sy2)

        frame_h, frame_w = video_frame.shape[:2]

        # Clamp
        sx1 = max(0, min(frame_w, sx1))
        sx2 = max(0, min(frame_w, sx2))
        sy1 = max(0, min(frame_h, sy1))
        sy2 = max(0, min(frame_h, sy2))

        if sx2 > sx1 and sy2 > sy1:

            result[
                sy1:sy2,
                sx1:sx2
            ] = 0

        # =====================================================
        # 3. GET CURRENT VIDEO REGION
        # =====================================================

        video_patch = video_frame[
            sy1:sy2,
            sx1:sx2
        ].copy()

        if video_patch.size == 0:
            return result

        patch_h, patch_w = video_patch.shape[:2]

        # =====================================================
        # 4. MOVE VIDEO PATCH
        # =====================================================

        x1, y1, x2, y2 = self.rectangle

        x1 = int(x1)
        y1 = int(y1)

        # Keep original size
        x2 = x1 + patch_w
        y2 = y1 + patch_h

        # -----------------------------------------------------
        # Destination clipping
        # -----------------------------------------------------

        dest_x1 = max(0, x1)
        dest_y1 = max(0, y1)

        dest_x2 = min(frame_w, x2)
        dest_y2 = min(frame_h, y2)

        if (
            dest_x2 > dest_x1
            and
            dest_y2 > dest_y1
        ):

            src_x1 = dest_x1 - x1
            src_y1 = dest_y1 - y1

            src_x2 = src_x1 + (
                dest_x2 - dest_x1
            )

            src_y2 = src_y1 + (
                dest_y2 - dest_y1
            )

            result[
                dest_y1:dest_y2,
                dest_x1:dest_x2
            ] = video_patch[
                src_y1:src_y2,
                src_x1:src_x2
            ]

        # =====================================================
        # 5. PUT HUMAN ON TOP
        # =====================================================

        mask_float = (
            human_mask.astype(np.float32) / 255.0
        )

        if mask_float.ndim == 2:
            mask_float = mask_float[:, :, None]

        result = (
            human_frame.astype(np.float32)
            * mask_float
            +
            result.astype(np.float32)
            * (1.0 - mask_float)
        ).astype(np.uint8)

        return result

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):

        self.detector.close()