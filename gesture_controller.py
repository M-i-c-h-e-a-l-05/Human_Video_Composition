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

        # =====================================================
        # CURRENT INTERACTION STATE
        # =====================================================

        self.mode = "NORMAL"

        # =====================================================
        # RECTANGLE DRAWING
        # =====================================================

        self.start_point = None
        self.current_point = None
        self.rectangle = None

        # =====================================================
        # FIST DRAGGING
        # =====================================================

        self.dragging = False

        self.drag_offset_x = 0
        self.drag_offset_y = 0

        # =====================================================
        # ACTUAL CUT-OUT DATA
        # =====================================================

        self.cut_source_rectangle = None
        self.cut_active = False

        # =====================================================
        # GESTURE STABILITY
        # =====================================================

        self.cut_release_frames = 5
        self.cut_release_counter = 0

        # =====================================================
        # TRASH BIN
        # =====================================================

        self.trash_width = 90
        self.trash_height = 90
        self.trash_margin = 20

        self.trash_rectangle = None
        self.over_trash = False

        # =====================================================
        # FIX BUTTON
        #
        # Sits to the left of the trash bin. Hovering the hand
        # over it while a cut exists LOCKS the cut so a new
        # one cannot be started. Only deleting the cut (trash)
        # clears the lock.
        # =====================================================

        self.fix_width = 90
        self.fix_height = 90
        self.fix_gap = 20

        self.fix_rectangle = None
        self.over_fix = False

        self.cut_locked = False

        # =====================================================
        # RESIZE
        # =====================================================

        # Minimum displayed cut size
        self.resize_min_size = 50

        # Maximum size relative to original cut
        self.resize_max_scale = 2.5

        # Previous thumb-index distance (smoothed each frame)
        self.last_resize_distance = None

        # Thumb-index distance at the moment the pinch started.
        # Growth/shrink is measured against this fixed baseline,
        # not frame-to-frame, so holding the fingers apart keeps
        # growing the cut without needing to keep moving them.
        self.resize_start_distance = None

        # Smoothing prevents jitter
        self.resize_smoothing = 0.15

        # How far past the baseline (pixels) before it starts
        # resizing at all — keeps a small resting wobble from
        # doing anything.
        self.resize_dead_zone = 15

        # How many pixels of width are added/removed per frame
        # for each pixel the fingers are held past the dead zone.
        self.resize_speed = 0.06

        # =====================================================
        # DRAG RELEASE DEBOUNCE
        #
        # Ending a fist/resize hold on a single ambiguous frame
        # (e.g. mid-transition between the two hand shapes)
        # would drop the cut too easily. Require several
        # consecutive "neither" frames first, same idea as
        # cut_release_frames above.
        # =====================================================

        self.drag_release_frames = 5
        self.drag_release_counter = 0

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

        # -----------------------------------------------------
        # Index
        # -----------------------------------------------------

        index_up = (
            landmarks[8].y <
            landmarks[6].y
        )

        # -----------------------------------------------------
        # Middle
        # -----------------------------------------------------

        middle_up = (
            landmarks[12].y <
            landmarks[10].y
        )

        # -----------------------------------------------------
        # Ring
        # -----------------------------------------------------

        ring_closed = (
            landmarks[16].y >
            landmarks[14].y
        )

        # -----------------------------------------------------
        # Pinky
        # -----------------------------------------------------

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
    # RESIZE GESTURE
    #
    # Only:
    #
    #     THUMB  = OPEN
    #     INDEX  = OPEN
    #     MIDDLE = CLOSED
    #     RING   = CLOSED
    #     PINKY  = CLOSED
    #
    # This gesture is ONLY processed while dragging.
    # =========================================================

    def is_resize_gesture(self, landmarks):

        # -----------------------------------------------------
        # THUMB OPEN
        #
        # Instead of checking X direction, compare the
        # thumb-tip distance from wrist with thumb-IP distance.
        #
        # This works better for either left or right hand.
        # -----------------------------------------------------

        wrist_x = landmarks[0].x
        wrist_y = landmarks[0].y

        thumb_tip_x = landmarks[4].x
        thumb_tip_y = landmarks[4].y

        thumb_ip_x = landmarks[3].x
        thumb_ip_y = landmarks[3].y

        thumb_tip_distance = np.sqrt(
            (thumb_tip_x - wrist_x) ** 2
            +
            (thumb_tip_y - wrist_y) ** 2
        )

        thumb_ip_distance = np.sqrt(
            (thumb_ip_x - wrist_x) ** 2
            +
            (thumb_ip_y - wrist_y) ** 2
        )

        thumb_open = (
            thumb_tip_distance >
            thumb_ip_distance * 1.10
        )

        # -----------------------------------------------------
        # INDEX OPEN
        # -----------------------------------------------------

        index_open = (
            landmarks[8].y <
            landmarks[6].y
        )

        # -----------------------------------------------------
        # MIDDLE CLOSED
        # -----------------------------------------------------

        middle_closed = (
            landmarks[12].y >
            landmarks[10].y
        )

        # -----------------------------------------------------
        # RING CLOSED
        # -----------------------------------------------------

        ring_closed = (
            landmarks[16].y >
            landmarks[14].y
        )

        # -----------------------------------------------------
        # PINKY CLOSED
        # -----------------------------------------------------

        pinky_closed = (
            landmarks[20].y >
            landmarks[18].y
        )

        return (
            thumb_open
            and
            index_open
            and
            middle_closed
            and
            ring_closed
            and
            pinky_closed
        )

    # =========================================================
    # THUMB ↔ INDEX DISTANCE
    # =========================================================

    def get_resize_distance(
        self,
        landmarks,
        width,
        height
    ):

        thumb_x = landmarks[4].x * width
        thumb_y = landmarks[4].y * height

        index_x = landmarks[8].x * width
        index_y = landmarks[8].y * height

        distance = np.sqrt(
            (index_x - thumb_x) ** 2
            +
            (index_y - thumb_y) ** 2
        )

        return distance

    # =========================================================
    # RESIZE CUT
    #
    # IMPORTANT:
    #
    # This function does NOTHING unless the cut is currently
    # grabbed.
    #
    # cut_source_rectangle stays unchanged.
    # Only self.rectangle changes.
    # =========================================================

    def resize_cut(
        self,
        landmarks,
        width,
        height
    ):

        # -----------------------------------------------------
        # SAFETY CHECK
        # -----------------------------------------------------

        if not self.dragging:

            self.last_resize_distance = None
            self.resize_start_distance = None

            return

        if self.rectangle is None:

            self.last_resize_distance = None
            self.resize_start_distance = None

            return

        # -----------------------------------------------------
        # Check resize gesture
        # -----------------------------------------------------

        if not self.is_resize_gesture(landmarks):

            self.last_resize_distance = None
            self.resize_start_distance = None

            return

        # -----------------------------------------------------
        # Current thumb-index distance
        # -----------------------------------------------------

        distance = self.get_resize_distance(
            landmarks,
            width,
            height
        )

        # -----------------------------------------------------
        # First resize frame
        #
        # We don't resize immediately.
        # We remember this distance as the baseline — how far
        # apart the fingers are from now on is measured against
        # THIS, not against last frame. That's what lets holding
        # the fingers apart keep growing the cut on its own.
        # -----------------------------------------------------

        if self.resize_start_distance is None:

            self.resize_start_distance = distance
            self.last_resize_distance = distance

            return

        # -----------------------------------------------------
        # Smooth the distance (jitter only, not the resize speed)
        # -----------------------------------------------------

        self.last_resize_distance = (
            self.last_resize_distance
            +
            (
                distance
                -
                self.last_resize_distance
            )
            *
            self.resize_smoothing
        )

        # -----------------------------------------------------
        # Offset from baseline
        #
        # Positive = fingers held wider than the start = GROW
        # Negative = fingers held narrower than the start = SHRINK
        # -----------------------------------------------------

        offset = (
            self.last_resize_distance
            -
            self.resize_start_distance
        )

        # Small resting wobble around the baseline does nothing
        if abs(offset) < self.resize_dead_zone:
            return

        # =====================================================
        # CURRENT RECTANGLE
        # =====================================================

        x1, y1, x2, y2 = self.rectangle

        current_width = x2 - x1
        current_height = y2 - y1

        if (
            current_width <= 0
            or
            current_height <= 0
        ):

            return

        # =====================================================
        # CHANGE SIZE
        #
        # Continuous step every frame the pose is held past the
        # dead zone — not tied to the fingers still moving.
        # =====================================================

        scale_change = (
            offset * self.resize_speed
        )

        new_width = (
            current_width
            +
            scale_change
        )

        # Preserve aspect ratio
        aspect_ratio = (
            current_height
            /
            current_width
        )

        new_height = (
            new_width
            *
            aspect_ratio
        )

        # =====================================================
        # MINIMUM SIZE
        # =====================================================

        if new_width < self.resize_min_size:

            new_width = (
                self.resize_min_size
            )

            new_height = (
                new_width
                *
                aspect_ratio
            )

        # =====================================================
        # MAXIMUM SIZE
        # =====================================================

        if self.cut_source_rectangle is not None:

            sx1, sy1, sx2, sy2 = (
                self.cut_source_rectangle
            )

            source_width = (
                sx2 - sx1
            )

            max_width = (
                source_width
                *
                self.resize_max_scale
            )

            if new_width > max_width:

                new_width = max_width

                new_height = (
                    new_width
                    *
                    aspect_ratio
                )

        # =====================================================
        # KEEP CENTER FIXED
        # =====================================================

        center_x = (
            x1 + x2
        ) / 2

        center_y = (
            y1 + y2
        ) / 2

        new_x1 = (
            center_x
            -
            new_width / 2
        )

        new_y1 = (
            center_y
            -
            new_height / 2
        )

        new_x2 = (
            center_x
            +
            new_width / 2
        )

        new_y2 = (
            center_y
            +
            new_height / 2
        )

        self.rectangle = (
            int(new_x1),
            int(new_y1),
            int(new_x2),
            int(new_y2)
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
            landmarks[8].x
            *
            width
        )

        y = int(
            landmarks[8].y
            *
            height
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

        x1, y1, x2, y2 = (
            self.rectangle
        )

        return (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        )

    # =========================================================
    # TRASH BIN RECTANGLE
    # =========================================================

    def get_trash_rectangle(
        self,
        width,
        height
    ):

        x2 = (
            width
            -
            self.trash_margin
        )

        x1 = (
            x2
            -
            self.trash_width
        )

        y1 = self.trash_margin

        y2 = (
            y1
            +
            self.trash_height
        )

        return (
            x1,
            y1,
            x2,
            y2
        )

    # =========================================================
    # POINT INSIDE TRASH
    # =========================================================

    def point_inside_trash(
        self,
        x,
        y
    ):

        if self.trash_rectangle is None:

            return False

        x1, y1, x2, y2 = (
            self.trash_rectangle
        )

        return (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        )

    # =========================================================
    # FIX BUTTON RECTANGLE (top-left corner)
    # =========================================================

    def get_fix_rectangle(
        self,
        width,
        height
    ):

        x1 = self.fix_gap

        x2 = (
            x1
            +
            self.fix_width
        )

        y1 = self.fix_gap

        y2 = (
            y1
            +
            self.fix_height
        )

        return (
            x1,
            y1,
            x2,
            y2
        )

    # =========================================================
    # POINT INSIDE FIX BUTTON
    # =========================================================

    def point_inside_fix(
        self,
        x,
        y
    ):

        if self.fix_rectangle is None:

            return False

        x1, y1, x2, y2 = (
            self.fix_rectangle
        )

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
    #
    # This function receives the ORIGINAL webcam frame.
    #
    # It does NOT draw anything on the frame.
    # =========================================================

    def process_hand(self, frame):

        height, width = (
            frame.shape[:2]
        )

        # -----------------------------------------------------
        # Update trash position
        # -----------------------------------------------------

        self.trash_rectangle = (
            self.get_trash_rectangle(
                width,
                height
            )
        )

        # -----------------------------------------------------
        # Update fix button position
        # -----------------------------------------------------

        self.fix_rectangle = (
            self.get_fix_rectangle(
                width,
                height
            )
        )

        # -----------------------------------------------------
        # Detect hand
        # -----------------------------------------------------

        landmarks = self.detect_hand(
            frame
        )

        # =====================================================
        # NO HAND
        # =====================================================

        if landmarks is None:

            # Do not immediately finish CUT.
            #
            # This prevents one-frame detection failures
            # from prematurely fixing the rectangle.

            if self.mode == "CUT":

                self.cut_release_counter += 1

                if (
                    self.cut_release_counter
                    >=
                    self.cut_release_frames
                ):

                    self.finish_cut()

            return None

        # Hand detected again
        self.cut_release_counter = 0

        # =====================================================
        # POSITIONS
        # =====================================================

        index_x, index_y = (
            self.get_index_position(
                landmarks,
                width,
                height
            )
        )

        hand_x, hand_y = (
            self.get_hand_center(
                landmarks,
                width,
                height
            )
        )

        # =====================================================
        # FIX BUTTON (place hand over to lock the current cut)
        #
        # Only relevant while a cut exists and isn't locked
        # yet. Once locked, only discard_cut() (trash) clears
        # it, so this never re-checks after that.
        # =====================================================

        self.over_fix = self.point_inside_fix(
            hand_x,
            hand_y
        )

        if (
            self.cut_active
            and
            not self.cut_locked
            and
            self.over_fix
        ):

            self.cut_locked = True

        # =====================================================
        # ✌️ CUT GESTURE
        # =====================================================

        if (
            self.is_cut_gesture(
                landmarks
            )
            and
            not self.cut_locked
        ):

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
        # ✊ FIST / GRAB
        # =====================================================

        if self.is_fist(
            landmarks
        ):

            if self.rectangle is not None:

                # -------------------------------------------------
                # FIRST FIST FRAME
                # -------------------------------------------------

                if not self.dragging:

                    # Hand MUST be inside rectangle
                    if self.point_inside_rectangle(
                        hand_x,
                        hand_y
                    ):

                        self.dragging = True

                        x1, y1, x2, y2 = (
                            self.rectangle
                        )

                        self.drag_offset_x = (
                            hand_x
                            -
                            x1
                        )

                        self.drag_offset_y = (
                            hand_y
                            -
                            y1
                        )

                        self.mode = "DRAGGING"

                        # Start fresh if resize happens
                        self.last_resize_distance = None
                        self.resize_start_distance = None

                        # Confirmed hold, cancel any pending release
                        self.drag_release_counter = 0

                # -------------------------------------------------
                # ALREADY DRAGGING
                # -------------------------------------------------

                if self.dragging:

                    x1, y1, x2, y2 = (
                        self.rectangle
                    )

                    rect_width = (
                        x2 - x1
                    )

                    rect_height = (
                        y2 - y1
                    )

                    new_x1 = (
                        hand_x
                        -
                        self.drag_offset_x
                    )

                    new_y1 = (
                        hand_y
                        -
                        self.drag_offset_y
                    )

                    self.rectangle = (
                        int(new_x1),
                        int(new_y1),
                        int(
                            new_x1
                            +
                            rect_width
                        ),
                        int(
                            new_y1
                            +
                            rect_height
                        )
                    )

                    self.over_trash = (
                        self.point_inside_trash(
                            hand_x,
                            hand_y
                        )
                    )

                    # We are moving with fist.
                    # Reset resize reference.
                    self.last_resize_distance = None
                    self.resize_start_distance = None

                    # Confirmed hold, cancel any pending release
                    self.drag_release_counter = 0

            return {
                "landmarks": landmarks,
                "gesture": "FIST",
                "position": (
                    hand_x,
                    hand_y
                )
            }

        # =====================================================
        # RESIZE
        #
        # IMPORTANT:
        #
        # This section is reached ONLY if the previous FIST
        # block did not run.
        #
        # Therefore resize is only possible while:
        #
        #     self.dragging == True
        #
        # =====================================================

        if self.dragging:

            if self.is_resize_gesture(
                landmarks
            ):

                self.resize_cut(
                    landmarks,
                    width,
                    height
                )

                # Resize gesture should not interact
                # with trash.
                self.over_trash = False

                # Confirmed hold, cancel any pending release
                self.drag_release_counter = 0

                return {
                    "landmarks": landmarks,
                    "gesture": "RESIZE",
                    "position": (
                        hand_x,
                        hand_y
                    )
                }

            else:

                # Gesture changed.
                # Prepare for a fresh resize gesture.
                self.last_resize_distance = None
                self.resize_start_distance = None

        # =====================================================
        # NO SPECIAL GESTURE
        # =====================================================

        # -----------------------------------------------------
        # Opening hand while dragging = DROP
        # -----------------------------------------------------

        if self.dragging:

            self.drag_release_counter += 1

            if (
                self.drag_release_counter
                >=
                self.drag_release_frames
            ):

                self.dragging = False

                self.drag_release_counter = 0

                # Stop resize mode
                self.last_resize_distance = None
                self.resize_start_distance = None

                # -------------------------------------------------
                # Drop on trash
                # -------------------------------------------------

                if self.over_trash:

                    self.discard_cut()

                # -------------------------------------------------
                # Normal drop
                # -------------------------------------------------

                else:

                    self.mode = "SELECTED"

                self.over_trash = False

        # =====================================================
        # FINISH DRAWING CUT
        # =====================================================

        if self.mode == "CUT":

            self.cut_release_counter += 1

            if (
                self.cut_release_counter
                >=
                self.cut_release_frames
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

            x1, y1, x2, y2 = (
                rectangle
            )

            if (
                abs(x2 - x1) > 20
                and
                abs(y2 - y1) > 20
            ):

                self.rectangle = rectangle

                self.cut_source_rectangle = (
                    rectangle
                )

                self.cut_active = True

                self.mode = "SELECTED"

        self.start_point = None
        self.current_point = None

        self.cut_release_counter = 0

        self.last_resize_distance = None
        self.resize_start_distance = None

    # =========================================================
    # DRAW FINAL OVERLAY
    #
    # Layer order:
    #
    #     HUMAN
    #       ↑
    #     CUT VIDEO
    #       ↑
    #     BACKGROUND VIDEO
    #       ↑
    #     OVERLAY
    #
    # The rectangle/trash are therefore drawn last.
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

        # -----------------------------------------------------
        # Draw trash bin
        # -----------------------------------------------------

        if self.cut_active:

            self.draw_trash_bin(
                frame,
                highlighted=(
                    self.dragging
                    and
                    self.over_trash
                )
            )

            self.draw_fix_button(
                frame,
                highlighted=self.over_fix,
                locked=self.cut_locked
            )

        return frame

    # =========================================================
    # TRASH BIN ICON
    # =========================================================

    def draw_trash_bin(
        self,
        frame,
        highlighted
    ):

        if self.trash_rectangle is None:
            return

        x1, y1, x2, y2 = (
            self.trash_rectangle
        )

        # Red when hovering
        color = (
            (0, 0, 255)
            if highlighted
            else
            (200, 200, 200)
        )

        thickness = (
            3
            if highlighted
            else
            2
        )

        # -----------------------------------------------------
        # Bin body
        # -----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

        # -----------------------------------------------------
        # Lid
        # -----------------------------------------------------

        lid_y = (
            y1
            +
            int(
                self.trash_height
                *
                0.18
            )
        )

        cv2.line(
            frame,
            (x1 - 6, lid_y),
            (x2 + 6, lid_y),
            color,
            thickness
        )

        # -----------------------------------------------------
        # Handle
        # -----------------------------------------------------

        handle_w = int(
            self.trash_width
            *
            0.4
        )

        handle_x = (
            x1
            +
            (
                self.trash_width
                -
                handle_w
            )
            //
            2
        )

        cv2.rectangle(
            frame,
            (
                handle_x,
                y1 - 8
            ),
            (
                handle_x + handle_w,
                lid_y
            ),
            color,
            thickness
        )

        # -----------------------------------------------------
        # Vertical ribs
        # -----------------------------------------------------

        for fraction in (
            0.3,
            0.5,
            0.7
        ):

            rib_x = (
                x1
                +
                int(
                    self.trash_width
                    *
                    fraction
                )
            )

            cv2.line(
                frame,
                (
                    rib_x,
                    lid_y + 6
                ),
                (
                    rib_x,
                    y2 - 6
                ),
                color,
                thickness
            )

    # =========================================================
    # FIX BUTTON ICON (padlock)
    # =========================================================

    def draw_fix_button(
        self,
        frame,
        highlighted,
        locked
    ):

        if self.fix_rectangle is None:
            return

        x1, y1, x2, y2 = (
            self.fix_rectangle
        )

        # Green once locked, yellow while hovering, gray otherwise
        if locked:

            color = (0, 200, 0)

        elif highlighted:

            color = (0, 255, 255)

        else:

            color = (200, 200, 200)

        thickness = (
            3
            if (highlighted or locked)
            else
            2
        )

        # -----------------------------------------------------
        # Padlock body
        # -----------------------------------------------------

        body_top = (
            y1
            +
            int(
                self.fix_height
                *
                0.4
            )
        )

        cv2.rectangle(
            frame,
            (x1, body_top),
            (x2, y2),
            color,
            thickness
        )

        # -----------------------------------------------------
        # Shackle
        # -----------------------------------------------------

        shackle_w = int(
            self.fix_width
            *
            0.5
        )

        shackle_x1 = (
            x1
            +
            (
                self.fix_width
                -
                shackle_w
            )
            //
            2
        )

        shackle_x2 = (
            shackle_x1
            +
            shackle_w
        )

        cv2.ellipse(
            frame,
            (
                (shackle_x1 + shackle_x2) // 2,
                body_top
            ),
            (
                shackle_w // 2,
                int(
                    self.fix_height
                    *
                    0.28
                )
            ),
            0,
            180,
            360,
            color,
            thickness
        )

        # -----------------------------------------------------
        # Keyhole
        # -----------------------------------------------------

        keyhole_x = (x1 + x2) // 2

        keyhole_y = (
            body_top
            +
            int(
                (y2 - body_top)
                *
                0.5
            )
        )

        cv2.circle(
            frame,
            (keyhole_x, keyhole_y),
            4,
            color,
            -1
        )

    # =========================================================
    # GLOWING RECTANGLE
    # =========================================================

    def draw_glowing_rectangle(
        self,
        frame,
        rectangle
    ):

        x1, y1, x2, y2 = (
            rectangle
        )

        # -----------------------------------------------------
        # Clamp coordinates
        # -----------------------------------------------------

        x1 = max(
            0,
            min(
                frame.shape[1] - 1,
                int(x1)
            )
        )

        y1 = max(
            0,
            min(
                frame.shape[0] - 1,
                int(y1)
            )
        )

        x2 = max(
            0,
            min(
                frame.shape[1] - 1,
                int(x2)
            )
        )

        y2 = max(
            0,
            min(
                frame.shape[0] - 1,
                int(y2)
            )
        )

        # =====================================================
        # GLOW
        # =====================================================

        glow = np.zeros_like(
            frame
        )

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

        # =====================================================
        # SHARP BORDER
        # =====================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

    # =========================================================
    # DISCARD CUT
    #
    # When dropped on trash:
    #
    #     cut_active = False
    #
    # apply_cut() automatically returns to:
    #
    #     VIDEO + HUMAN
    # =========================================================

    def discard_cut(self):

        self.rectangle = None

        self.cut_source_rectangle = None

        self.cut_active = False

        self.dragging = False

        self.drag_offset_x = 0
        self.drag_offset_y = 0

        self.start_point = None
        self.current_point = None

        self.over_trash = False

        self.last_resize_distance = None
        self.resize_start_distance = None

        self.drag_release_counter = 0

        self.over_fix = False

        self.cut_locked = False

        self.mode = "NORMAL"

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

        self.over_trash = False

        self.last_resize_distance = None
        self.resize_start_distance = None

        self.drag_release_counter = 0

        self.over_fix = False

        self.cut_locked = False

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
            sum(
                point.x
                for point in points
            )
            /
            len(points)
            *
            width
        )

        y = int(
            sum(
                point.y
                for point in points
            )
            /
            len(points)
            *
            height
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
    # The cut region comes from the CURRENT video frame.
    #
    # Therefore the video continues playing normally.
    # =========================================================

    def apply_cut(
        self,
        video_frame,
        human_frame,
        human_mask
    ):

        result = video_frame.copy()

        # =====================================================
        # NO CUT
        # =====================================================

        if (
            not self.cut_active
            or
            self.cut_source_rectangle is None
            or
            self.rectangle is None
        ):

            # -------------------------------------------------
            # Normal composition:
            #
            # HUMAN
            #   +
            # VIDEO
            # -------------------------------------------------

            mask_float = (
                human_mask.astype(
                    np.float32
                )
                /
                255.0
            )

            if mask_float.ndim == 2:

                mask_float = (
                    mask_float[:, :, None]
                )

            result = (
                human_frame.astype(
                    np.float32
                )
                *
                mask_float
                +
                video_frame.astype(
                    np.float32
                )
                *
                (
                    1.0
                    -
                    mask_float
                )
            ).astype(
                np.uint8
            )

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

        frame_h, frame_w = (
            video_frame.shape[:2]
        )

        # -----------------------------------------------------
        # Clamp
        # -----------------------------------------------------

        sx1 = max(
            0,
            min(
                frame_w,
                sx1
            )
        )

        sx2 = max(
            0,
            min(
                frame_w,
                sx2
            )
        )

        sy1 = max(
            0,
            min(
                frame_h,
                sy1
            )
        )

        sy2 = max(
            0,
            min(
                frame_h,
                sy2
            )
        )

        # -----------------------------------------------------
        # Black out source
        # -----------------------------------------------------

        if (
            sx2 > sx1
            and
            sy2 > sy1
        ):

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

        patch_h, patch_w = (
            video_patch.shape[:2]
        )

        # =====================================================
        # 4. MOVE / RESIZE VIDEO PATCH
        # =====================================================

        x1, y1, x2, y2 = (
            self.rectangle
        )

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # -----------------------------------------------------
        # Desired destination size
        #
        # This is different from the source size because
        # the user can resize the cut.
        # -----------------------------------------------------

        dest_width = (
            x2 - x1
        )

        dest_height = (
            y2 - y1
        )

        if (
            dest_width <= 0
            or
            dest_height <= 0
        ):

            return result

        # =====================================================
        # RESIZE CURRENT VIDEO PATCH
        # =====================================================

        resized_patch = cv2.resize(
            video_patch,
            (
                dest_width,
                dest_height
            ),
            interpolation=cv2.INTER_LINEAR
        )

        # =====================================================
        # 5. DESTINATION CLIPPING
        # =====================================================

        dest_x1 = max(
            0,
            x1
        )

        dest_y1 = max(
            0,
            y1
        )

        dest_x2 = min(
            frame_w,
            x2
        )

        dest_y2 = min(
            frame_h,
            y2
        )

        if (
            dest_x2 > dest_x1
            and
            dest_y2 > dest_y1
        ):

            # -------------------------------------------------
            # Corresponding source region from resized patch
            # -------------------------------------------------

            src_x1 = (
                dest_x1
                -
                x1
            )

            src_y1 = (
                dest_y1
                -
                y1
            )

            src_x2 = (
                src_x1
                +
                (
                    dest_x2
                    -
                    dest_x1
                )
            )

            src_y2 = (
                src_y1
                +
                (
                    dest_y2
                    -
                    dest_y1
                )
            )

            result[
                dest_y1:dest_y2,
                dest_x1:dest_x2
            ] = resized_patch[
                src_y1:src_y2,
                src_x1:src_x2
            ]

        # =====================================================
        # 6. PUT HUMAN ON TOP
        # =====================================================

        mask_float = (
            human_mask.astype(
                np.float32
            )
            /
            255.0
        )

        if mask_float.ndim == 2:

            mask_float = (
                mask_float[:, :, None]
            )

        result = (
            human_frame.astype(
                np.float32
            )
            *
            mask_float
            +
            result.astype(
                np.float32
            )
            *
            (
                1.0
                -
                mask_float
            )
        ).astype(
            np.uint8
        )

        return result

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):

        self.detector.close()