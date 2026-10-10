import cv2
import time


class CutController:

    def __init__(self):

        # ============================================================
        # CUT / POINT SELECTION
        # ============================================================

        self.start_point = None
        self.current_point = None

        self.point1 = None
        self.point2 = None

        self.rectangle = None

        # Original source area for the live cut
        self.cut_source_rectangle = None

        self.cut_active = False
        self.mode = "NORMAL"

        # ============================================================
        # STABLE POINT SELECTION
        # ============================================================

        self.hold_time = 2.0

        # Maximum allowed movement while holding
        self.stability_threshold = 15

        self.stable_start_time = None
        self.stable_reference_point = None

        # POINT_1 -> waiting for first point
        # POINT_2 -> waiting for second point
        # COMPLETE -> rectangle created
        self.point_selection_state = "POINT_1"

        self.hold_progress = 0.0

        # ============================================================
        # DRAGGING
        # ============================================================

        self.dragging = False

        self.drag_offset_x = 0
        self.drag_offset_y = 0

        self.drag_release_frames = 5
        self.drag_release_counter = 0

        # ============================================================
        # TRASH
        # ============================================================

        self.trash_width = 90
        self.trash_height = 90
        self.trash_margin = 20

        self.trash_rectangle = None
        self.over_trash = False

        # ============================================================
        # FIX / LOCK
        # ============================================================

        self.fix_width = 90
        self.fix_height = 90
        self.fix_gap = 20

        self.fix_rectangle = None
        self.over_fix = False

        self.cut_locked = False

        # ============================================================
        # RESIZE
        # ============================================================

        self.min_size = 50
        self.max_scale = 2.5

        self.resize_smoothing = 0.15
        self.resize_dead_zone = 15
        self.resize_speed = 0.06

        self.last_resize_distance = None
        self.resize_start_distance = None

    # ================================================================
    # RECTANGLE
    # ================================================================

    def make_rectangle(self, point1, point2):

        x1 = min(
            point1[0],
            point2[0]
        )

        y1 = min(
            point1[1],
            point2[1]
        )

        x2 = max(
            point1[0],
            point2[0]
        )

        y2 = max(
            point1[1],
            point2[1]
        )

        return (
            int(x1),
            int(y1),
            int(x2),
            int(y2)
        )

    # ================================================================
    # POINT INSIDE RECTANGLE
    # ================================================================

    def point_inside_rectangle(self, x, y):

        if self.rectangle is None:
            return False

        x1, y1, x2, y2 = self.rectangle

        return (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        )

    # ================================================================
    # POINT INSIDE TRASH
    # ================================================================

    def point_inside_trash(self, x, y):

        if self.trash_rectangle is None:
            return False

        x1, y1, x2, y2 = self.trash_rectangle

        return (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        )

    # ================================================================
    # POINT INSIDE FIX
    # ================================================================

    def point_inside_fix(self, x, y):

        if self.fix_rectangle is None:
            return False

        x1, y1, x2, y2 = self.fix_rectangle

        return (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        )

    # ================================================================
    # UI RECTANGLES
    # ================================================================

    def update_ui_rectangles(
        self,
        width,
        height
    ):

        # ------------------------------------------------------------
        # Trash - top right
        # ------------------------------------------------------------

        tx2 = width - self.trash_margin
        ty1 = self.trash_margin

        tx1 = tx2 - self.trash_width
        ty2 = ty1 + self.trash_height

        self.trash_rectangle = (
            tx1,
            ty1,
            tx2,
            ty2
        )

        # ------------------------------------------------------------
        # Fix - top left
        # ------------------------------------------------------------

        fx1 = self.trash_margin
        fy1 = self.trash_margin

        fx2 = fx1 + self.fix_width
        fy2 = fy1 + self.fix_height

        self.fix_rectangle = (
            fx1,
            fy1,
            fx2,
            fy2
        )

    # ================================================================
    # START CUT
    # ================================================================

    def start_cut(self, point):

        if self.cut_locked:
            return

        self.mode = "CUT"

        self.current_point = (
            int(point[0]),
            int(point[1])
        )

        # ------------------------------------------------------------
        # Start stability timer only once
        # ------------------------------------------------------------

        if self.stable_reference_point is None:

            self.stable_reference_point = (
                self.current_point
            )

            self.stable_start_time = (
                time.monotonic()
            )

            self.hold_progress = 0.0

    # ================================================================
    # UPDATE CUT
    # ================================================================

    def update_cut(self, point):

        if self.cut_locked:
            return

        point = (
            int(point[0]),
            int(point[1])
        )

        self.current_point = point

        # ------------------------------------------------------------
        # Initialize stability tracking
        # ------------------------------------------------------------

        if self.stable_reference_point is None:

            self.stable_reference_point = point

            self.stable_start_time = (
                time.monotonic()
            )

            self.hold_progress = 0.0

            return

        # ------------------------------------------------------------
        # Calculate movement
        # ------------------------------------------------------------

        dx = (
            point[0] -
            self.stable_reference_point[0]
        )

        dy = (
            point[1] -
            self.stable_reference_point[1]
        )

        distance = (
            dx * dx +
            dy * dy
        ) ** 0.5

        # ------------------------------------------------------------
        # Movement exceeded tolerance
        # ------------------------------------------------------------

        if distance > self.stability_threshold:

            self.stable_reference_point = point

            self.stable_start_time = (
                time.monotonic()
            )

            self.hold_progress = 0.0

            return

        # ------------------------------------------------------------
        # Stable
        # ------------------------------------------------------------

        if self.stable_start_time is None:

            self.stable_start_time = (
                time.monotonic()
            )

        elapsed = (
            time.monotonic()
            -
            self.stable_start_time
        )

        self.hold_progress = min(
            elapsed / self.hold_time,
            1.0
        )

        # ------------------------------------------------------------
        # Point locked after 2 seconds
        # ------------------------------------------------------------

        if elapsed >= self.hold_time:

            self.lock_current_point()

    # ================================================================
    # LOCK CURRENT POINT
    # ================================================================

    def lock_current_point(self):

        if self.cut_locked:
            return

        # ============================================================
        # FIRST POINT
        # ============================================================

        if self.point_selection_state == "POINT_1":

            if self.stable_reference_point is None:
                return

            self.point1 = (
                int(self.stable_reference_point[0]),
                int(self.stable_reference_point[1])
            )

            self.point_selection_state = "POINT_2"

            self.stable_reference_point = None
            self.stable_start_time = None
            self.hold_progress = 0.0

            return

        # ============================================================
        # SECOND POINT
        # ============================================================

        if self.point_selection_state == "POINT_2":

            if self.stable_reference_point is None:
                return

            self.point2 = (
                int(self.stable_reference_point[0]),
                int(self.stable_reference_point[1])
            )

            # --------------------------------------------------------
            # Create rectangle
            # --------------------------------------------------------

            self.rectangle = self.make_rectangle(
                self.point1,
                self.point2
            )

            # --------------------------------------------------------
            # Source stays fixed.
            #
            # Dragging and resizing modify only destination.
            # --------------------------------------------------------

            self.cut_source_rectangle = (
                self.rectangle
            )

            self.cut_active = True

            self.mode = "NORMAL"

            self.point_selection_state = "COMPLETE"

            self.stable_reference_point = None
            self.stable_start_time = None
            self.hold_progress = 0.0

    # ================================================================
    # FINISH CUT
    # ================================================================

    def finish_cut(self):

        if self.cut_locked:
            return

        if self.point_selection_state != "COMPLETE":

            self.reset_point_selection()

            self.mode = "NORMAL"

    # ================================================================
    # RESET POINT SELECTION
    # ================================================================

    def reset_point_selection(self):

        self.point1 = None
        self.point2 = None

        self.start_point = None
        self.current_point = None

        self.stable_reference_point = None
        self.stable_start_time = None

        self.hold_progress = 0.0

        self.point_selection_state = "POINT_1"

    # ================================================================
    # START DRAG
    # ================================================================

    def start_drag(
        self,
        hand_x,
        hand_y
    ):

        if self.rectangle is None:
            return

        if self.cut_locked:
            return

        x1, y1, x2, y2 = self.rectangle

        self.drag_offset_x = (
            hand_x - x1
        )

        self.drag_offset_y = (
            hand_y - y1
        )

        self.dragging = True

        self.drag_release_counter = 0

        self.last_resize_distance = None
        self.resize_start_distance = None

    # ================================================================
    # DRAG
    # ================================================================

    def drag(
        self,
        hand_x,
        hand_y
    ):

        if not self.dragging:
            return

        if self.rectangle is None:
            return

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

        self.over_trash = (
            self.point_inside_trash(
                hand_x,
                hand_y
            )
        )

        self.last_resize_distance = None
        self.resize_start_distance = None

        self.drag_release_counter = 0

    # ================================================================
    # RELEASE DRAG
    # ================================================================

    def release_drag(self):

        self.dragging = False

        self.drag_release_counter = 0

        if self.over_trash:

            self.discard_cut()

            return

        self.mode = "NORMAL"

    # ================================================================
    # RESIZE
    # ================================================================

    def resize_cut(
        self,
        distance
    ):

        if self.rectangle is None:
            return

        if self.cut_locked:
            return

        x1, y1, x2, y2 = self.rectangle

        center_x = (
            x1 + x2
        ) / 2

        center_y = (
            y1 + y2
        ) / 2

        width = x2 - x1
        height = y2 - y1

        # ------------------------------------------------------------
        # Initialize resize
        # ------------------------------------------------------------

        if self.resize_start_distance is None:

            self.resize_start_distance = distance
            self.last_resize_distance = distance

            return

        delta = (
            distance -
            self.last_resize_distance
        )

        # ------------------------------------------------------------
        # Dead zone
        # ------------------------------------------------------------

        if abs(delta) < self.resize_dead_zone:
            return

        # ------------------------------------------------------------
        # Smooth resize
        # ------------------------------------------------------------

        smoothed_delta = (
            delta *
            self.resize_smoothing
        )

        width += (
            smoothed_delta *
            self.resize_speed
        )

        height += (
            smoothed_delta *
            self.resize_speed
        )

        # ------------------------------------------------------------
        # Original source dimensions
        # ------------------------------------------------------------

        if self.cut_source_rectangle:

            source_width = (
                self.cut_source_rectangle[2]
                -
                self.cut_source_rectangle[0]
            )

            source_height = (
                self.cut_source_rectangle[3]
                -
                self.cut_source_rectangle[1]
            )

            original_width = max(
                self.min_size,
                source_width
            )

            original_height = max(
                self.min_size,
                source_height
            )

        else:

            original_width = max(
                self.min_size,
                width
            )

            original_height = max(
                self.min_size,
                height
            )

        # ------------------------------------------------------------
        # Maximum dimensions
        # ------------------------------------------------------------

        max_width = (
            original_width *
            self.max_scale
        )

        max_height = (
            original_height *
            self.max_scale
        )

        width = max(
            self.min_size,
            min(
                width,
                max_width
            )
        )

        height = max(
            self.min_size,
            min(
                height,
                max_height
            )
        )

        # ------------------------------------------------------------
        # Rebuild rectangle around center
        # ------------------------------------------------------------

        self.rectangle = (
            int(
                center_x -
                width / 2
            ),
            int(
                center_y -
                height / 2
            ),
            int(
                center_x +
                width / 2
            ),
            int(
                center_y +
                height / 2
            )
        )

        self.last_resize_distance = distance

    # ================================================================
    # LOCK CUT
    # ================================================================

    def lock_cut(self):

        if not self.cut_active:
            return

        self.cut_locked = True

        self.dragging = False

        self.mode = "NORMAL"

    # ================================================================
    # DISCARD CUT
    # ================================================================

    def discard_cut(self):

        self.start_point = None
        self.current_point = None

        self.point1 = None
        self.point2 = None

        self.rectangle = None
        self.cut_source_rectangle = None

        self.cut_active = False

        self.mode = "NORMAL"

        self.dragging = False

        self.drag_offset_x = 0
        self.drag_offset_y = 0

        self.drag_release_counter = 0

        self.over_trash = False
        self.over_fix = False

        self.cut_locked = False

        self.reset_point_selection()

        self.last_resize_distance = None
        self.resize_start_distance = None

    # ================================================================
    # RESET
    # ================================================================

    def reset(self):

        self.start_point = None
        self.current_point = None

        self.point1 = None
        self.point2 = None

        self.rectangle = None
        self.cut_source_rectangle = None

        self.cut_active = False

        self.mode = "NORMAL"

        self.dragging = False

        self.drag_offset_x = 0
        self.drag_offset_y = 0

        self.drag_release_counter = 0

        self.over_trash = False
        self.over_fix = False

        self.cut_locked = False

        self.reset_point_selection()

        self.last_resize_distance = None
        self.resize_start_distance = None

    # ================================================================
    # NORMALIZE HUMAN MASK
    #
    # Converts all supported mask formats into:
    #
    #     (height, width)
    #
    # This prevents boolean indexing errors when the segmentation
    # model returns (height, width, 1).
    # ================================================================

    def normalize_mask(
        self,
        human_mask,
        width,
        height
    ):

        if human_mask is None:
            return None

        mask = human_mask

        # ------------------------------------------------------------
        # Remove unnecessary singleton dimensions
        #
        # (H, W, 1) -> (H, W)
        # ------------------------------------------------------------

        if len(mask.shape) == 3:

            if mask.shape[2] == 1:

                mask = mask[:, :, 0]

            elif mask.shape[2] == 3:

                mask = cv2.cvtColor(
                    mask,
                    cv2.COLOR_BGR2GRAY
                )

            elif mask.shape[2] == 4:

                mask = cv2.cvtColor(
                    mask,
                    cv2.COLOR_BGRA2GRAY
                )

        # ------------------------------------------------------------
        # Handle any remaining singleton dimensions
        # ------------------------------------------------------------

        mask = mask.squeeze()

        # ------------------------------------------------------------
        # Ensure mask is 2D
        # ------------------------------------------------------------

        if len(mask.shape) != 2:

            return None

        # ------------------------------------------------------------
        # Resize mask if necessary
        # ------------------------------------------------------------

        if (
            mask.shape[0] != height
            or
            mask.shape[1] != width
        ):

            mask = cv2.resize(
                mask,
                (
                    width,
                    height
                ),
                interpolation=cv2.INTER_NEAREST
            )

        return mask

    # ================================================================
    # APPLY CUT
    #
    # Layer order:
    #
    #     HUMAN SILHOUETTE
    #           ↑
    #       MOVING CUT
    #           ↑
    #     VIDEO BACKGROUND
    #
    # Gesture detection and segmentation are still based on the
    # original camera frame outside this controller.
    # ================================================================

    def apply_cut(
        self,
        video_frame,
        human_frame,
        human_mask
    ):

        # ------------------------------------------------------------
        # Start with video background
        # ------------------------------------------------------------

        result = video_frame.copy()

        frame_height, frame_width = (
            human_frame.shape[:2]
        )

        # ============================================================
        # NORMALIZE MASK
        # ============================================================

        mask = self.normalize_mask(
            human_mask,
            frame_width,
            frame_height
        )

        # ------------------------------------------------------------
        # If no usable mask exists, return background.
        # ------------------------------------------------------------

        if mask is None:
            return result

        # ============================================================
        # APPLY MOVING CUT
        # ============================================================

        if (
            self.cut_active
            and
            self.rectangle is not None
            and
            self.cut_source_rectangle is not None
        ):

            # --------------------------------------------------------
            # SOURCE RECTANGLE
            # --------------------------------------------------------

            sx1, sy1, sx2, sy2 = (
                self.cut_source_rectangle
            )

            sx1 = max(
                0,
                min(
                    int(sx1),
                    frame_width
                )
            )

            sy1 = max(
                0,
                min(
                    int(sy1),
                    frame_height
                )
            )

            sx2 = max(
                0,
                min(
                    int(sx2),
                    frame_width
                )
            )

            sy2 = max(
                0,
                min(
                    int(sy2),
                    frame_height
                )
            )

            # --------------------------------------------------------
            # Valid source
            # --------------------------------------------------------

            if (
                sx2 > sx1
                and
                sy2 > sy1
            ):

                # ----------------------------------------------------
                # Extract original camera region
                # ----------------------------------------------------

                source_patch = (
                    human_frame[
                        sy1:sy2,
                        sx1:sx2
                    ].copy()
                )

                source_mask = (
                    mask[
                        sy1:sy2,
                        sx1:sx2
                    ]
                )

                # ----------------------------------------------------
                # Remove the person from the cut source.
                #
                # IMPORTANT:
                # source_mask is now guaranteed to be 2D.
                # ----------------------------------------------------

                source_patch[
                    source_mask > 0
                ] = 0

                # ----------------------------------------------------
                # DESTINATION RECTANGLE
                # ----------------------------------------------------

                dx1, dy1, dx2, dy2 = (
                    self.rectangle
                )

                dx1 = int(dx1)
                dy1 = int(dy1)
                dx2 = int(dx2)
                dy2 = int(dy2)

                dest_width = (
                    dx2 -
                    dx1
                )

                dest_height = (
                    dy2 -
                    dy1
                )

                if (
                    dest_width > 0
                    and
                    dest_height > 0
                ):

                    # ------------------------------------------------
                    # Resize source patch
                    # ------------------------------------------------

                    patch = cv2.resize(
                        source_patch,
                        (
                            dest_width,
                            dest_height
                        ),
                        interpolation=cv2.INTER_LINEAR
                    )

                    # ------------------------------------------------
                    # Clip destination to frame
                    # ------------------------------------------------

                    clip_x1 = max(
                        0,
                        dx1
                    )

                    clip_y1 = max(
                        0,
                        dy1
                    )

                    clip_x2 = min(
                        frame_width,
                        dx2
                    )

                    clip_y2 = min(
                        frame_height,
                        dy2
                    )

                    if (
                        clip_x2 > clip_x1
                        and
                        clip_y2 > clip_y1
                    ):

                        # --------------------------------------------
                        # Patch coordinates
                        # --------------------------------------------

                        patch_x1 = (
                            clip_x1 -
                            dx1
                        )

                        patch_y1 = (
                            clip_y1 -
                            dy1
                        )

                        patch_x2 = (
                            patch_x1 +
                            (
                                clip_x2 -
                                clip_x1
                            )
                        )

                        patch_y2 = (
                            patch_y1 +
                            (
                                clip_y2 -
                                clip_y1
                            )
                        )

                        # --------------------------------------------
                        # Additional safety clipping
                        # --------------------------------------------

                        patch_x1 = max(
                            0,
                            min(
                                patch_x1,
                                patch.shape[1]
                            )
                        )

                        patch_y1 = max(
                            0,
                            min(
                                patch_y1,
                                patch.shape[0]
                            )
                        )

                        patch_x2 = max(
                            patch_x1,
                            min(
                                patch_x2,
                                patch.shape[1]
                            )
                        )

                        patch_y2 = max(
                            patch_y1,
                            min(
                                patch_y2,
                                patch.shape[0]
                            )
                        )

                        patch_crop = patch[
                            patch_y1:patch_y2,
                            patch_x1:patch_x2
                        ]

                        # --------------------------------------------
                        # Make sure dimensions match destination
                        # --------------------------------------------

                        target_height = (
                            clip_y2 -
                            clip_y1
                        )

                        target_width = (
                            clip_x2 -
                            clip_x1
                        )

                        if (
                            patch_crop.shape[0] ==
                            target_height
                            and
                            patch_crop.shape[1] ==
                            target_width
                        ):

                            result[
                                clip_y1:clip_y2,
                                clip_x1:clip_x2
                            ] = patch_crop

        # ============================================================
        # RESTORE HUMAN SILHOUETTE
        #
        # This is ALWAYS performed after the cut.
        #
        # Boolean mask is guaranteed to be (H, W).
        # Therefore it can safely index the 3-channel frame.
        # ============================================================

        human_pixels = (
            mask > 0
        )

        result[
            human_pixels
        ] = human_frame[
            human_pixels
        ]

        return result
