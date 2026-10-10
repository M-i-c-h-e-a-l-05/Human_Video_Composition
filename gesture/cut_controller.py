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

        # Smallest allowed cut side (pixels)
        self.min_cut_size = 20

        # Seconds P1 waits for P2 before the selection is cancelled
        self.point_timeout = 10.0

        self.point1_time = None

        # True = leave a black hole where the video was cut from
        self.show_source_hole = True

        # ============================================================
        # CUT AS FULL BACKGROUND
        # ============================================================

        # True  = show the cut part extended to the frame shape
        #         (no stretching)
        # False = stretch the exact cut part to the full frame
        self.background_fit_aspect = True

        # Visible part of the video as fractions of the full frame:
        # (x1, y1, x2, y2). None = whole video.
        self.background_crop = None

        self.background_rectangle = None
        self.over_background = False

        # Back button (shown only while a background is active)
        self.back_rectangle = None
        self.over_back = False

        self.frame_width = None
        self.frame_height = None

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

        # ------------------------------------------------------------
        # Background button - a little below Fix
        # ------------------------------------------------------------

        self.background_rectangle = (
            fx1,
            fy2 + self.fix_gap,
            fx2,
            fy2 + self.fix_gap + self.fix_height
        )

        # ------------------------------------------------------------
        # Back button - below the Background button
        # ------------------------------------------------------------

        bx1, by1, bx2, by2 = self.background_rectangle

        self.back_rectangle = (
            bx1,
            by2 + self.fix_gap,
            bx2,
            by2 + self.fix_gap + self.fix_height
        )

        self.frame_width = width
        self.frame_height = height

    # ================================================================
    # POINT INSIDE BACKGROUND BUTTON
    # ================================================================

    def point_inside_background(self, x, y):

        if self.background_rectangle is None:
            return False

        x1, y1, x2, y2 = self.background_rectangle

        return (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        )

    # ================================================================
    # POINT INSIDE BACK BUTTON
    # ================================================================

    def point_inside_back(self, x, y):

        if self.back_rectangle is None:
            return False

        x1, y1, x2, y2 = self.back_rectangle

        return (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        )

    # ================================================================
    # BACK TO THE ORIGINAL BACKGROUND
    # ================================================================

    def reset_background(self):

        self.background_crop = None

        self.over_back = False

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

            self.point1_time = time.monotonic()

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

            candidate = (
                int(self.stable_reference_point[0]),
                int(self.stable_reference_point[1])
            )

            x1, y1, x2, y2 = self.make_rectangle(
                self.point1,
                candidate
            )

            # --------------------------------------------------------
            # Too small: keep waiting for a real second point
            # --------------------------------------------------------

            if (
                abs(x2 - x1) <= self.min_cut_size
                or
                abs(y2 - y1) <= self.min_cut_size
            ):

                self.stable_reference_point = None
                self.stable_start_time = None
                self.hold_progress = 0.0

                return

            self.point2 = candidate

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

        self.point1_time = None

        self.point_selection_state = "POINT_1"

    # ================================================================
    # CLEAR HOLD
    #
    # Called when the cut gesture is not being shown.
    # Stops a half-finished 2-second hold from resuming later,
    # and cancels P1 if P2 never arrives.
    # ================================================================

    def clear_hold(self):

        if self.point_selection_state == "COMPLETE":
            return

        self.stable_reference_point = None
        self.stable_start_time = None
        self.hold_progress = 0.0

        if (
            self.point_selection_state == "POINT_2"
            and
            self.point1_time is not None
            and
            time.monotonic() - self.point1_time
            > self.point_timeout
        ):

            self.reset_point_selection()

            self.mode = "NORMAL"

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
    #
    # Pinch distance is compared with the distance at pinch start.
    # Wider than start (past the dead zone) = keeps growing.
    # Narrower than start = keeps shrinking.
    # Aspect ratio is always kept.
    # ================================================================

    def resize_cut(
        self,
        distance
    ):

        if self.rectangle is None:
            return

        if self.cut_locked:
            return

        # ------------------------------------------------------------
        # Initialize resize
        # ------------------------------------------------------------

        if self.resize_start_distance is None:

            self.resize_start_distance = distance
            self.last_resize_distance = distance

            return

        delta = (
            distance -
            self.resize_start_distance
        )

        # ------------------------------------------------------------
        # Dead zone
        # ------------------------------------------------------------

        if abs(delta) < self.resize_dead_zone:
            return

        # ------------------------------------------------------------
        # Scale factor for this frame
        # ------------------------------------------------------------

        if delta > 0:
            factor = 1.0 + self.resize_speed
        else:
            factor = 1.0 - self.resize_speed

        x1, y1, x2, y2 = self.rectangle

        center_x = (
            x1 + x2
        ) / 2

        center_y = (
            y1 + y2
        ) / 2

        width = (x2 - x1) * factor
        height = (y2 - y1) * factor

        # ------------------------------------------------------------
        # Limits (uniform, so aspect ratio never changes)
        # ------------------------------------------------------------

        if factor < 1.0:

            if (
                width < self.min_size
                or
                height < self.min_size
            ):
                return

        else:

            if self.cut_source_rectangle:

                max_width = (
                    (
                        self.cut_source_rectangle[2]
                        -
                        self.cut_source_rectangle[0]
                    )
                    *
                    self.max_scale
                )

                max_height = (
                    (
                        self.cut_source_rectangle[3]
                        -
                        self.cut_source_rectangle[1]
                    )
                    *
                    self.max_scale
                )

                if (
                    width > max_width
                    or
                    height > max_height
                ):
                    return

        # ------------------------------------------------------------
        # Rebuild rectangle around center
        # ------------------------------------------------------------

        self.rectangle = (
            int(center_x - width / 2),
            int(center_y - height / 2),
            int(center_x + width / 2),
            int(center_y + height / 2)
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
    # UNLOCK CUT
    # ================================================================

    def unlock_cut(self):

        self.cut_locked = False

    # ================================================================
    # CUT PART -> FULL BACKGROUND
    #
    # The video part that was cut becomes the whole background.
    # The cut itself is cleared so a new one can be made.
    # Press R to go back to the full video.
    # ================================================================

    def set_as_background(self):

        if (
            not self.cut_active
            or
            self.cut_source_rectangle is None
            or
            self.frame_width is None
        ):
            return False

        width = self.frame_width
        height = self.frame_height

        sx1, sy1, sx2, sy2 = self.cut_source_rectangle

        sx1 = max(0, min(width, sx1))
        sx2 = max(0, min(width, sx2))
        sy1 = max(0, min(height, sy1))
        sy2 = max(0, min(height, sy2))

        crop_w = sx2 - sx1
        crop_h = sy2 - sy1

        if crop_w < 2 or crop_h < 2:
            return False

        # ------------------------------------------------------------
        # Extend to frame shape so nothing is stretched
        # ------------------------------------------------------------

        if self.background_fit_aspect:

            aspect = width / height

            if crop_w / crop_h < aspect:
                crop_w = crop_h * aspect
            else:
                crop_h = crop_w / aspect

            if crop_w > width:
                crop_h *= width / crop_w
                crop_w = width

            if crop_h > height:
                crop_w *= height / crop_h
                crop_h = height

            center_x = (sx1 + sx2) / 2
            center_y = (sy1 + sy2) / 2

            sx1 = max(0, min(center_x - crop_w / 2, width - crop_w))
            sy1 = max(0, min(center_y - crop_h / 2, height - crop_h))

            sx2 = sx1 + crop_w
            sy2 = sy1 + crop_h

        # ------------------------------------------------------------
        # Map from the frame being shown to the original video
        # ------------------------------------------------------------

        if self.background_crop is None:
            cx1, cy1, cx2, cy2 = 0.0, 0.0, 1.0, 1.0
        else:
            cx1, cy1, cx2, cy2 = self.background_crop

        span_x = cx2 - cx1
        span_y = cy2 - cy1

        self.background_crop = (
            cx1 + (sx1 / width) * span_x,
            cy1 + (sy1 / height) * span_y,
            cx1 + (sx2 / width) * span_x,
            cy1 + (sy2 / height) * span_y
        )

        self.discard_cut()

        return True

    # ================================================================
    # APPLY BACKGROUND CROP
    #
    # Call on every video frame BEFORE apply_cut.
    # ================================================================

    def apply_background(self, frame):

        if self.background_crop is None:
            return frame

        height, width = frame.shape[:2]

        fx1, fy1, fx2, fy2 = self.background_crop

        x1 = int(fx1 * width)
        y1 = int(fy1 * height)
        x2 = int(fx2 * width)
        y2 = int(fy2 * height)

        if x2 - x1 < 2 or y2 - y1 < 2:
            return frame

        return cv2.resize(
            frame[y1:y2, x1:x2],
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

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

        self.background_crop = None
        self.over_background = False

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
                # Extract LIVE video region
                #
                # video_frame is a new frame every loop, so the cut
                # keeps playing wherever it is moved or resized.
                # ----------------------------------------------------

                source_patch = (
                    video_frame[
                        sy1:sy2,
                        sx1:sx2
                    ].copy()
                )

                # ----------------------------------------------------
                # Leave a hole where the video was cut from
                # ----------------------------------------------------

                if self.show_source_hole:

                    result[
                        sy1:sy2,
                        sx1:sx2
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