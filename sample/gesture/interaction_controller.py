class InteractionController:

    def __init__(
        self,
        hand_detector,
        gesture_recognizer,
        cut_controller,
        grid_controller
    ):

        self.hand_detector = hand_detector

        self.gesture_recognizer = (
            gesture_recognizer
        )

        self.cut_controller = (
            cut_controller
        )

        self.grid_controller = (
            grid_controller
        )

        # ============================================================
        # DRAG GRACE
        # ============================================================

        self.drag_gesture_grace_frames = 10
        self.drag_hand_lost_grace_frames = 10

        self.drag_gesture_lost_counter = 0
        self.drag_hand_lost_counter = 0

        self.last_drag_position = None

    # ================================================================
    # PROCESS HAND
    # ================================================================

    def process_hand(self, frame):

        height, width = frame.shape[:2]

        cut = self.cut_controller
        recognizer = self.gesture_recognizer
        grid = self.grid_controller

        # ------------------------------------------------------------
        # Update UI geometry
        # ------------------------------------------------------------

        cut.update_ui_rectangles(
            width,
            height
        )

        grid.update_grid(
            width,
            height
        )

        # ------------------------------------------------------------
        # Detect hand from ORIGINAL CAMERA FRAME
        # ------------------------------------------------------------

        landmarks = self.hand_detector.detect(frame)

        # ============================================================
        # NO HAND DETECTED
        # ============================================================

        if landmarks is None:

            grid.button_was_inside = False

            # --------------------------------------------------------
            # Grid handling
            # --------------------------------------------------------

            if grid.active and grid.selection_active:

                grid.finish_selection()
                grid.create_cut(cut)

            # --------------------------------------------------------
            # Dragging
            #
            # Give MediaPipe a few frames to recover.
            # --------------------------------------------------------

            if cut.dragging:

                cut.drag_release_counter += 1
                self.drag_hand_lost_counter += 1

                if (
                    self.drag_hand_lost_counter >=
                    self.drag_hand_lost_grace_frames
                ):

                    cut.release_drag()

                    self.drag_hand_lost_counter = 0
                    self.drag_gesture_lost_counter = 0
                    self.last_drag_position = None

            return None

        # ============================================================
        # HAND DETECTED AGAIN
        # ============================================================

        cut.drag_release_counter = 0

        self.drag_hand_lost_counter = 0

        # ============================================================
        # LANDMARK POSITIONS
        # ============================================================

        # Index fingertip = landmark 8
        index_x, index_y = (
            recognizer.get_index_position(
                landmarks,
                width,
                height
            )
        )

        # Middle fingertip = landmark 12
        middle_x, middle_y = (
            recognizer.get_middle_position(
                landmarks,
                width,
                height
            )
        )

        # Hand center
        hand_x, hand_y = (
            recognizer.get_hand_center(
                landmarks,
                width,
                height
            )
        )

        # ============================================================
        # GRID BUTTON
        # ============================================================

        inside_button = (
            grid.point_inside_button(
                index_x,
                index_y
            )
        )

        if (
            inside_button
            and
            not grid.button_was_inside
        ):

            # --------------------------------------------------------
            # Locked cut blocks grid activation
            # --------------------------------------------------------

            if cut.cut_locked:

                grid.button_was_inside = True

                return {
                    "landmarks": landmarks,
                    "gesture": "LOCKED",
                    "position": (
                        index_x,
                        index_y
                    )
                }

            # --------------------------------------------------------
            # Toggle grid
            # --------------------------------------------------------

            if grid.active:

                if grid.selection_active:

                    grid.finish_selection()
                    grid.create_cut(cut)

                grid.deactivate()

            else:

                grid.activate(
                    width,
                    height
                )

            grid.button_was_inside = True

            return {
                "landmarks": landmarks,
                "gesture": "GRID_BUTTON",
                "position": (
                    index_x,
                    index_y
                )
            }

        grid.button_was_inside = inside_button

        # ============================================================
        # GRID MODE
        # ============================================================

        if grid.active:

            # --------------------------------------------------------
            # Grid selection
            # --------------------------------------------------------

            if recognizer.is_grid_gesture(
                landmarks
            ):

                if not grid.selection_active:

                    grid.start_selection(
                        (index_x, index_y),
                        (middle_x, middle_y)
                    )

                else:

                    grid.update_selection(
                        (index_x, index_y),
                        (middle_x, middle_y)
                    )

                return {
                    "landmarks": landmarks,
                    "gesture": "GRID",
                    "position": (
                        index_x,
                        index_y
                    )
                }

            # --------------------------------------------------------
            # Release grid selection
            # --------------------------------------------------------

            if grid.selection_active:

                grid.finish_selection()
                grid.create_cut(cut)

                return {
                    "landmarks": landmarks,
                    "gesture": "GRID_RELEASE",
                    "position": (
                        index_x,
                        index_y
                    )
                }

            return {
                "landmarks": landmarks,
                "gesture": "GRID_IDLE",
                "position": (
                    index_x,
                    index_y
                )
            }

        # ============================================================
        # FIX / LOCK BUTTON
        # ============================================================

        cut.over_fix = (
            cut.point_inside_fix(
                hand_x,
                hand_y
            )
        )

        if (
            cut.cut_active
            and
            not cut.cut_locked
            and
            cut.over_fix
        ):

            cut.lock_cut()

        # ============================================================
        # TWO-POINT CUT SYSTEM
        # ============================================================

        if (
            recognizer.is_cut_gesture(
                landmarks
            )
            and
            not cut.cut_locked
        ):

            # --------------------------------------------------------
            # CutController handles:
            #
            #   Point 1
            #   Point 2
            #   fingertip stability
            #   2-second hold
            #   rectangle creation
            #
            # Position is ALWAYS the index fingertip (landmark 8).
            # --------------------------------------------------------

            current_point = (
                index_x,
                index_y
            )

            cut.start_cut(
                current_point
            )

            cut.update_cut(
                current_point
            )

            # A CUT gesture means the user is actively interacting.
            self.drag_gesture_lost_counter = 0

            return {
                "landmarks": landmarks,
                "gesture": "CUT",
                "position": current_point
            }

        # ============================================================
        # FIST / DRAG
        # ============================================================

        if recognizer.is_fist(
            landmarks
        ):

            self.drag_gesture_lost_counter = 0

            # --------------------------------------------------------
            # Start dragging only when fist is inside rectangle
            # --------------------------------------------------------

            if cut.rectangle is not None:

                if not cut.dragging:

                    if cut.point_inside_rectangle(
                        hand_x,
                        hand_y
                    ):

                        cut.start_drag(
                            hand_x,
                            hand_y
                        )

                # ----------------------------------------------------
                # Continue dragging
                # ----------------------------------------------------

                if cut.dragging:

                    cut.drag(
                        hand_x,
                        hand_y
                    )

                    self.last_drag_position = (
                        hand_x,
                        hand_y
                    )

            return {
                "landmarks": landmarks,
                "gesture": "FIST",
                "position": (
                    hand_x,
                    hand_y
                )
            }

        # ============================================================
        # RESIZE
        # ============================================================

        if cut.dragging:

            if recognizer.is_resize_gesture(
                landmarks
            ):

                distance = (
                    recognizer.get_resize_distance(
                        landmarks,
                        width,
                        height
                    )
                )

                cut.resize_cut(
                    distance
                )

                cut.over_trash = False

                cut.drag_release_counter = 0

                self.drag_gesture_lost_counter = 0

                self.last_drag_position = (
                    hand_x,
                    hand_y
                )

                return {
                    "landmarks": landmarks,
                    "gesture": "RESIZE",
                    "position": (
                        hand_x,
                        hand_y
                    )
                }

            else:

                cut.last_resize_distance = None
                cut.resize_start_distance = None

        # ============================================================
        # DRAG GRACE PERIOD
        # ============================================================

        if cut.dragging:

            self.drag_gesture_lost_counter += 1

            # --------------------------------------------------------
            # Keep the last valid position during short recognition
            # instability.
            # --------------------------------------------------------

            if (
                self.last_drag_position is not None
                and
                self.drag_gesture_lost_counter <=
                self.drag_gesture_grace_frames
            ):

                cut.drag(
                    self.last_drag_position[0],
                    self.last_drag_position[1]
                )

                return {
                    "landmarks": landmarks,
                    "gesture": "DRAG_GRACE",
                    "position": (
                        self.last_drag_position[0],
                        self.last_drag_position[1]
                    )
                }

            # --------------------------------------------------------
            # Gesture has actually been released
            # --------------------------------------------------------

            if (
                self.drag_gesture_lost_counter >
                self.drag_gesture_grace_frames
            ):

                cut.release_drag()

                self.last_drag_position = None
                self.drag_gesture_lost_counter = 0

        # ============================================================
        # NORMAL
        # ============================================================

        return {
            "landmarks": landmarks,
            "gesture": "NORMAL",
            "position": (
                index_x,
                index_y
            )
        }

    # ================================================================
    # RESET
    # ================================================================

    def reset(self):

        self.cut_controller.reset()

        self.grid_controller.deactivate()

        self.grid_controller.button_was_inside = False

        self.drag_gesture_lost_counter = 0

        self.drag_hand_lost_counter = 0

        self.last_drag_position = None
