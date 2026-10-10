import cv2
import math


class GestureOverlay:

    def __init__(
        self,
        cut_controller,
        grid_controller
    ):

        self.cut_controller = cut_controller
        self.grid_controller = grid_controller

    # ================================================================
    # MAIN DRAW
    # ================================================================

    def draw(self, frame, hand_result=None):

        cut = self.cut_controller
        grid = self.grid_controller

        # ------------------------------------------------------------
        # Grid button
        # ------------------------------------------------------------

        self.draw_grid_button(frame)

        # ------------------------------------------------------------
        # Grid itself
        # ------------------------------------------------------------

        self.draw_grid(frame)

        # ------------------------------------------------------------
        # Two-point cropping UI
        # ------------------------------------------------------------

        if hand_result is not None:

            self.draw_cut_points(
                frame,
                hand_result
            )

        else:

            # Still show locked points even if hand temporarily
            # disappears.
            self.draw_locked_points(frame)

        # ------------------------------------------------------------
        # Existing cut rectangle
        # ------------------------------------------------------------

        if (
            cut.rectangle is not None and
            cut.cut_active
        ):

            self.draw_cut_rectangle(
                frame,
                cut.rectangle
            )

        # ------------------------------------------------------------
        # Trash + fix buttons
        # ------------------------------------------------------------

        if cut.cut_active:

            self.draw_trash_bin(frame)

            self.draw_fix_button(frame)

        return frame

    # ================================================================
    # CUT POINTS
    # ================================================================

    def draw_cut_points(
        self,
        frame,
        hand_result
    ):

        cut = self.cut_controller

        # ------------------------------------------------------------
        # Get fingertip position from InteractionController result
        # ------------------------------------------------------------

        position = None

        if isinstance(hand_result, dict):

            position = hand_result.get(
                "position"
            )

        # ------------------------------------------------------------
        # Draw Point 1
        # ------------------------------------------------------------

        if cut.point1 is not None:

            self.draw_point(
                frame,
                cut.point1,
                label="P1",
                locked=True
            )

        # ------------------------------------------------------------
        # Draw Point 2
        # ------------------------------------------------------------

        if cut.point2 is not None:

            self.draw_point(
                frame,
                cut.point2,
                label="P2",
                locked=True
            )

        # ------------------------------------------------------------
        # Draw current fingertip
        # ------------------------------------------------------------

        if (
            position is not None and
            cut.point_selection_state != "COMPLETE"
        ):

            self.draw_current_point(
                frame,
                position,
                cut.hold_progress
            )

        # ------------------------------------------------------------
        # Draw line between Point 1 and current fingertip
        # ------------------------------------------------------------

        if (
            cut.point1 is not None and
            cut.point2 is None and
            position is not None
        ):

            cv2.line(
                frame,
                (
                    int(cut.point1[0]),
                    int(cut.point1[1])
                ),
                (
                    int(position[0]),
                    int(position[1])
                ),
                (255, 255, 255),
                2
            )

    # ================================================================
    # LOCKED POINTS
    # ================================================================

    def draw_locked_points(self, frame):

        cut = self.cut_controller

        if cut.point1 is not None:

            self.draw_point(
                frame,
                cut.point1,
                label="P1",
                locked=True
            )

        if cut.point2 is not None:

            self.draw_point(
                frame,
                cut.point2,
                label="P2",
                locked=True
            )

    # ================================================================
    # CURRENT FINGERTIP
    # ================================================================

    def draw_current_point(
        self,
        frame,
        position,
        progress
    ):

        x = int(position[0])
        y = int(position[1])

        # ------------------------------------------------------------
        # Outer circle
        # ------------------------------------------------------------

        cv2.circle(
            frame,
            (x, y),
            12,
            (255, 255, 255),
            2
        )

        # ------------------------------------------------------------
        # Inner dot
        # ------------------------------------------------------------

        cv2.circle(
            frame,
            (x, y),
            5,
            (255, 255, 255),
            -1
        )

        # ------------------------------------------------------------
        # Hold progress ring
        # ------------------------------------------------------------

        progress = max(
            0.0,
            min(
                float(progress),
                1.0
            )
        )

        if progress > 0:

            cv2.ellipse(
                frame,
                (x, y),
                (17, 17),
                -90,
                0,
                int(360 * progress),
                (255, 255, 255),
                3
            )

        # ------------------------------------------------------------
        # 2-second message
        # ------------------------------------------------------------

        if progress > 0:

            remaining = max(
                0.0,
                2.0 - (
                    progress * 2.0
                )
            )

            text = f"{remaining:.1f}s"

            cv2.putText(
                frame,
                text,
                (
                    x + 20,
                    y - 10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

    # ================================================================
    # DRAW LOCKED POINT
    # ================================================================

    def draw_point(
        self,
        frame,
        point,
        label="P",
        locked=True
    ):

        x = int(point[0])
        y = int(point[1])

        # ------------------------------------------------------------
        # Outer ring
        # ------------------------------------------------------------

        cv2.circle(
            frame,
            (x, y),
            12,
            (255, 255, 255),
            2
        )

        # ------------------------------------------------------------
        # Center
        # ------------------------------------------------------------

        cv2.circle(
            frame,
            (x, y),
            6,
            (255, 255, 255),
            -1
        )

        # ------------------------------------------------------------
        # Crosshair
        # ------------------------------------------------------------

        cv2.line(
            frame,
            (x - 18, y),
            (x + 18, y),
            (255, 255, 255),
            1
        )

        cv2.line(
            frame,
            (x, y - 18),
            (x, y + 18),
            (255, 255, 255),
            1
        )

        # ------------------------------------------------------------
        # Label
        # ------------------------------------------------------------

        cv2.putText(
            frame,
            label,
            (
                x + 18,
                y + 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # ================================================================
    # CUT RECTANGLE
    # ================================================================

    def draw_cut_rectangle(
        self,
        frame,
        rectangle
    ):

        x1, y1, x2, y2 = rectangle

        # ------------------------------------------------------------
        # Main rectangle
        # ------------------------------------------------------------

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 255, 255),
            2
        )

        # ------------------------------------------------------------
        # Corner markers
        # ------------------------------------------------------------

        corner_length = 15

        # Top-left
        cv2.line(
            frame,
            (int(x1), int(y1)),
            (
                int(x1 + corner_length),
                int(y1)
            ),
            (255, 255, 255),
            3
        )

        cv2.line(
            frame,
            (int(x1), int(y1)),
            (
                int(x1),
                int(y1 + corner_length)
            ),
            (255, 255, 255),
            3
        )

        # Top-right
        cv2.line(
            frame,
            (int(x2), int(y2 - (y2 - y1))),
            (
                int(x2 - corner_length),
                int(y1)
            ),
            (255, 255, 255),
            3
        )

        cv2.line(
            frame,
            (int(x2), int(y1)),
            (
                int(x2),
                int(y1 + corner_length)
            ),
            (255, 255, 255),
            3
        )

        # Bottom-left
        cv2.line(
            frame,
            (int(x1), int(y2)),
            (
                int(x1 + corner_length),
                int(y2)
            ),
            (255, 255, 255),
            3
        )

        cv2.line(
            frame,
            (int(x1), int(y2)),
            (
                int(x1),
                int(y2 - corner_length)
            ),
            (255, 255, 255),
            3
        )

        # Bottom-right
        cv2.line(
            frame,
            (int(x2), int(y2)),
            (
                int(x2 - corner_length),
                int(y2)
            ),
            (255, 255, 255),
            3
        )

        cv2.line(
            frame,
            (int(x2), int(y2)),
            (
                int(x2),
                int(y2 - corner_length)
            ),
            (255, 255, 255),
            3
        )

    # ================================================================
    # GRID BUTTON
    # ================================================================

    def draw_grid_button(self, frame):

        height, width = frame.shape[:2]

        button_width = 100
        button_height = 60

        x1 = (
            width -
            button_width
        ) // 2

        y1 = 20

        x2 = x1 + button_width
        y2 = y1 + button_height

        self.grid_controller.button_rectangle = (
            x1,
            y1,
            x2,
            y2
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "GRID",
            (
                x1 + 20,
                y1 + 38
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # ================================================================
    # GRID
    # ================================================================

    def draw_grid(self, frame):

        grid = self.grid_controller

        if not grid.active:
            return

        if grid.grid_rectangle is None:
            return

        x1, y1, x2, y2 = (
            grid.grid_rectangle
        )

        # ------------------------------------------------------------
        # Grid rectangle
        # ------------------------------------------------------------

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 255, 255),
            2
        )

        # ------------------------------------------------------------
        # Grid lines
        # ------------------------------------------------------------

        if hasattr(grid, "rows") and hasattr(
            grid,
            "cols"
        ):

            rows = max(
                1,
                int(grid.rows)
            )

            cols = max(
                1,
                int(grid.cols)
            )

            cell_width = (
                x2 - x1
            ) / cols

            cell_height = (
                y2 - y1
            ) / rows

            for col in range(
                1,
                cols
            ):

                x = int(
                    x1 +
                    col *
                    cell_width
                )

                cv2.line(
                    frame,
                    (x, int(y1)),
                    (x, int(y2)),
                    (255, 255, 255),
                    1
                )

            for row in range(
                1,
                rows
            ):

                y = int(
                    y1 +
                    row *
                    cell_height
                )

                cv2.line(
                    frame,
                    (int(x1), y),
                    (int(x2), y),
                    (255, 255, 255),
                    1
                )

        # ------------------------------------------------------------
        # Selected cells
        # ------------------------------------------------------------

        if hasattr(
            grid,
            "selected_cells"
        ):

            for cell in grid.selected_cells:

                try:

                    row, col = cell

                    if (
                        hasattr(grid, "rows") and
                        hasattr(grid, "cols")
                    ):

                        rows = max(
                            1,
                            int(grid.rows)
                        )

                        cols = max(
                            1,
                            int(grid.cols)
                        )

                        cell_width = (
                            x2 - x1
                        ) / cols

                        cell_height = (
                            y2 - y1
                        ) / rows

                        cx1 = int(
                            x1 +
                            col *
                            cell_width
                        )

                        cy1 = int(
                            y1 +
                            row *
                            cell_height
                        )

                        cx2 = int(
                            x1 +
                            (col + 1) *
                            cell_width
                        )

                        cy2 = int(
                            y1 +
                            (row + 1) *
                            cell_height
                        )

                        cv2.rectangle(
                            frame,
                            (cx1, cy1),
                            (cx2, cy2),
                            (255, 255, 255),
                            -1
                        )

                except (
                    TypeError,
                    ValueError
                ):
                    continue

    # ================================================================
    # TRASH BIN
    # ================================================================

    def draw_trash_bin(self, frame):

        cut = self.cut_controller

        if cut.trash_rectangle is None:
            return

        x1, y1, x2, y2 = (
            cut.trash_rectangle
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        # Bin body
        body_margin = 25

        cv2.rectangle(
            frame,
            (
                x1 + body_margin,
                y1 + 30
            ),
            (
                x2 - body_margin,
                y2 - 15
            ),
            (255, 255, 255),
            2
        )

        # Lid
        cv2.line(
            frame,
            (
                x1 + 20,
                y1 + 25
            ),
            (
                x2 - 20,
                y1 + 25
            ),
            (255, 255, 255),
            3
        )

        # Handle
        cv2.line(
            frame,
            (
                x1 + 35,
                y1 + 18
            ),
            (
                x2 - 35,
                y1 + 18
            ),
            (255, 255, 255),
            3
        )

        cv2.putText(
            frame,
            "BIN",
            (
                x1 + 22,
                y2 - 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # ================================================================
    # FIX BUTTON
    # ================================================================

    def draw_fix_button(self, frame):

        cut = self.cut_controller

        if cut.fix_rectangle is None:
            return

        x1, y1, x2, y2 = (
            cut.fix_rectangle
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        if cut.cut_locked:

            text = "LOCK"

        else:

            text = "FIX"

        text_size = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2
        )[0]

        text_x = (
            x1 +
            (
                (x2 - x1) -
                text_size[0]
            ) // 2
        )

        text_y = (
            y1 +
            (
                (y2 - y1) +
                text_size[1]
            ) // 2
        )

        cv2.putText(
            frame,
            text,
            (
                text_x,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
