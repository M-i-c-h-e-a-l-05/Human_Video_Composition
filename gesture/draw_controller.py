import cv2
import numpy as np


class DrawController:

    def __init__(self):

        # ============================================================
        # SETTINGS
        # ============================================================

        # BGR pen color
        self.pen_color = (0, 200, 255)

        self.pen_thickness = 6

        # Palm eraser size (pixels)
        self.erase_radius = 60

        # 1.0 = no smoothing, lower = smoother (but laggier) lines
        self.smoothing = 0.5

        # ============================================================
        # STATE
        # ============================================================

        self.canvas = None
        self.ink = None

        self.last_point = None
        self.smoothed_point = None
        self.last_erase_point = None

    # ================================================================
    # ENSURE CANVAS SIZE
    # ================================================================

    def ensure_size(self, width, height):

        if (
            self.canvas is None
            or
            self.canvas.shape[0] != height
            or
            self.canvas.shape[1] != width
        ):

            self.canvas = np.zeros(
                (height, width, 3),
                np.uint8
            )

            self.ink = np.zeros(
                (height, width),
                np.uint8
            )

            self.lift()

    # ================================================================
    # DRAW (index finger only)
    # ================================================================

    def draw(self, point, width, height):

        self.ensure_size(width, height)

        self.last_erase_point = None

        x, y = point

        if self.smoothed_point is None:

            sx, sy = x, y

        else:

            sx = (
                self.smoothed_point[0]
                +
                self.smoothing * (x - self.smoothed_point[0])
            )

            sy = (
                self.smoothed_point[1]
                +
                self.smoothing * (y - self.smoothed_point[1])
            )

        current = (int(sx), int(sy))

        radius = max(1, self.pen_thickness // 2)

        if self.last_point is not None:

            cv2.line(
                self.canvas,
                self.last_point,
                current,
                self.pen_color,
                self.pen_thickness
            )

            cv2.line(
                self.ink,
                self.last_point,
                current,
                255,
                self.pen_thickness
            )

        cv2.circle(
            self.canvas,
            current,
            radius,
            self.pen_color,
            -1
        )

        cv2.circle(
            self.ink,
            current,
            radius,
            255,
            -1
        )

        self.smoothed_point = (sx, sy)
        self.last_point = current

    # ================================================================
    # ERASE (open palm rubbing)
    # ================================================================

    def erase(self, point, width, height):

        self.ensure_size(width, height)

        self.last_point = None
        self.smoothed_point = None

        current = (int(point[0]), int(point[1]))

        radius = self.erase_radius

        if self.last_erase_point is not None:

            cv2.line(
                self.canvas,
                self.last_erase_point,
                current,
                (0, 0, 0),
                radius * 2
            )

            cv2.line(
                self.ink,
                self.last_erase_point,
                current,
                0,
                radius * 2
            )

        cv2.circle(
            self.canvas,
            current,
            radius,
            (0, 0, 0),
            -1
        )

        cv2.circle(
            self.ink,
            current,
            radius,
            0,
            -1
        )

        self.last_erase_point = current

    # ================================================================
    # LIFT PEN
    #
    # Called whenever the hand is not in the draw / palm pose, so the
    # next stroke starts fresh instead of joining the last one.
    # ================================================================

    def lift(self):

        self.last_point = None
        self.smoothed_point = None
        self.last_erase_point = None

    # ================================================================
    # CLEAR EVERYTHING
    # ================================================================

    def clear(self):

        if self.canvas is not None:

            self.canvas[:] = 0
            self.ink[:] = 0

        self.lift()

    # ================================================================
    # APPLY ON FRAME
    # ================================================================

    def apply(self, frame):

        if self.canvas is None:
            return frame

        if self.canvas.shape[:2] != frame.shape[:2]:
            return frame

        mask = self.ink > 0

        frame[mask] = self.canvas[mask]

        return frame