import numpy as np


class GestureRecognizer:

    def finger_states(self, landmarks):

        index_up = (
            landmarks[8].y <
            landmarks[6].y
        )

        middle_up = (
            landmarks[12].y <
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
            index_up,
            middle_up,
            ring_closed,
            pinky_closed
        )

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

    def is_resize_gesture(self, landmarks):

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

        index_open = (
            landmarks[8].y <
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

    def get_resize_distance(
        self,
        landmarks,
        width,
        height
    ):

        thumb_x = (
            landmarks[4].x *
            width
        )

        thumb_y = (
            landmarks[4].y *
            height
        )

        index_x = (
            landmarks[8].x *
            width
        )

        index_y = (
            landmarks[8].y *
            height
        )

        distance = np.sqrt(
            (index_x - thumb_x) ** 2
            +
            (index_y - thumb_y) ** 2
        )

        return distance

    def get_index_position(
        self,
        landmarks,
        width,
        height
    ):

        x = int(
            landmarks[8].x *
            width
        )

        y = int(
            landmarks[8].y *
            height
        )

        return x, y

    def get_hand_center(
        self,
        landmarks,
        width,
        height
    ):

        points = [
            landmarks[0],
            landmarks[5],
            landmarks[9],
            landmarks[13],
            landmarks[17]
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

    def is_grid_gesture(self, landmarks):

        index_up = (
            landmarks[8].y <
            landmarks[6].y
        )

        middle_up = (
            landmarks[12].y <
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
            index_up
            and
            middle_up
            and
            ring_closed
            and
            pinky_closed
        )

    def get_middle_position(
        self,
        landmarks,
        width,
        height
    ):

        x = int(
            landmarks[12].x *
            width
        )

        y = int(
            landmarks[12].y *
            height
        )

        return x, y

    # ================================================================
    # DRAW POSE: only the index finger is up (thumb ignored)
    # ================================================================

    def is_draw_gesture(self, landmarks):

        index_up = (
            landmarks[8].y <
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
            index_up
            and
            middle_closed
            and
            ring_closed
            and
            pinky_closed
        )

    # ================================================================
    # PALM POSE: all four fingers up (open hand) = eraser
    # ================================================================

    def is_palm_gesture(self, landmarks):

        index_up = (
            landmarks[8].y <
            landmarks[6].y
        )

        middle_up = (
            landmarks[12].y <
            landmarks[10].y
        )

        ring_up = (
            landmarks[16].y <
            landmarks[14].y
        )

        pinky_up = (
            landmarks[20].y <
            landmarks[18].y
        )

        return (
            index_up
            and
            middle_up
            and
            ring_up
            and
            pinky_up
        )