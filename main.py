import cv2

from human_segmenter import HumanSegmenter
from video_background import VideoBackground

from gesture.hand_detector import HandDetector
from gesture.gesture_recognizer import GestureRecognizer
from gesture.cut_controller import CutController
from gesture.interaction_controller import InteractionController
from gesture.gesture_overlay import GestureOverlay
from gesture.grid_controller import GridController
from gesture.draw_controller import DrawController


def main():

    # =====================================================
    # WEBCAM
    # =====================================================

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print(
            "Error: Could not open webcam."
        )

        return

    # =====================================================
    # HUMAN SEGMENTER
    # =====================================================

    segmenter = HumanSegmenter(
        "models/selfie_segmenter.tflite"
    )

    # =====================================================
    # VIDEO BACKGROUND
    # =====================================================

    background = VideoBackground(
        "output_trimmed.mp4"
    )

    # =====================================================
    # HAND DETECTOR
    # =====================================================

    hand_detector = HandDetector(
        "models/hand_landmarker.task"
    )

    # =====================================================
    # GESTURE RECOGNIZER
    # =====================================================

    gesture_recognizer = (
        GestureRecognizer()
    )

    # =====================================================
    # CUT CONTROLLER
    # =====================================================

    cut_controller = CutController()

    # =====================================================
    # GRID CONTROLLER
    # =====================================================

    grid_controller = GridController()

    # =====================================================
    # INTERACTION CONTROLLER
    # =====================================================

    draw_controller = DrawController()

    interaction_controller = (
        InteractionController(
            hand_detector,
            gesture_recognizer,
            cut_controller,
            grid_controller,
            draw_controller
        )
    )

    # =====================================================
    # OVERLAY
    # =====================================================

    overlay = GestureOverlay(
        cut_controller,
        grid_controller,
        draw_controller
    )

    # =====================================================
    # MAIN LOOP
    # =====================================================

    while True:

        # -------------------------------------------------
        # READ WEBCAM
        # -------------------------------------------------

        ret, frame = camera.read()

        if not ret:

            print(
                "Error: Could not read webcam frame."
            )

            break

        # -------------------------------------------------
        # MIRROR WEBCAM
        # -------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )

        height, width = (
            frame.shape[:2]
        )

        # =================================================
        # PROCESS HAND
        # =================================================

        hand_result = (
            interaction_controller.process_hand(
                frame
            )
        )

        # =================================================
        # HUMAN MASK
        # =================================================

        mask = segmenter.get_mask(
            frame
        )

        # =================================================
        # VIDEO FRAME
        # =================================================

        video_frame = background.get_frame(
            width,
            height
        )

        if video_frame is None:

            print(
                "Error: Could not get video frame."
            )

            break

        # =================================================
        # CUT PART AS BACKGROUND (if the button was used)
        # =================================================

        video_frame = (
            cut_controller.apply_background(
                video_frame
            )
        )

        # =================================================
        # APPLY CUT + HUMAN
        # =================================================

        result = cut_controller.apply_cut(
            video_frame,
            frame,
            mask
        )

        # =================================================
        # GRID
        #
        # Drawn AFTER apply_cut so grid lines never get
        # copied into the floating cut.
        # =================================================

        grid_controller.draw(
            result
        )

        # =================================================
        # WRITING (index finger) - on top of the picture
        # =================================================

        result = draw_controller.apply(
            result
        )

        # =================================================
        # UI OVERLAY
        # =================================================

        result = overlay.draw(
            result,
            hand_result
        )

        # =================================================
        # DISPLAY
        # =================================================

        cv2.imshow(
            "Human in Video",
            result
        )

        # =================================================
        # KEYBOARD
        # =================================================

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )

        # -------------------------------------------------
        # ESC
        # -------------------------------------------------

        if key == 27:

            break

        # -------------------------------------------------
        # RESET
        # -------------------------------------------------

        if key == ord("r"):

            interaction_controller.reset()

    # =====================================================
    # CLEANUP
    # =====================================================

    camera.release()

    background.close()

    segmenter.close()

    hand_detector.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()