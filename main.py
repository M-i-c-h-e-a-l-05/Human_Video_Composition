import cv2

from human_segmenter import HumanSegmenter
from video_background import VideoBackground
from gesture_controller import GestureController


def main():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("Error: Could not open webcam.")
        return

    segmenter = HumanSegmenter()

    background = VideoBackground(
        "output_trimmed.mp4"
    )

    gesture = GestureController()

    while True:

        # =================================================
        # 1. WEBCAM
        # =================================================

        ret, frame = camera.read()

        if not ret:

            print(
                "Error: Could not read webcam frame."
            )

            break

        # Mirror webcam
        frame = cv2.flip(
            frame,
            1
        )

        height, width = frame.shape[:2]

        # =================================================
        # 2. HAND / GESTURE DETECTION
        # =================================================

        gesture.process_hand(frame)

        # =================================================
        # 3. HUMAN SEGMENTATION
        # =================================================

        mask = segmenter.get_mask(frame)

        # =================================================
        # 4. CURRENT VIDEO FRAME
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
        # 5. LIVE CUT COMPOSITION
        #
        # Human = highest
        # Cut video = middle
        # Video = lowest
        # =================================================

        result = gesture.apply_cut(
            video_frame,
            frame,
            mask
        )

        # =================================================
        # 6. DRAW GLOWING RECTANGLE
        #
        # Highest visual overlay
        # =================================================

        result = gesture.draw_overlay(
            result
        )

        # =================================================
        # 7. DISPLAY
        # =================================================

        cv2.imshow(
            "Human in Video",
            result
        )

        key = cv2.waitKey(1) & 0xFF

        # ESC
        if key == 27:

            break

        # R = reset cut
        if key == ord("r"):

            gesture.reset()

    # =====================================================
    # CLEANUP
    # =====================================================

    camera.release()

    background.close()

    segmenter.close()

    gesture.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()