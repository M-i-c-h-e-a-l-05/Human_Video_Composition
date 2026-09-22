import cv2


class VideoBackground:

    def __init__(self, video_path):

        self.video = cv2.VideoCapture(video_path)

        if not self.video.isOpened():
            raise ValueError(
                f"Could not open video: {video_path}"
            )

    def get_frame(self, width, height):

        ret, frame = self.video.read()

        # Restart video when it reaches the end
        if not ret:

            self.video.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            ret, frame = self.video.read()

        if not ret:
            return None

        # Resize to webcam resolution
        frame = cv2.resize(
            frame,
            (width, height)
        )

        return frame

    def close(self):

        self.video.release()