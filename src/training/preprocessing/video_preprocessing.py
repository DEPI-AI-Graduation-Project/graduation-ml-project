
import cv2
import numpy as np


class VideoPreprocessor:
    def __init__(
        self,
        num_frames=16,
        image_size=224
    ):
        self.num_frames = num_frames
        self.image_size = image_size

    def resize_with_padding(self, frame):

        height, width = frame.shape[:2]

        scale = min(
            self.image_size / width,
            self.image_size / height
        )

        new_width = int(width * scale)
        new_height = int(height * scale)

        resized = cv2.resize(
            frame,
            (new_width, new_height)
        )

        canvas = np.zeros(
            (self.image_size, self.image_size, 3),
            dtype=np.uint8
        )

        x_offset = (self.image_size - new_width) // 2
        y_offset = (self.image_size - new_height) // 2

        canvas[
            y_offset:y_offset + new_height,
            x_offset:x_offset + new_width
        ] = resized

        return canvas

    def process_video(self, video_path):

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:
            cap.release()
            raise ValueError("Video contains no readable frames")

        indices = np.linspace(
            0,
            total_frames - 1,
            self.num_frames,
            dtype=int
        )

        frames = []

        for idx in indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))

            success, frame = cap.read()

            if not success:
                cap.release()
                raise ValueError(
                    f"Could not read frame {idx}"
                )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame = self.resize_with_padding(frame)

            frames.append(frame)

        cap.release()

        video_tensor = np.array(
            frames,
            dtype=np.float32
        )

        video_tensor /= 255.0

        return video_tensor
