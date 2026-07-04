"""
Pose & Hand Landmark Extractor using MediaPipe Holistic.

Processes sign language video frames to extract body pose (33 landmarks),
left hand (21 landmarks), and right hand (21 landmarks) per frame.
"""

import cv2
import mediapipe as mp
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PoseExtractor:
    """Extracts pose and hand landmarks from sign language videos using MediaPipe Holistic."""

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_holistic = mp.solutions.holistic

    def process_video(self, video_path: str, sample_rate: int = 1) -> dict:
        """
        Process a video file and extract pose/hand landmarks for every sampled frame.

        Args:
            video_path: Absolute path to the video file.
            sample_rate: Process every Nth frame (1 = every frame, 2 = every other, etc.)

        Returns:
            Dictionary with keys: fps, total_frames, width, height, frames.
            Each frame contains pose, left_hand, right_hand landmark arrays.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Video info: {width}x{height} @ {fps:.1f}fps, {total_frames} frames"
        )

        result_data = {
            "fps": fps / sample_rate,
            "original_fps": fps,
            "width": width,
            "height": height,
            "total_frames": 0,
            "original_total_frames": total_frames,
            "frames": [],
        }

        with self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        ) as holistic:

            frame_idx = 0
            processed = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate != 0:
                    frame_idx += 1
                    continue

                # MediaPipe requires RGB input
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = holistic.process(rgb_frame)

                frame_data = self._extract_landmarks(results)
                result_data["frames"].append(frame_data)

                processed += 1
                if processed % 30 == 0:
                    logger.info(f"Processed {processed} frames...")

                frame_idx += 1

        cap.release()
        result_data["total_frames"] = len(result_data["frames"])
        logger.info(
            f"Extraction complete: {result_data['total_frames']} frames processed"
        )

        return result_data

    def _extract_landmarks(self, results) -> dict:
        """Extract pose, left hand, and right hand landmarks from MediaPipe results."""
        frame_data = {
            "pose": None,
            "left_hand": None,
            "right_hand": None,
        }

        # 33 pose landmarks
        if results.pose_landmarks:
            frame_data["pose"] = [
                {
                    "x": round(lm.x, 6),
                    "y": round(lm.y, 6),
                    "z": round(lm.z, 6),
                    "v": round(lm.visibility, 4),
                }
                for lm in results.pose_landmarks.landmark
            ]

        # 21 left-hand landmarks
        if results.left_hand_landmarks:
            frame_data["left_hand"] = [
                {
                    "x": round(lm.x, 6),
                    "y": round(lm.y, 6),
                    "z": round(lm.z, 6),
                }
                for lm in results.left_hand_landmarks.landmark
            ]

        # 21 right-hand landmarks
        if results.right_hand_landmarks:
            frame_data["right_hand"] = [
                {
                    "x": round(lm.x, 6),
                    "y": round(lm.y, 6),
                    "z": round(lm.z, 6),
                }
                for lm in results.right_hand_landmarks.landmark
            ]

        return frame_data
