import numpy as np
import cv2
import mediapipe as mp
from typing import Optional, List


class PoseEstimator():
    def __init__(self, min_detection_confidence: Optional[float] = .5, min_tracking_confidence: Optional[float] = .5) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

    def __call__(self, img: np.ndarray, revert_channels: Optional[bool] = False) -> List:

        if revert_channels:
            RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            RGB = img

        pose_landmarks = self.pose.process(RGB)
        return pose_landmarks

    def draw_landmarks(self, frame: np.ndarray, pose_landmarks: np.ndarray):
        self.mp_drawing.draw_landmarks(
            frame, pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
