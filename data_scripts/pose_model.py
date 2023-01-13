import numpy as np
import cv2
import mediapipe as mp
from typing import Optional, List


class PoseEstimator():
    def __init__(self, min_detection_confidence: Optional[float] = .5, min_tracking_confidence: Optional[float] = .5) -> None:
        """To initialize parameters for mediapipe pose estimation model

        Args:
            min_detection_confidence (Optional[float], optional): Detection confidence. Defaults to .5.
            min_tracking_confidence (Optional[float], optional): Tracking confidence. Defaults to .5.
        """
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

    def __call__(self, img: np.ndarray, revert_channels: Optional[bool] = False) -> List:
        """Returns the list of landmarks of estimated pose

        Args:
            img (np.ndarray): Image array
            revert_channels (Optional[bool], optional): Revert channels from BGR to RGB. Defaults to False.

        Returns:
            List: List of estimated pose landmarks
        """

        if revert_channels:
            RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            RGB = img

        pose_landmarks = self.pose.process(RGB)
        return pose_landmarks

    def draw_landmarks(self, frame: np.ndarray, pose_landmarks: np.ndarray):
        """Draws landmarks on the frame of the video

        Args:
            frame (np.ndarray): Image array
            pose_landmarks (np.ndarray): List of estimated pose landmarks
        """
        self.mp_drawing.draw_landmarks(
            frame, pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
