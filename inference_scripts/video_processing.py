import cv2
import av
import pandas as pd
import numpy as np
import mediapipe as mp
from io import BytesIO
from typing import Optional, List

# initialize Pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

landmarks_to_get_from_video = {'Nose': 0,
                               'LShoulder': 11,
                               'RShoulder': 12,
                               'LElbow': 13,
                               'RElbow': 14,
                               'LWrist': 15,
                               'RWrist': 16,
                               'LHip': 23,
                               'RHip': 24,
                               'LKnee': 25,
                               'RKnee': 26,
                               'LAnkle': 27,
                               'RAnkle': 28,
                               'LFoot': 31,
                               'RFoot': 32}

width, height, fps = 160, 120, 25


def track_points_to_dataframe(trackpoints: List, frames_diff: int):
    trackpoints = trackpoints[::frames_diff]
    df_dict = {}

    for landmark_name in landmarks_to_get_from_video:
        df_dict[f'{landmark_name}_X'] = []
        df_dict[f'{landmark_name}_Y'] = []

    for landmarks in trackpoints:
        for landmark_name, index in landmarks_to_get_from_video.items():
            df_dict[f'{landmark_name}_X'].append(
                round(landmarks.pose_landmarks.landmark[index].x, 4))
            df_dict[f'{landmark_name}_Y'].append(
                round(landmarks.pose_landmarks.landmark[index].y, 4))

    return pd.DataFrame(df_dict)


def process_video(path: str,
                  detect_conf: Optional[float] = 0.5,
                  track_conf: Optional[float] = 0.5,
                  model_complexity: Optional[int] = 1,
                  frames_diff: Optional[int] = 5):

    track_points = []

    pose = mp_pose.Pose(
        min_detection_confidence=detect_conf,
        min_tracking_confidence=track_conf, model_complexity=model_complexity)

    cap = cv2.VideoCapture(path)

    output_memory_file = BytesIO()
    output = av.open(output_memory_file, 'w', format="mp4")
    stream = output.add_stream('h264', str(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': '17'}

    while cap.isOpened():
        _, frame = cap.read()

        try:
            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(RGB)
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if results.pose_landmarks:
                track_points.append(results)

            packet = stream.encode(
                av.VideoFrame.from_ndarray(frame, format='bgr24'))
            output.mux(packet)
        except:
            break
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    packet = stream.encode(None)
    output.mux(packet)
    output.close()
    output_memory_file.seek(0)

    dataframe = track_points_to_dataframe(track_points, frames_diff)

    return output_memory_file, dataframe
