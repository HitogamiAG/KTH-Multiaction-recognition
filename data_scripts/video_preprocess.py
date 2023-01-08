import argparse
import pandas as pd
import os
from pose_model import PoseEstimator
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import cv2
import mediapipe as mp


def process_video(video_path: str, pose_estimator: PoseEstimator, frames_diff: int):
    cap = cv2.VideoCapture(video_path)

    frame_counter = 0
    results = []
    while cap.isOpened():
        _, frame = cap.read()

        if frame_counter % frames_diff == 0:
            try:
                pose_landmarks = pose_estimator(
                    img=frame, revert_channels=True)
                if not pose_landmarks.pose_landmarks:
                    continue
                results.append(pose_landmarks)
            except:
                break
        if cv2.waitKey(1) == 'q':
            break

        frame_counter += 1

    return results


def video_to_csv(data_path: str,
                 csv_name: str,
                 pose_estimator: PoseEstimator,
                 wish_landmarks: dict,
                 frames_diff: Optional[int] = 5,
                 rewrite_exist: Optional[bool] = False):
    data_path = Path(data_path)

    csv_dict = {}
    for landmark_name in wish_landmarks:
        csv_dict[f'{landmark_name}_X'] = []
        csv_dict[f'{landmark_name}_Y'] = []
    csv_dict['action'], csv_dict['src_video'] = [], []

    if not csv_name.endswith('.csv'):
        csv_name += '.csv'

    path_to_csv: Path = data_path / csv_name
    if path_to_csv.is_file() and not rewrite_exist:
        print(f'File {csv_name} already extists. Skipping...')
        return

    class_names = [folder for folder in os.listdir(
        data_path) if os.path.isdir(data_path / folder)]
    print(
        f'In {str(data_path)} found the folloding classes: {", ".join(class_names)}')
    for class_ in class_names:
        list_videos = os.listdir(data_path / class_)
        print(f'In class {class_} found {len(list_videos)} videos')

        for video_name in tqdm(list_videos):
            path_to_video = data_path / class_ / video_name
            landmarks_from_video = process_video(video_path=str(
                path_to_video), pose_estimator=pose_estimator, frames_diff=frames_diff)
            for landmarks in landmarks_from_video:
                for landmark_name, index in wish_landmarks.items():
                    csv_dict[f'{landmark_name}_X'].append(
                        round(landmarks.pose_landmarks.landmark[index].x, 4))
                    csv_dict[f'{landmark_name}_Y'].append(
                        round(landmarks.pose_landmarks.landmark[index].y, 4))
                csv_dict['action'].append(class_)
                csv_dict['src_video'].append(video_name)

    if len(csv_dict):
        df_to_csv = pd.DataFrame(csv_dict)
        df_to_csv.to_csv(path_to_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Video data processing script')
    parser.add_argument('-trdp', '--train_data_path', type=str, default=None)
    parser.add_argument('-vdp', '--val_data_path', type=str, default=None)
    parser.add_argument('-tsdp', '--test_data_path', type=str, default=None),
    parser.add_argument('-trcn', '--train_csv_name',
                        type=str, default='train_data')
    parser.add_argument('-vcn', '--val_csv_name', type=str, default='val_data')
    parser.add_argument('-tscn', '--test_csv_name',
                        type=str, default='test_data')
    parser.add_argument('-rw', '--rewrite_csv', type=bool, default=False)
    parser.add_argument('-tconf', '--track_confidence',
                        type=float, default=0.1)
    parser.add_argument('-dconf', '--detect_confidence',
                        type=float, default=0.1)
    parser.add_argument('-fd', '--frames_diff', type=int, default=3)

    args = parser.parse_args()

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

    if args.train_data_path:
        video_to_csv(args.train_data_path, args.train_csv_name, PoseEstimator(min_tracking_confidence=args.track_confidence,
                                                                              min_detection_confidence=args.detect_confidence), landmarks_to_get_from_video, frames_diff=args.frames_diff)

    if args.val_data_path:
        video_to_csv(args.val_data_path, args.val_csv_name, PoseEstimator(min_tracking_confidence=args.track_confidence,
                                                                          min_detection_confidence=args.detect_confidence), landmarks_to_get_from_video, frames_diff=args.frames_diff)

    if args.test_data_path:
        video_to_csv(args.test_data_path, args.test_csv_name, PoseEstimator(min_tracking_confidence=args.track_confidence,
                                                                            min_detection_confidence=args.detect_confidence), landmarks_to_get_from_video, frames_diff=args.frames_diff)
