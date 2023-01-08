import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, path: str, num_recors_in_input: int, step: int) -> None:
        super(CustomDataset, self).__init__()
        self.labels = ['boxing', 'jogging', 'running',
                       'walking', 'handclapping', 'handwaving']
        self.dataframe = pd.read_csv(path, index_col=0).reset_index()
        self.num_records_in_input = num_recors_in_input
        self.step = step
        self.indices_to_input = []
        self.classes = []

        self.extract_input_indices()

    def extract_input_indices(self):
        grouped_df = self.dataframe.groupby('src_video').agg(
            {'Nose_X': 'count', 'action': 'first', 'index': 'min'}).rename(columns={'Nose_X': 'n_frames'})

        for record in grouped_df.iterrows():
            n_frames, action, index = record[1]
            if n_frames < self.num_records_in_input:
                continue

            curr_index = index
            while (curr_index + self.num_records_in_input) < (index + n_frames):
                self.indices_to_input.append(curr_index)
                self.classes.append(self.labels.index(action))
                curr_index += self.step

    def __len__(self):
        return len(self.indices_to_input)

    def __getitem__(self, index):
        internal_index = self.indices_to_input[index]
        label = self.classes[index]
        X = self.dataframe.iloc[internal_index:internal_index+self.num_records_in_input,
                                :].drop(['index', 'action', 'src_video'], axis=1).to_numpy(dtype=np.float32)

        X = np.abs(X - X[0])

        return X * 100, label


def create_dataloader(path_to_data: str, num_records_in_input: int, step: int, batch_size: int, shuffle: bool, num_workers: int):
    dataset = CustomDataset(path_to_data, num_records_in_input, step)
    dataloader = DataLoader(dataset, batch_size,
                            shuffle=shuffle, num_workers=num_workers)
    return dataloader
