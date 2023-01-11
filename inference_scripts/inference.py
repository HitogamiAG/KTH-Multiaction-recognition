import torch
import numpy as np
from torch import nn

labels = ['boxing', 'jogging', 'running',
                       'walking', 'handclapping', 'handwaving']

def most_frequent(List):
    return max(set(List), key=List.count)


def predict(ort_session, dataframe, seq_length, step):

    dataframe = dataframe.to_numpy().astype(np.float32)
    X = []
    n_frames = len(dataframe)
    indices_to_input = np.arange(0, n_frames - seq_length, step)
    for index in indices_to_input:
        X.append(
            np.abs(dataframe[index:index + seq_length] - dataframe[index]) * 100)

    X = np.array(X, dtype=np.float32)

    y_pred_concatenated = np.array([], dtype=np.float32).reshape(0, 6)
    
    for i in range(X.shape[0] // 8):
        y_pred = ort_session.run(None, {'input':X[8*i:8*(i+1)]})[0]
        y_pred_concatenated = np.concatenate([y_pred_concatenated, y_pred], axis=0)

    y_pred = torch.from_numpy(y_pred_concatenated)

    classes = torch.argmax(torch.softmax(y_pred, axis=1), axis=1)
    classes = [int(class_.item()) for class_ in classes]

    predicted_class = labels[most_frequent(classes)]
    return predicted_class
