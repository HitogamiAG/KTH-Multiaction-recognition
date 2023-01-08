import torch
import numpy as np
from torch import nn

def most_frequent(List):
    return max(set(List), key = List.count)

def predict(model, dataframe, seq_length, step):
    model.eval()
    
    dataframe = dataframe.to_numpy().astype(np.float32)
    X = []
    n_frames = len(dataframe)
    indices_to_input = np.arange(0, n_frames - seq_length, step)
    for index in indices_to_input:
        X.append(np.abs(dataframe[index:index + seq_length] - dataframe[index]) * 100)
        
    X = np.array(X, dtype=np.float32)
    X = torch.from_numpy(X)
    
    with torch.inference_mode():
        y_pred = model(X)
        
    classes = torch.argmax(torch.softmax(y_pred, axis=1),axis=1)
    classes = [int(class_.item()) for class_ in classes]
    
    predicted_class = model.labels[most_frequent(classes)]
    return predicted_class
    
    