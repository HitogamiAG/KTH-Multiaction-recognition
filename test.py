import numpy as np
from tqdm import tqdm
import argparse

import torch
from torch import nn

from sklearn.utils.class_weight import compute_class_weight
import os
import sys
from time import time

from data_scripts.dataloader import create_dataloader
from models.model import LSTMModel
from engine import train_step, val_step, evaluate
from local_utils import load_model, report

parser = argparse.ArgumentParser('Parse train parameters')
parser.add_argument('-tp', '--test_path', type=str, default=None, help='Path to the folder with test videos')
parser.add_argument('-numc', '--num_classes', type=int, default=6, help='Number of classes')
parser.add_argument('-numf', '--num_features', type=int, default=30, help='Number of values in one record')
parser.add_argument('-numr', '--num_records', type=int, default=15, help='Sequence length')
parser.add_argument('-s', '--step', type=int, default=5, help='Step between frames')
parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('-d', '--device', type=str, default='cpu', help='Device (CPU/GPU)')
parser.add_argument('-mp', '--model_path', type=str, default=None, help='Path to the model weights')

args = parser.parse_args()

if not args.test_path:
    print('Test data path is not specified.')
    sys.exit()

if not args.model_path:
    print('Model path is not specified.')
    sys.exit()

test_shuffle = False
num_workers = os.cpu_count()

test_dataloader = create_dataloader(
    args.test_path, args.num_records, args.step, args.batch_size, test_shuffle, num_workers)

model = LSTMModel(num_classes=args.num_classes, input_size=args.num_features,
                  num_layers=1, seq_length=args.num_records)
model = load_model(model, args.model_path)
loss_fn = nn.CrossEntropyLoss()

y_pred, y, test_loss, test_acc = evaluate(
    model, test_dataloader, loss_fn, args.device)
print(f'Test loss: {test_loss} | Test acc: {test_acc}')

report(y_pred, y)
