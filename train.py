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
from engine import train_step, val_step
from utils import save_model

parser = argparse.ArgumentParser('Parse train parameters')
parser.add_argument('-tp', '--train_path', type=str, default=None)
parser.add_argument('-vp', '--val_path', type=str, default=None)
parser.add_argument('-numc', '--num_classes', type=int, default=6)
parser.add_argument('-numf', '--num_features', type=int, default=30)
parser.add_argument('-numr', '--num_records', type=int, default=15)
parser.add_argument('-nume', '--num_epochs', type=int, default=100)
parser.add_argument('-bw', '--balanced_class_weights', type=bool, default=True)
parser.add_argument('-s', '--step', type=int, default=5)
parser.add_argument('-b', '--batch_size', type=int, default=8)
parser.add_argument('-d', '--device', type=str, default='cpu')
parser.add_argument('-mn', '--model_name', type=str,
                    default='model' + str(round(time())))

args = parser.parse_args()

train_shuffle = True
val_shuffle = False
num_workers = os.cpu_count()

if not args.train_path:
    print('Train path is not specified. Stopping...')
    sys.exit()

train_dataloader = create_dataloader(
    args.train_path, args.num_records, args.step, args.batch_size, train_shuffle, num_workers)

if args.val_path:
    val_dataloader = create_dataloader(
        args.val_path, args.num_records, args.step, args.batch_size, val_shuffle, num_workers)

model = LSTMModel(num_classes=args.num_classes, input_size=args.num_features,
                  num_layers=1, seq_length=args.num_records)

if args.balanced_class_weights:
    class_weights = compute_class_weight('balanced', classes=np.unique(
        train_dataloader.dataset.classes), y=train_dataloader.dataset.classes)
    class_weights = torch.from_numpy(class_weights).type(
        torch.float32).to(args.device)
    loss_fn = nn.CrossEntropyLoss(class_weights, reduction='mean')
else:
    loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.RMSprop(
    model.parameters(), lr=1e-4, weight_decay=1e-5, momentum=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

for epoch in tqdm(range(1, args.num_epochs+1)):
    train_loss, train_acc = train_step(
        model, train_dataloader, loss_fn, optimizer, args.device)

    if epoch % 5 == 0:
        print(
            f'Epoch: {epoch} | Train loss: {train_loss} | Train acc: {train_acc}')

    if args.val_path:
        val_loss, val_acc = val_step(
            model, val_dataloader, loss_fn, args.device)
        if epoch % 5 == 0:
            print(f'Val loss: {val_loss} | Val acc: {val_acc}')

    scheduler.step()

save_model(model, 'checkpoints/', args.model_name)
