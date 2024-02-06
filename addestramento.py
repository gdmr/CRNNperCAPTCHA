import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from dataset import Dataset
from ocr import OCR
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


TRAIN_DIR = #directory del dataset
VAL_DIR = #directory del validationset
BATCH_SIZE = 8
N_WORKERS = 0
CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWYXZ'
VOCAB_SIZE = len(CHARS) + 1

lr = 0.02
weight_decay = 1e-5
momentum = 0.7

EPOCHS = 10

# Dataset
train_dataset = Dataset(TRAIN_DIR)
val_dataset = Dataset(VAL_DIR)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, 
    num_workers=N_WORKERS, shuffle=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, 
    num_workers=N_WORKERS, shuffle=False
)

# Model
ocr = OCR()

# Optimizer
optimizer = optim.SGD(
    ocr.crnn.parameters(), lr=lr, nesterov=True,
    weight_decay=weight_decay, momentum=momentum
)


# train
train_losses, val_losses = ocr.train(EPOCHS, optimizer, train_loader, val_loader, print_every=1)

# view samples
sample_result = []

for i in range(10):
    idx = np.random.randint(len(val_dataset))
    img, label = val_dataset[idx]
    logits = ocr.predict(img.unsqueeze(0))
    pred_text = ocr.decode(logits.cpu())

    sample_result.append( (img, label, pred_text) )

fig = plt.figure(figsize=(17, 5))
for i in range(10):
    ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])

    img, label, pred_text = sample_result[i]
    title = f'Truth: {label} | Pred: {pred_text}'

    ax.imshow(img.permute(1, 2, 0))
    ax.set_title(title)

plt.show() 


# plot loss stats
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Valid Loss')
plt.title('Loss stats')
plt.legend()
plt.show()