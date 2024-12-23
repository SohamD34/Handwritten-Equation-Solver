import os 
os.chdir('/home/bcca/Desktop/Soham/DL Course Project/Handwritten-Equation-Solver/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, ReLU, Module, Dropout, Sigmoid, Linear, BatchNorm2d
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, Resize, ToTensor, Normalize
import torchvision.utils as vutils
import pickle
import os
from PIL import Image
import cv2
import json
from utils.utils import adversarial_train, adversarial_validate, fgsm_attack
from src.CharacterNet import CharacterNet


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print("Device set to", device)

batch_size = 128
transformations = Compose([Grayscale(num_output_channels=1),
                           Resize((28,28)),
                           ToTensor()])



# IMPORTING THE DATA

data = ImageFolder(root='../Handwritten-Equation-Solver/data/character_data',transform=transformations)

mapping = {}
for idx, (class_name, _) in enumerate(data.class_to_idx.items()):

    if(class_name[0]=='_' and class_name[2]=='_'):
        mapping[idx] = class_name[1]
    else:
        mapping[idx] = class_name
    data.classes[idx] = idx

train_data, val_data = train_test_split(data, test_size=0.3, shuffle=True)

with open('my_dict.json', 'w') as f:
    json.dump(mapping, f)



# CREATING THE DATALOADER

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

for i, (images, labels) in enumerate(train_loader):
    print("Image size =",images.shape)
    print("Label size =",labels.shape)
    img = images[0]
    plt.figure(figsize=(2.5,2.5))
    plt.imshow(img[0], cmap='gray')
    plt.show()
    break


# MODEL INITIALIZATION

model = CharacterNet()
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable Parameters:", trainable_params)





# ---------------------------------- ADVERSARIAL TRAINING WITHOUT SCHEDULER --------------------------------------------- #

model = CharacterNet().to(device)
num_epochs = 200
optimizer = Adam(model.parameters(), lr =0.001)
criterion = nn.CrossEntropyLoss()
epsilona = 0.3

training_losses = []
val_losses = []
training_accuracies = []
val_accuracies = []

klm = 0
                                                                                  # Training loop with adversarial training
for epoch in range(num_epochs):
    print("Epoch",epoch, end=" ")
    klm+=1
    train_loss, train_acc = adversarial_train(model, data, device, train_loader, optimizer, criterion, epsilona, klm, custom=False)
    val_loss, val_acc = adversarial_validate(model, device, val_loader, criterion)

    training_losses.append(train_loss)
    val_losses.append(val_loss)
    training_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

plt.plot(training_losses, label='Train loss')
plt.plot(val_losses, label='Val loss')
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.legend()
plt.savefig('results/adversarial_train/No_Scheduler/train_val_loss.png')





# ---------------------------------- ADVERSARIAL TRAINING WITH ReduceLROnPlateau SCHEDULER --------------------------------------------- #

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

model = CharacterNet().to(device)
num_epochs = 100
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
epsilona = 0.3

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

klm =0
training_losses = []
val_losses = []
training_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    print("Epoch", epoch, end=" ")
    klm+=1
    train_loss, train_acc = adversarial_train(model, data, device, train_loader, optimizer, criterion, epsilona, klm, custom=False)
    val_loss, val_acc = adversarial_validate(model, device, val_loader, criterion)

    training_losses.append(train_loss)
    val_losses.append(val_loss)
    training_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    scheduler.step(val_loss)                    


plt.plot(training_losses, label='Train loss')
plt.plot(val_losses, label='Val loss')
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.legend()
plt.savefig('results/adversarial_train/ReduceLROnPlateau/train_val_loss.png')

model_path = "models/ScheduledCharacterNet_model.pth"
torch.save(model.state_dict(), model_path)





# ---------------------------------- ADVERSARIAL TRAINING WITH CUSTOM SCHEDULER --------------------------------------------- #

model = CharacterNet().to(device)
num_epochs = 100
optimizer = Adam(model.parameters(), lr =0.01)
criterion = nn.CrossEntropyLoss()
epsilona = 0.3
training_losses = []
val_losses = []
training_accuracies = []
val_accuracies = []

klm = 0
prev_loss = -1                                                                                  # Training loop with adversarial training
for epoch in range(num_epochs):
    print("Epoch",epoch, end=" ")
    klm+=1
    train_loss, train_acc = adversarial_train(model, data, device, train_loader, optimizer, criterion, epsilona, klm, custom=True)
    val_loss, val_acc = adversarial_validate(model, device, val_loader, criterion)
    
    training_losses.append(train_loss)
    val_losses.append(val_loss)
    training_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    if(train_acc >= 0.92):
        break


plt.plot(training_losses, label='Train loss')
plt.plot(val_losses, label='Val loss')
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.legend()
plt.savefig('results/adversarial_train/Custome_Scheduler/train_val_loss.png')

model_path = "models/CustomScheduledCharacterNet_model.pth"
torch.save(model.state_dict(), model_path)