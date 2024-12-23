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

data = ImageFolder(root='data/character_data',transform=transformations)

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
criterion = nn.CrossEntropyLoss()


# UNSCHEDULED CHARACTER NET MODEL

count = 0
total = 0
base_model = CharacterNet()

base_model.load_state_dict(torch.load('models/UnscheduledCharacterNet_model.pth'))
base_model.to(device)

with torch.no_grad():
    for images, labels in val_loader:
        
        images = images.to(device)
        labels = labels.to(device)

        outputs = base_model(images)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)

        for i in range(len(images)):
            total += 1
            img = images[i].cpu().numpy()
            pred = predicted[i].cpu().item()
            label = labels[i].cpu().item()

            if(label == pred):
                count += 1
        break

print("Accuracy of Unscheduled Character Net =",count/total)



# SCHEDULED CHARACTER NET MODEL

count = 0
total = 0
model1 = CharacterNet()

model1.load_state_dict(torch.load('models/ScheduledCharacterNet_model.pth'))
model1.to(device)

with torch.no_grad():
    for images, labels in val_loader:
        
        images = images.to(device)
        labels = labels.to(device)

        outputs = model1(images)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)

        for i in range(len(images)):
            total += 1
            img = images[i].cpu().numpy()
            pred = predicted[i].cpu().item()
            label = labels[i].cpu().item()

            if(label == pred):
                count += 1
        break

print("Accuracy of Scheduled Character Net=",count/total)


# CUSTOM SCHEDULED CHJARACTER NET MODEL


count = 0
total = 0
model2 = CharacterNet()

model2.load_state_dict(torch.load('models/CustomScheduledCharacterNet_model.pth'))
model2.to(device)

with torch.no_grad():
    for images, labels in val_loader:
        
        images = images.to(device)
        labels = labels.to(device)

        outputs = model2(images)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)

        for i in range(len(images)):
            total += 1
            img = images[i].cpu().numpy()
            pred = predicted[i].cpu().item()
            label = labels[i].cpu().item()

            if(label == pred):
                count += 1
        break

print("Accuracy of Custom Scheduled Character Net =",count/total)


# RANDOM TESTING A FEW TEST IMAGES

for i in range(5,7):
    
    ii = cv2.imread('../Handwritten-Equation-Solver/test_data/'+ str(i) + '.jpeg')
    gray_image = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (28,28))
    gray_image = gray_image.reshape((1,28,28))
    data = torch.Tensor(np.array([gray_image])).to(device)
    output = base_model(data)
    _, predicted = torch.max(output.data, 1)
    plt.figure(figsize=(3,3))
    plt.imshow(gray_image[0])
    plt.show()
    print(mapping[predicted.cpu().numpy()[0]])

for i in range(7,9):

    ii = cv2.imread('../Handwritten-Equation-Solver/test_data/'+ str(i) + '.png')
    gray_image = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (28,28))
    gray_image = gray_image.reshape((1,28,28))
    data = torch.Tensor(np.array([gray_image])).to(device)
    output = base_model(data)
    _, predicted = torch.max(output.data, 1)
    plt.figure(figsize=(3,3))
    plt.imshow(gray_image[0])
    plt.show()
    print(mapping[predicted.cpu().numpy()[0]])

for i in range(9,11):

    ii = cv2.imread('../Handwritten-Equation-Solver/test_data/'+ str(i) + '.jpeg')
    gray_image = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (28,28))
    gray_image = gray_image.reshape((1,28,28))
    data = torch.Tensor(np.array([gray_image])).to(device)
    output = base_model(data)
    _, predicted = torch.max(output.data, 1)
    plt.figure(figsize=(3,3))
    plt.imshow(gray_image[0])
    plt.show()
    print(mapping[predicted.cpu().numpy()[0]])