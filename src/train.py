import os
os.chdir('/home/bcca/Desktop/Soham/DL Course Project/Handwritten-Equation-Solver/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
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
from PIL import Image
import cv2
import json
from utils.utils import plot_image
from src.CharacterNet import CharacterNet
import warnings
warnings.filterwarnings("ignore")



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print("Device set to", device)

batch_size = 128
transformations = Compose([Grayscale(num_output_channels=1),
                           Resize((28,28)),
                           ToTensor()])



# LOADING THE DATA

data = ImageFolder(root='../Handwritten-Equation-Solver/data/character_data',transform=transformations)

mapping = {}
for idx, (class_name, _) in enumerate(data.class_to_idx.items()):

    if(class_name[0]=='_' and class_name[2]=='_'):
        mapping[idx] = class_name[1]
    else:
        mapping[idx] = class_name
    data.classes[idx] = idx

with open('my_dict.json', 'w') as f:
    json.dump(mapping, f)



# CREATING THE DATALOADER

train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

for i, (images, labels) in enumerate(train_loader):
    print("Image size =",images.shape)
    print("Label size =",labels.shape)
    img = images[0]
    plt.figure(figsize=(2.5,2.5))
    plt.imshow(img[0], cmap='gray')
    plt.show()
    break


# MODEL INITIALIZATION

model = CharacterNet().to(device)

num_epochs = 125
num_classes = len(data.classes)   # 107
learning_rate = 1e-3

criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)


# MODEL TRAINING

total_step = len(train_loader)
loss_list = []
acc_list = []

for epoch in range(num_epochs):

        total_correct = 0
        total_images = 0
        running_loss = 0.0

        all_predicted = []
        all_expected = []
    
        for i, (images, labels) in enumerate(train_loader):
    
            images = images.to(device)
 
            encoder = OneHotEncoder(categories=[data.classes])
            labels_encoded = encoder.fit_transform(labels.reshape(-1, 1)).toarray()
            labels_encoded = torch.Tensor(labels_encoded).to(device)
    
            outputs = model(images)

            loss = criterion(outputs, labels_encoded)
            running_loss += loss.item()
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            _, expected = torch.max(labels_encoded, 1)
            correct = (predicted == expected).sum().item()

            all_expected.extend(list(expected.cpu().numpy()))
            all_predicted.extend(list(predicted.cpu().numpy()))
            
            total_correct += correct
            total_images += total

        acc_list.append(total_correct / total_images)
        loss_list.append(running_loss / total_step)
        
        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
              .format(epoch+1, num_epochs, (running_loss/len(train_loader)), (total_correct / total_images) * 100))
        

plt.figure(figsize=(10,4))
plt.plot(acc_list)
plt.title("Accuracy v/s Epochs")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('results/train/train_accuracy.png')


plt.figure(figsize=(10,4))
plt.plot(loss_list)
plt.title("Training loss v/s Epochs")
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.savefig('results/train/train_loss.png')


# VALIDATION

# img = Image.open('../test_data/6.jpeg').resize((128,128)).convert('L')   5 to 9

for i in range(5,7):
    ii = cv2.imread('../Handwritten-Equation-Solver/test_data/'+ str(i) + '.jpeg')
    plot_image(ii)

for i in range(7,9):
    ii = cv2.imread('../Handwritten-Equation-Solver/test_data/'+ str(i) + '.png')
    plot_image(ii)

for i in range(9,11):
    ii = cv2.imread('../Handwritten-Equation-Solver/test_data/' + str(i)+ '.jpeg')
    plot_image(ii)  



# SAVING THE FINAL MODEL

torch.save(model.state_dict(), 'models/CharacterNet2.pth')