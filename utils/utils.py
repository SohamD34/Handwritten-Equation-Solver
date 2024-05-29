import numpy as np
import cv2
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import torch
import sklearn

def load_images_from_folder(folder):

    ''' Function to load images from a folder/directory
        Performs reshaping, conversion to grayscale for uniformity
        Returns a list of the tabulated pixel data ''' 

    train_data=[]
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_GRAYSCALE)
        img = ~img

        if img is not None:
            ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
            ctrs, heirarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            cnt = sorted(ctrs, key = lambda ctr: cv2.boundingRect(ctr)[0])
            w = int(32)
            h = int(32)
            maxi = 0
            for c in cnt:
                x,y,w,h = cv2.boundingRect(c)
                maxi = max(w*h,maxi)
                if maxi == w*h:
                    x_max = x
                    y_max = y
                    w_max = w
                    h_max = h
            im_crop = thresh[y_max:y_max+h_max+10, x_max:x_max+w_max+10]
            im_resize = cv2.resize(im_crop,(32,32))
            im_resize = np.reshape(im_resize,(1024,1))
            train_data.append(im_resize)
            
    return train_data      


#_________________________________________________________________________________________
    

def label_to_char(label):
    ''' Function to convert label to character '''

    global label_to_char_dict

    unique_labels = [i for i in range(0, 67)]
    label_to_char_dict = {}
    dirs = ['0','1','2','3','4','5','6','7','8','9','equals','plus','minus','times','div']

    for ascii_num in range(97, 123):    # a-z
        dirs.append(chr(ascii_num))
    for ascii_num in range(65, 91):    # A-Z
        dirs.append(chr(ascii_num))
    for i in range(len(dirs)):
        label_to_char_dict[unique_labels[i]] = dirs[i]  

    return label_to_char_dict[label]   

#______________________________________________________________________________________________

def fgsm_attack(data, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_data = data + epsilon * sign_data_grad
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data

#______________________________________________________________________________________________

from sklearn.preprocessing import OneHotEncoder

def adversarial_train(model, data, device, train_loader, optimizer, criterion, epsilon, klm, custom=True):

    model.train()
    running_loss = 0.0

    all_labels = []
    all_predictions = []

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)

        encoder = OneHotEncoder(categories=[data.classes])
        labels_encoded = encoder.fit_transform(labels.reshape(-1, 1)).toarray()
        labels_encoded = torch.Tensor(labels_encoded).to(device)

        images.requires_grad = True

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels_encoded)
        loss.backward(retain_graph=True)  # Add retain_graph=True

        # Generate adversarial examples
        data_grad = images.grad.data
        perturbed_images = fgsm_attack(images, epsilon, data_grad)

        # Update model with both original and adversarial examples
        outputs_adv = model(perturbed_images)
        
        loss_adv = criterion(outputs_adv, labels_encoded)
        total_loss = loss + loss_adv
        factor = 1

        if custom == True:
            factor = (101- (4*(klm-1)/100) )

        total_loss = total_loss*factor  # this should be remove

        total = labels.size(0)
        _, predicted = torch.max(outputs_adv.data, 1)
        _, expected = torch.max(labels_encoded, 1)
        correct = (predicted == expected).sum().item()

        all_labels.extend(list(expected.cpu().numpy()))
        all_predictions.extend(list(predicted.cpu().numpy()))

        running_loss += total_loss.item()
        loss_adv.backward()  # Add retain_graph=True
        optimizer.step()  

    train_loss = running_loss/(factor*len(train_loader))
    train_acc = sklearn.metrics.accuracy_score(all_labels, all_predictions)
    print('Loss:',train_loss,end=" ") 
    print('Train accuracy:',train_acc, end=" ")

    return train_loss, train_acc

#______________________________________________________________________________________________

    
def adversarial_validate(model, device, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(list(labels.cpu().numpy()))
            all_predictions.extend(list(predicted.cpu().numpy()))

            running_loss += loss.item()

    val_loss = running_loss / len(val_loader)
    val_acc = sklearn.metrics.accuracy_score(all_labels, all_predictions)
    print('Validation Loss:', val_loss, end=" ")
    print('Validation Accuracy:',val_acc)

    return val_loss, val_acc   

#______________________________________________________________________________________________

def plot_image(img):
    gray_image = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (28,28))
    gray_image = gray_image.reshape((1,28,28))
    data = torch.Tensor(np.array([gray_image])).to(device)
    output = model(data)
    _, predicted = torch.max(output.data, 1)
    plt.figure(figsize=(3,3))
    plt.imshow(gray_image[0])
    plt.show()
    print(mapping[predicted.cpu().numpy()[0]])

#______________________________________________________________________________________________


def evaluate(expression):
    def apply_operator(operators, values):
        operator = operators.pop()
        right = values.pop()
        left = values.pop()
        if operator == '+':
            values.append(left + right)
        elif operator == '-':
            values.append(left - right)
        elif operator == '*':
            values.append(left * right)
        elif operator == '/':
            values.append(left / right)

    def greater_precedence(op1, op2):
        precedences = {'+': 1, '-': 1, '*': 2, '/': 2}
        return precedences[op1] > precedences[op2]

    operators = []
    values = []
    i = 0
    while i < len(expression):
        if expression[i] == ' ':
            i += 1
            continue
        if expression[i] in '0123456789':
            j = i
            while j < len(expression) and expression[j] in '0123456789':
                j += 1
            values.append(int(expression[i:j]))
            i = j
        else:
            if expression[i] == '(':
                operators.append(expression[i])
            elif expression[i] == ')':
                while operators and operators[-1] != '(':
                    apply_operator(operators, values)
                operators.pop()
            else:
                while (operators and operators[-1] != '(' and
                       greater_precedence(operators[-1], expression[i])):
                    apply_operator(operators, values)
                operators.append(expression[i])
            i += 1

    while operators:
        apply_operator(operators, values)

    return values[0]