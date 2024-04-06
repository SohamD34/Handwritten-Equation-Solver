import numpy as np
import cv2
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import pandas as pd

unique_labels = [i for i in range(0, 67)]
label_to_char_dict = {}
dirs = ['0','1','2','3','4','5','6','7','8','9','equals','plus','minus','times','div']

for ascii_num in range(97, 123):    # a-z
    dirs.append(chr(ascii_num))
for ascii_num in range(65, 91):    # A-Z
    dirs.append(chr(ascii_num))
for i in range(len(dirs)):
    label_to_char_dict[unique_labels[i]] = dirs[i]

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

def label_to_char(label):
    ''' Function to convert label to character '''
    global label_to_char_dict
    return label_to_char_dict[label]          