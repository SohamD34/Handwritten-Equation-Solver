import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from utils import load_images_from_folder

data=[]
dirs = ['0','1','2','3','4','5','6','7','8','9','equals','plus','minus','times','div']

for ascii_num in range(97, 123):    # a-z
    dirs.append(chr(ascii_num))

for ascii_num in range(65, 91):    # A-Z
    dirs.append(chr(ascii_num))

for d in dirs:
    datax = load_images_from_folder('images/' + d)
    for i in range(0,len(datax)):
        datax[i] = np.append(datax[i],[str(dirs.index(d))])

    if(len(data)!=0):
        data = np.concatenate((data,datax))
    else:
        data = datax

df = pd.DataFrame(data,index=None)
df.to_csv('data/character_csv/characters.csv',index=False)

char_images = pd.read_csv('../Handwritten-Equation-Solver/data/character_csv/characters.csv') 

plt.figure(figsize=(20,4))
fig,ax = plt.subplots(2,5)

for j in range(2):
    for i in range(5):
        img = char_images.iloc[7 + (i*j+i+1)*1000,:-1]
        img = np.array(img)
        ax[j][i].imshow(img.reshape(32,32),cmap='gray')

plt.tight_layout()
plt.show()