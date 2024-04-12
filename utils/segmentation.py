import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_segments(img):

    '''
    Takes a input image 'img' and creates segments/borders in the image to identify the start and end margins of characters
    Returns a list of split character images
    '''

    if img is None:
        print("Invalid Image - None")
        return 

    img = ~img

    ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ctrs,ret = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    w = int(28)
    h = int(28)
    train_data = []

    rects=[]
    for c in cnt :
        x,y,w,h = cv2.boundingRect(c)
        rect = [x,y,w,h]
        rects.append(rect)

    bool_rect=[]
    for r in rects:
        l = []
        for rec in rects:
            flag = 0
            if rec!=r:
                if r[0]<(rec[0]+rec[2]+10) and rec[0]<(r[0]+r[2]+10) and r[1]<(rec[1]+rec[3]+10) and rec[1]<(r[1]+r[3]+10):
                    flag = 1
                l.append(flag)
            if rec == r:
                l.append(0)
        bool_rect.append(l)

    dump_rect=[]
    for i in range(0,len(cnt)):
        for j in range(0,len(cnt)):
            if bool_rect[i][j]==1:
                area1 = rects[i][2]*rects[i][3]
                area2 = rects[j][2]*rects[j][3]
                if(area1 == min(area1,area2)):
                    dump_rect.append(rects[i])

    final_rect=[i for i in rects if i not in dump_rect]

    for r in final_rect:
        x = r[0]
        y = r[1]
        w = r[2]
        h = r[3]
        im_crop = thresh[y:y+h+10,x:x+w+10]
        
        im_resize = cv2.resize(im_crop,(28,28))
        im_resize = np.reshape(im_resize,(28,28,1))
        train_data.append(im_resize)

    return train_data