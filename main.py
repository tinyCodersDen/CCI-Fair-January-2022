from flask import *
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import imutils
from imutils.contours import sort_contours
import time
import os
import json
# Loading the Model:
store = open('storage.txt','w')
count = 1
check = False
model_path = 'model.h5'
model = load_model(model_path)
app = Flask(__name__)
ans = ''
color = ''
current_numeral = ''
# roman = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000,'IV':4,'IX':9,'XL':40,'XC':90,'CD':400,'CM':900}
rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
def convert(numeral):
    try:
        int_val = 0
        for i in range(len(numeral)):
            if i > 0 and rom_val[numeral[i]] > rom_val[numeral[i - 1]]:
                int_val += rom_val[numeral[i]] - 2 * rom_val[numeral[i - 1]]
            else:
                int_val += rom_val[numeral[i]]
        return int_val
    except:
        return numeral+' is not a valid roman numeral'
    

@app.route('/', methods=['GET','POST'])
def home():
    global ans
    global count
    global check
    global current_numeral
    global color
    if request.method=='POST':
        if check!=False:
            try:
                time.sleep(1)
                # os.remove("C:/Users/vihan/Downloads/drawing.png")
                # os.rename("C:/Users/vihan/Downloads/drawing (1).png", "C:/Users/vihan/Downloads/drawing.png")
                # image_path = 'C:/Users/vihan/Downloads/drawing ('+str(count)+').png'
                image_path = 'C:/Users/vihan/Downloads/drawing.png'
                image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
                # image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
                # store.write(str(count))
                # count+=1
                trans_mask = image[:,:,3] == 0
                image[trans_mask] = [255, 255, 255, 255]
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cropped = gray
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edged = cv2.Canny(blurred, 30, 250) #low_threshold, high_threshold
                cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                cnts = sort_contours(cnts, method="left-to-right")[0]
                chars = []
                # loop over the contours
                for c in cnts:
                    # compute the bounding box of the contour and isolate ROI
                    (x, y, w, h) = cv2.boundingRect(c)
                    roi = cropped[y:y + h, x:x + w]
                    
                    #binarize image, finds threshold with OTSU method
                    thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                    
                    # resize largest dimension to input size
                    (tH, tW) = thresh.shape
                    if tW > tH:
                        thresh = imutils.resize(thresh, width=28)
                    # otherwise, resize along the height
                    else:
                        thresh = imutils.resize(thresh, height=28)

                    # find how much is needed to pad
                    (tH, tW) = thresh.shape
                    dX = int(max(0, 28 - tW) / 2.0)
                    dY = int(max(0, 28 - tH) / 2.0)
                    # pad the image and force 28 x 28 dimensions
                    padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                        left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                        value=(0, 0, 0))
                    padded = cv2.resize(padded, (28, 28))
                    # reshape and rescale padded image for the model
                    padded = padded.astype("float32") / 255.0
                    padded = np.expand_dims(padded, axis=-1)
                    # append image and bounding box data in char list
                    chars.append((padded, (x, y, w, h)))
                n_cols = 10
                n_rows = np.floor(len(chars)/ n_cols)+1
                boxes = [b[1] for b in chars]
                chars = np.array([c[0] for c in chars], dtype="float32")
                # OCR the characters using our handwriting recognition model
                preds = model.predict(chars)
                # define the list of label names
                labelNames = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                image = cv2.imread(image_path)
                cropped = image[:,:]
                pred_numeral = ''
                for (pred, (x, y, w, h)) in zip(preds, boxes):
                    # find the index of the label with the largest corresponding
                    # probability, then extract the probability and label
                    i = np.argmax(pred)
                    prob = pred[i]
                    label = labelNames[i]
                    # draw the prediction on the image and it's probability
                    pred_numeral+=label
                print(pred_numeral)
                ans = convert(pred_numeral)
                if type(ans)==int:
                    color='blue'
                else:
                    color='red'
                current_numeral = list(pred_numeral).copy()
                current_numeral = ''.join(current_numeral)
                pred_numeral = ''
                print(ans)
                
            except Exception as e:
                # print('No Contours Detected; Try again or with a larger scale(bigger fonts)')
                print(e)
        else:
            check=True
        if os.path.exists("C:/Users/vihan/Downloads/drawing.png"):
            os.remove("C:/Users/vihan/Downloads/drawing.png")
    return render_template('home.html', x = ans, y=color, k = current_numeral)
if __name__=='__main__':
    app.run()