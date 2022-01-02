from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
from imutils.contours import sort_contours
from matplotlib import cm
model_path = 'model.h5'
model = load_model(model_path)
# Image and masking
image_path = 'C:/Users/vihan/Downloads/drawing.png'
image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
trans_mask = image[:,:,3] == 0
image[trans_mask] = [255, 255, 255, 255]
image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
# Convert to grayscale, crop, and use GaussianBlur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cropped = gray[:,:]
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Canny edge the output
edged = cv2.Canny(blurred, 30, 250) #low_threshold, high_threshold
# Find contours of numerals
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
# Initalizing contour validation
n_cols = 10
n_rows = np.floor(len(chars)/ n_cols)+1
fig = plt.figure(figsize=(1.5*n_cols,1.5*n_rows))
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