 #  __  __                _____           _       _      __        ___  
 # |  \/  |              / ____|         (_)     | |    /_ |      / _ \ 
 # | \  / | __ _ _ __   | (___   ___ _ __ _ _ __ | |_    | |     | | | |
 # | |\/| |/ _` | '_ \   \___ \ / __| '__| | '_ \| __|   | |     | | | |
 # | |  | | (_| | |_) |  ____) | (__| |  | | |_) | |_    | |  _  | |_| |
 # |_|  |_|\__,_| .__/  |_____/ \___|_|  |_| .__/ \__|   |_| (_)  \___/ 
 #              | |                        | |                          
 #              |_|                        |_|                          


# Import the Libaries
import cv2
# Open cv Libary
import numpy as np
# for Array Manipulation
import math
# for basic Math operation , like Ceil and floor function


# Reading the Image from User
path = r'input_images/uk.png'
img = cv2.imread(path)

# open have BGR Convetion so converting it to RGB
x = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

# Converting RGB to HSV
hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)

# Red color Color range for Edge Color Detection
light_red = (2, 100, 100)
dark_red = (16, 255, 255)

# Extracting the Boundary from Image
mask = cv2.inRange(hsv, light_red, dark_red)
result1 = cv2.bitwise_and(x, x, mask=mask)

# For Display:
# cv2.imshow('Boundary Extracted from Image', result1)
# cv2.waitKey(0)

# Building Contour
imgGreyScale = cv2.cvtColor(result1, cv2.COLOR_BGR2GRAY)
imgCanny = cv2.Canny(result1, 100, 200)
imgBlur = cv2.GaussianBlur(imgCanny, (3, 3), 0)
contours, hierarchy = cv2.findContours(imgBlur,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(result1, contours, -1, (255, 255, 255),2)
# cv2.imshow('Contours', result1)
# cv2.waitKey(0)


# Convert img to grayscale
gray = cv2.cvtColor(result1, cv2.COLOR_BGR2GRAY)

# Threshold
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

# Use morphology to close figure
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35,35))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, )

# Find contours and bounding boxes
mask = np.zeros_like(thresh)
cv2.waitKey(0)
contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
for count in contours:
    cv2.drawContours(mask, [count], 0, 255, -1)

# save result
cv2.imwrite("result.jpg", mask)
cv2.waitKey(0)

# Getting Dimension of Image
hh, ww = result1.shape[:2]

# create a single tile as black circle on white background
circle = np.full((11, 11), 255, dtype=np.uint8)
circle = cv2.circle(circle, (7, 7), 3, 50, -1)

# tile out the tile pattern to the size of the input
numht = math.ceil(hh / 11)
numwd = math.ceil(ww / 11)
tiled_circle = np.tile(circle, (numht, numwd))
tiled_circle = tiled_circle[0:hh, 0:ww]

# read Saved image for more rectification
img = cv2.imread('result.jpg')


# Rectifying Image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35,35))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, )
mask = np.zeros_like(thresh)
cv2.waitKey(0)
contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
for count in contours:
    cv2.drawContours(mask, [count], 0, 255, -1)

# composite tiled_circle with mask
result = cv2.bitwise_and(tiled_circle, tiled_circle, mask=mask)

# save result
cv2.imwrite("result.jpg", result)
# cv2.imshow("Final Result", result)
# cv2.waitKey(0)
# Change BG to White
img = cv2.imread('result.jpg')
black_pixels = np.where(
    (img[:, :, 0] == 0) &
    (img[:, :, 1] == 0) &
    (img[:, :, 2] == 0)
)
# set those pixels to white
img[black_pixels] = [255, 255, 255]
cv2.imshow("Final Map Output",img)
cv2.imwrite('uk_output.jpg',img)
cv2.waitKey(0)
