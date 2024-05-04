'''
* Class   :  Image Proccessing with Deep Learning
* OpenCV  :  LAB3 : Tension Detection of Rolling Metal Sheet
* Created :  5/4/2024
* Name    :  Eunji Ko
* Number  :  22100034
'''
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.pyplot as plt

#  -------------------------- 0. Functions Definitions  ----------------------------

def draw_quadratic_function(image, a, b, c, x_offset, y_offset, color, thickness):
    height, width = image.shape[:2]
    x = np.linspace(0, width-1, num=width) + x_offset               # Consider the Real size - x_offset
    y = a * x**2 + b * x + c + y_offset                             # Consider the Real size, quadratic function - y_offset
    points = np.column_stack((x.astype(int), y.astype(int)))
    cv.polylines(image, [points], isClosed=False, color=color, thickness=thickness)

    cv.namedWindow('regression on source image', cv.WINDOW_NORMAL)  # Print the regression function of the image
    cv.imshow('regression on source image', image)  
    max_y = np.max(y)                                               # Find the y_max and return it
    return max_y

#  -------------------------- 1. Load the Image -------------------------------------

# Read Image
src = cv.imread('../../Challenging_Dataset/LV1.png')
#src = cv.imread('../../Challenging_Dataset/LV2.png')
#src = cv.imread('../../Challenging_Dataset/LV3.png')
cv.namedWindow('1. source', cv.WINDOW_NORMAL)
cv.imshow('1. source', src) 

#  -------------------------- 2. Image preprocessing --------------------------------

#Filter
image = median = cv.medianBlur(src,5)
cv.namedWindow('2. Median', cv.WINDOW_NORMAL)
cv.imshow('2. Median', image) 

# Convert it to grayscale
gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

# Split to Red
b, g, r = cv.split(image)
H,W = r.shape # original image size
cv.namedWindow('3. Split to Red', cv.WINDOW_NORMAL)
cv.imshow('3. Split to Red', r) 

# ROI
ROI_x = 3
ROI_y = 429
ROI_w = 750
ROI_h = 600

if ROI_w and ROI_h:
    roi = r[ROI_y:ROI_y+ROI_h, ROI_x:ROI_x+ROI_w]
    cv.namedWindow('4. ROI', cv.WINDOW_NORMAL)
    cv.imshow('4. ROI', roi)  
    cv.moveWindow('4. ROI', 0, 0) 
    
# Threshold
ret,thresh =cv.threshold(roi,100,200,cv.THRESH_BINARY)
thres=thresh
cv.namedWindow('5. Threshold', cv.WINDOW_NORMAL)
cv.imshow('5. Threshold', thres)  

# Edge Detection - Canny
dst = cv.Canny(thres, 100, 200)
cv.namedWindow('6. Edge, Canny', cv.WINDOW_NORMAL)
cv.imshow('6. Edge, Canny', dst)  

contour_img = np.zeros_like(dst)                                                                     # make the same size with the ROI image to draw the regression line

# Contours
contour_length_thresh = 2000
contours, hierachy = cv.findContours(dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for contour in contours:
    contour_length = cv.arcLength(contour, True)                                                     # calculate the length of contours

    
    if contour_length > contour_length_thresh:                                                       # draw the contours when they satisfy the threshold value, 2000
        cv.drawContours(contour_img, [contour], -1, (255, 255, 255), thickness=cv.FILLED)
        x, y, w, h = cv.boundingRect(contour)
        cropped_img = contour_img[y:y+h, x:x+w]                                                      # contract the contours section
        indices = np.where(cropped_img == 255)                                                       # when the line is white(=255)
        x_values = indices[1] + x                                                                    # make the point as x coordinate
        y_values = indices[0] + y                                                                    # make the point as x coordinate

#  ------------------------------- 3. Regression  ------------------------------------

        poly = np.polyfit(x_values, y_values, 2)                                                     # regression of qudratic function

        # Draw the graph
        x_range = np.arange(min(x_values), max(x_values), 1)
        y_range = np.polyval(poly, x_range)
        plt.plot(x_range, y_range, color='red')
        draw_quadratic_function(src, poly[0], poly[1], poly[2], ROI_x, ROI_y, (0, 0, 255), 2)        # draw the graph on the source image

        max_y = draw_quadratic_function(src, poly[0], poly[1], poly[2], ROI_x, ROI_y, (0, 0, 255), 2)# y-position [px] of the curvature vertex
        #print(max_y)

        score = H - max_y                                                                            #  y-position [px] of the curvature vertex from the bottom of the image

        # Level
        if max_y > 960:
            Level = 3
        elif max_y >830:
            Level = 2
        else:
            Level = 1


# ------------------------------- 4. Print the Results --------------------------------

# 1) Output on the raw image - draw_quadratic_function

# 2) Level and Score
print("Level is",Level)
print("Score is",score)

# 3) Qudratic funtion on ROI grid
plt.imshow(contour_img, cmap='gray', interpolation='nearest') 
plt.show() 

cv.waitKey(0)
cv.destroyAllWindows()
