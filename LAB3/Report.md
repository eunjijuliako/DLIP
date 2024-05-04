# **LAB: Tension Detection of Rolling Metal Sheet**

**Date:**  2024-May-4

**Author:**  Eunji Ko 22100034

**Github:** 

[DLIP/LAB3 at main · eunjijuliako/DLIP](https://github.com/eunjijuliako/DLIP/tree/main/LAB3)

---

# Introduction

![introduction](https://github.com/eunjijuliako/DLIP/assets/164735460/b3863dca-204f-402d-9c13-76e20e370124)

Figure 1. Given Final Result

 In the industry, many cases require a management system for a machine using only video or images. Because of the alteration of managing processes from human to digital can make the production system more efficient. 

In this case, the copper plate will be rolled, and someone in the factory should manage the copper plate to prevent excessive stretching. In this lab, image processing will be applied to replace the role of the person who manages the copper plate.

The edge of a copper plate was approximated to a quadratic function with a mathematical model. The level of the copper plate will be decided by the maximum y point of three cases. Through this process, we can manage the copper plate only with the image.

## 1. Objective

**Goal**: Given a photo of a copper plate, the appropriate pre-processing will be applied to the image. The image will then be approximated as a quadratic function using OpenCV functions to find the inflection point.

## 2. Preparation

### Software Installation

- OpenCV 4.90, Visual Studio Code

### Dataset

[Challenging dataset](https://github.com/ykkimhgu/DLIP-src/blob/1d4183c65fb41d2ed5bf093c526c7ef49afa2d8e/LAB_Tension_Detection_dataset/Challenging_Dataset.zip)

## **3. Algorithm Conditions**

- Use Python OpenCV (*.py)
- Measure the metal sheet tension level from Level 1 to Level 3.
    - Use the minimum y-axis position of the metal sheet curvature
    - Level 1: >250px from the bottom of the image
    - Level 2: 120~250 px from the bottom of the image
    - Level 3: < 120 px from the bottom of the image
- Display the output on the raw image
    - Tension level: Level 1~3
    - Score: y-position [px] of the curvature vertex from the bottom of the image
    - Curvature edge
- ROI has already decided on the detection of the target area.

# Algorithm

## 1. Overview

**Flow chart**

![Overview](https://github.com/eunjijuliako/DLIP/assets/164735460/0b285a1f-5770-4c10-adad-446513f5be34)

Figure 2. Overview Flow Chart

First, when we load the image, with the gray-scaled image, the color will be split to red to detect the edge more clearly. With the known ROI(ROI; Region of Interest) section, we will find the edge of the copper plate. After, contouring and thresholding the image, the quadratic function will be calculated through the regression function. Lastly, we will get the maximum point of y, and get the level and score of the image.

## 2. Procedure

### Load the Source Image

![source](https://github.com/eunjijuliako/DLIP/assets/164735460/1202cc02-1b63-494a-a452-86de8741de5b)

Figure 3. Source Image

### Image Pre-processing

- Median Filter

![median](https://github.com/eunjijuliako/DLIP/assets/164735460/3d2ec970-93ac-4f66-b35d-f357dd926ac6)

Figure 4. Median Filter

I removed the salt and pepper noise with the median filter.

- Gray-Scale

![gray](https://github.com/eunjijuliako/DLIP/assets/164735460/939a6d9a-55da-4eca-8730-bd7a81fddb77)

Figure 5. Gray Scale

To split the image, the source image should be gray-scaled.

- Split

![blue](https://github.com/eunjijuliako/DLIP/assets/164735460/2acc5a62-780b-4f0f-9f50-e2d472d0faf0)

Figure 6. Split the Image to Blue

![green](https://github.com/eunjijuliako/DLIP/assets/164735460/0e24ba27-f177-43ec-bff5-47878b44bb5b)

Figure 7. Split the Image to Green

![red](https://github.com/eunjijuliako/DLIP/assets/164735460/8aa45107-3eca-42c3-9ce9-215568a41813)

Figure 8. Split the Image to Red

Because of the most color of the source image is red, it is easier to see the copper plate clearly with the image which is split to red.

- ROI

![ROI](https://github.com/eunjijuliako/DLIP/assets/164735460/35ffdd2b-3af4-4fa7-b1d8-7979123d159e)

Figure 9. ROI

With the the image which is split to red, I cropped the image with ROI. The value of ROI was chosen with the maximum x,y,w,h points of the three source images, levels 1,2, and 3.

```python
# ROI
ROI_x = 3
ROI_y = 429
ROI_w = 750
ROI_h = 600
```

- Threshold

![threshold](https://github.com/eunjijuliako/DLIP/assets/164735460/6ac15ea0-9244-4cdf-923f-5998a0141096)

Figure 10. Threshold

To detect the whole edge of the copper plate, I applied the thresholding method, and it separates the target object from the background better.

```python
# Threshold
ret,thresh =cv.threshold(roi,100,200,cv.THRESH_BINARY)
thres=thresh
cv.namedWindow('5. Threshold', cv.WINDOW_NORMAL)
cv.imshow('5. Threshold', thres)  
```

### Regression

- Edge Detection

![canny](https://github.com/eunjijuliako/DLIP/assets/164735460/5b792b5b-bdad-4b79-ad85-4f64a3bdf20e)

Figure 11. Edge Detection, Canny

For detect the edge line of the copper plate, i applied the canny method, and the thresholding values are same with below.

```python
# Edge Detection - Canny
dst = cv.Canny(thres, 100, 200)
cv.namedWindow('6. Edge, Canny', cv.WINDOW_NORMAL)
cv.imshow('6. Edge, Canny', dst)  
```

- Regression to Quadratic Function

```python
# Contours
contour_length_thresh = 2000
contours, hierachy = cv.findContours(dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for contour in contours:
    contour_length = cv.arcLength(contour, True)                                                     # calculate the length of contours
    
    if contour_length > contour_length_thresh:                                                       # draw the contours when they satisfy the threshold value, 2000
        cv.drawContours(contour_img, [contour], -1, (255, 255, 255), thickness=cv.FILLED)      
```

From the result of the canny, I applied the contours on the edge. After calculating the size of the length of the contours, I selected the thresholding value to take only a long line. 

```python
x, y, w, h = cv.boundingRect(contour)
        cropped_img = contour_img[y:y+h, x:x+w]                                                      # contract the contours section
        indices = np.where(cropped_img == 255)                                                       # when the line is white(=255)
        x_values = indices[1] + x                                                                    # make the point as x coordinate
        y_values = indices[0] + y    
```

Then, the (x,y) coordinate of the line will be the (x,y) dataset of the regression for making the quadratic function. 

![Regression on ROI](https://github.com/eunjijuliako/DLIP/assets/164735460/46420feb-b7b2-404f-87e1-eb717619b679)

Figure 12. Regression on the ROI Grid

When we succeed in making the quadratic function with ‘polyfit’ and ‘draw_quadratic_function’, I draw the function on the source image. To conduct this, calculate the offset of the x and y, which are the ROI x and ROI y will be considered.

### Detail of the Functions

**polyfit function** 

```python
poly = np.polyfit(x_values, y_values, 2)                                                     # regression of qudratic function

        # Draw the graph
        x_range = np.arange(min(x_values), max(x_values), 1)
        y_range = np.polyval(poly, x_range)
        plt.plot(x_range, y_range, color='red')
        draw_quadratic_function(src, poly[0], poly[1], poly[2], ROI_x, ROI_y, (0, 0, 255), 2)        # draw the graph on the source image

        max_y = draw_quadratic_function(src, poly[0], poly[1], poly[2], ROI_x, ROI_y, (0, 0, 255), 2)# y-position [px] of the curvature vertex
        #print(max_y)
```

Parameters of polyfit function 

Input:

- x, y: data input
- n: degree of polynomial to be approximated

Output:

- p: It is a one-dimensional array (vector) with the values of the coefficients of the polynomial in descending order. In this case, it is a,b, and c.

**draw_quadratic_function function** 

```python
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
```

Parameters of function

- image: the image that i want to draw the graph
- a, b, c: coefficients of qudratic function
- x_offset, y_offset: offset because of the ROI
- color, thickness : features of the graph line

```python
    points = np.column_stack((x.astype(int), y.astype(int)))
```

It changes the x and y values to integer and connets the 1-dimension vector recongnized as a column vector.

```python
    cv.polylines(image, [points], isClosed=False, color=color, thickness=thickness)

```

It draws the line on the image. ‘isClosed=False’ means that this function draws the line not the polygon**.**

# Result and Discussion

## 1. Final Result

### Level 1

![result_lev1](https://github.com/eunjijuliako/DLIP/assets/164735460/a8ff2541-ad9e-4752-96fc-30cce5782cc7)

Figure 13. Result of Level 1


![result_lev1_values](https://github.com/eunjijuliako/DLIP/assets/164735460/cbdb5d55-d2f8-44d1-b078-2a0c99b25413)

Figure 14. Result of Level 1 and it’s score

### Level 2

![result_lev2](https://github.com/eunjijuliako/DLIP/assets/164735460/e9884f35-2a0f-4279-b26a-7baad4d8258c)

Figure 15. Result of Level 2

![result_lev2_values](https://github.com/eunjijuliako/DLIP/assets/164735460/46523510-9ae7-4fb2-afe8-4ccc09187e3a)

Figure 16. Result of Level 2 and it’s score

### Level 3

![result_lev3](https://github.com/eunjijuliako/DLIP/assets/164735460/93ccdfe6-452a-4948-8820-305cc4da2a28)

Figure 17. Result of Level 3

![result_lev3_values](https://github.com/eunjijuliako/DLIP/assets/164735460/2e51958b-7960-451d-873b-17730910137c)

Figure 18. Result of Level 3 and it’s score

| Image | Result | Score [px] | Accuracy [%] |
| --- | --- | --- | --- |
| Level 1 | Level 1 | 301  | 100% |
| Level 2 | Level 2 | 135 | 100% |
| Level 3 | Level 3 | 78 | 100% |

## 2. Discussion

 As a result, each level of images are detected as the real level with the algorithm. 

The level is determined as the length of the y of the approximated quadratic function. First image, which is level 1, the length of y is 301 px. If the length of the y is more than 250 px from the bottom, then it is level 1. If the size of y is between 120 px and 250 px, then the level is 2, so the second image is level 2 because the score of it is 135. Lastly, the value of y of the third image is 78, and it is level 3 because if y is smaller than 120 px, then it is level 3.

# Conclusion

 Through the lab, with the given three photos of copper plates, we were able to determine the y-position of the curvature vertex. The algorithm involved two important processes. Firstly, pre-processing was necessary to handle the noise and other objects in the source image. This involved carefully selecting an appropriate thresholding value and finding the ROI. All of the thresholding values were applied to all cases. Secondly, finding the approximate quadratic function was important for generating the dataset. I used 'polyfit' and created 'draw_quadratic_function' to find the coefficient of the quadratic function and the curve.

In conclusion, the algorithm was successful in finding the y maximum value, level, and score of the copper plates. However, it can only find second-order equations. To allow the algorithm to find all types of polynomials, it would need to be adapted to handle N-order equations in the future.

---

# Appendix

### Code

```python
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
```
