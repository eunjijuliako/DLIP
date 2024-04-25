# LAB: Dimension Measurement with 2D camera

**Date:**  2024-April-25

**Author:**  Eunji Ko 22100034

**Github:** 

[DLIP/LAB2 at main · eunjijuliako/DLIP](https://github.com/eunjijuliako/DLIP/tree/main/LAB2)

**Demo Video:** [https://youtu.be/BQeB-paibqc?feature=shared](https://youtu.be/BQeB-paibqc?feature=shared)

---

# Introduction

 Recently, there have been a lot of applications of deep learning and image processing. Representatively, knowing the size of objects or distances between cars in an autonomous car is highly important for safe and accurate driving. To do this in image processing, calibration, which provides a pixel-to-real distance conversion factor is required. Hence, there are some assumptions when we apply the calibration and know the real size in the image process. Therefore, we want to compare two methods for getting the exact size and will find what assumptions are necessary when we detect the exact size of 3D objects through this lab.  

This lab is conducted to find how to measure the whole dimension of a rectangular object with images of a smartphone by using an image processing method. Team members are Hyeonho Moon and Eunji Ko, and we want to suggest two methods to find the dimensions of an object with a phone. 

## 1. Objective

**Goal**: Find how to measure the whole dimension of a rectangular object with images of a smartphone by using an image processing method. 

## 2. Preparation

### Software Installation

- OpenCV 4.90, Visual Studio 2022

### Hardware

- Galaxy Ultra 22 camera

## **3. Algorithm Conditions**

To find the sizes of a 3D rectangular object, below are the conditions for the algorithm.

**Conditions for all Methods**

- Assume the user already knows the exact width of the target object.
- Because we get the calibration factor with Hyeonho’s phone, take a photo with Hyeonho’s phone.

**Conditions for Method 1** 

- Users have to detect the four corners in the image processing.
- Users have to press the ‘s’ or ‘S’ when saving the ratio(pixel to [mm]) for calculating real size of the object.

**Condition for Method 2** 

- Users have to take a frontal photo.

# Algorithm

## 1. Overview

**Flow charts for Method 1 and 2**

[Figure1. Overview Flow chart for method 1](https://github.com/eunjijuliako/DLIP/assets/164735460/d931e23f-42e5-4252-9529-36ab75f81e8c)

Figure1. Overview Flow chart for method 1

[Figure2. Overview Flow chart for method 2](https://github.com/eunjijuliako/DLIP/assets/164735460/81d6c252-fde9-428c-b9ac-eea18358caef)

Figure2. Overview Flow chart for method 2

## 2. Procedure

### **Method 1: Size Detection with Manual Corner Matching**

[Figure 3. A user enters the Real Reference Width, 50 mm](https://github.com/eunjijuliako/DLIP/assets/164735460/9f0559c2-d696-4eed-9f06-db91b35840c1)

Figure 3. A user enters the Real Reference Width, 50 mm

After reading the camera parameters from the calibrated XML file, a user should input the real reference width of a reference rectangular. 

[Figure 4. WL_Reference size capture](https://github.com/eunjijuliako/DLIP/assets/164735460/50c188a4-9497-4059-a1ba-8498bccd6caa)

Figure 4. WL_Reference size capture

[Figure 5. WL_Calculated Reference size](https://github.com/eunjijuliako/DLIP/assets/164735460/e0ffa4c7-5a2c-4303-8d80-6a77d8b9ed0a)

Figure 5. WL_Calculated Reference size

This picture was taken along the Z-axis to get the length of the object. If the user presses ‘s’ or ‘S’, the algorithm recognizes the current pixel width as a reference pixel width. With the reference real and pixel width, the ratio represents the relationship between pixel and [mm] conversion will be calculated. 

[Figure 6. WL_Retangular Object](https://github.com/eunjijuliako/DLIP/assets/164735460/67775e94-8a95-47b9-91e5-a317fe5fb767)

Figure 6. WL_Retangular Object

[Figure 7. WL_Calcuated Length](https://github.com/eunjijuliako/DLIP/assets/164735460/8d7be876-bffd-4e7c-bc74-d50a5de248c6)

Figure 7. WL_Calcuated Length

With the ratio, the target object’s calculated real width and length will be calculated. In figure 6, The target object’s real width is 100 mm and length is 99 [mm]. 

[Figure 8. WH_Reference Pixel Width](https://github.com/eunjijuliako/DLIP/assets/164735460/c59fd462-5d0a-4204-b60b-be12cb1eb59b)

Figure 8. WH_Reference Pixel Width

[Figure 9. WH_Ratio and Calculated Length](https://github.com/eunjijuliako/DLIP/assets/164735460/5bfdfcb0-eb3f-4e87-a261-5b58bf226d3e)

Figure 9. WH_Ratio and Calculated Length

This picture was taken with the sides(ZX, ZY) to get the height of the rectangular. The pixel width of the reference rectangular is calculated in the same way. After calculating the ratio with the real reference width of 50 mm, the real height of the reference rectangular is calculated which is 49 mm.

[Figure 10. WH_Retangular Object](https://github.com/eunjijuliako/DLIP/assets/164735460/356bf5ff-af04-4a37-938b-c7131327ce73)

Figure 10. WH_Retangular Object

[Figure 11. WH_Calcuated Height](https://github.com/eunjijuliako/DLIP/assets/164735460/07e16340-d619-4fb4-a2f0-73b7de00e4c6)

Figure 11. WH_Calcuated Height

With the ratio, the real width of the target rectangular is calculated which is 101 and it’s height is 51 mm.

[Figure 12. Distorted_Reference Pixel Width](https://github.com/eunjijuliako/DLIP/assets/164735460/3cff348e-e23a-42e0-8664-e7ea72c2a710)

Figure 12. Distorted_Reference Pixel Width

[Figure 13. Distorted_Ratio and Calculated Length](https://github.com/eunjijuliako/DLIP/assets/164735460/5555baa1-cdd3-4e2e-bfd6-d05810a70cb3)

Figure 13. Distorted_Ratio and Calculated Length

We tested this algorithm with the distorted image. The width of the reference rectangular is 50 mm, and the calculated length is 50 mm. However, the recognized length is 44 mm. This result doesn’t satisfy the accuracy requirement because the error is more than 3mm.

[Figure 14. Distorted_Retangular Object](https://github.com/eunjijuliako/DLIP/assets/164735460/cb711099-7d35-40aa-a8dc-f9e54c393d01)

Figure 14. Distorted_Retangular Object

[Figure 15. Distorted_Calcuated Height](https://github.com/eunjijuliako/DLIP/assets/164735460/c42a0295-e28a-4a0f-9786-62c52d7be540)

Figure 15. Distorted_Calcuated Height

Also, the error of the target object’s size is more than 3mm which is 86 mm in width and 72 mm in length when the true value is 100mm each. 

### **Method 2: Size Detection with Frontal Photo**

[Figure 16. (left)WL, (right)WH](https://github.com/12-dimension-cat/neko/assets/144550430/a3f3faf2-a635-4b05-a23c-0be519b5f320)

Figure 16. (left)WL, (right)WH

The picture was the same as the Figure 3,5,7,9, and it was taken on the same line with the object.

[Figure 17. Morphology applied (left)WL, (right)WH](https://github.com/12-dimension-cat/neko/assets/144550430/cfa68d6e-7c9b-4f84-b5cf-82f461976ca1)

Figure 17. Morphology applied (left)WL, (right)WH

We applied morphology to obtain the lines of the photos for calculating real width, length, and height.

[Figure 18. Enter the reference width](https://github.com/12-dimension-cat/neko/assets/144550430/7e2a6e40-6c1c-4c1f-968a-b01773178d6c)

Figure 18. Enter the reference width

The width of the reference object is inputted from the user.

The following sequence of code allows us to calculate the ratio between real millimeters and pixels, and to determine the actual length of the target object:

```cpp
// Find contours
findContours(src_gray, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

// Bounding box
 for (const auto& contour : contours) {                                   // Extracting bounding boxes for all contours
     if (contourArea(contour) > minArea) {
         boundingBoxes.push_back(boundingRect(contour));
     }
 }
 
 // Calculate the size
 double firstObjectRealWidth = boundingBoxes[0].width * Ratio;             // Calculate the dimensions of the first object
```

[Figure 19. Calculated (left)WL, (right)WH](https://github.com/12-dimension-cat/neko/assets/144550430/f2dc9f98-507a-4efd-9a9b-77de6ffe025c)

Figure 19. Calculated (left)WL, (right)WH

The ratio is calculated based on the width of the input reference object to output W, L, and H of each object.

# Result and Discussion

## 1. Final Result

**Demo Video**

[https://youtu.be/BQeB-paibqc?feature=shared](https://youtu.be/BQeB-paibqc?feature=shared)

**Accuracy Result** 

$$
Percentage  Acuuracy = 100 - ( | True - Estimated | / True ) * 100
$$

**Method 1**

| Items | True [mm] | Estimated [mm] | Acuuracy [%] |
| --- | --- | --- | --- |
| reference objects W | 50 | 50 | 100 |
| reference objects L | 5 | 49 | 98 |
| reference objects H | 50 | 49 | 98 |
| rectangular object W | 100 | 101 | 99 |
| rectangular object L | 100 | 99 | 99 |
| rectangular object H | 50 | 51 | 98 |
| distorted object W | 100 | 86 | 86 |
| distorted object L | 100 | 72 | 72 |

**Method 2**

| Items | True [mm] | Estimated [mm] | Acuuracy[%] |
| --- | --- | --- | --- |
| reference objects W | 50 | 52.99 | 94.02 |
| reference objects L | 50 | 49.87 | 99.74 |
| reference objects H | 50 | 50.52 | 98.96 |
| rectangular object W | 100 | 102.99 | 97.01 |
| rectangular object L | 100 | 100.91 | 99.09 |
| rectangular object H | 50 | 51.3 | 97.4 |

## 2. Discussion

The average accuracy of Method 1 is 93.75%. With the frontal photos, the accuracy is high. However, the accuracy is very low when applied the distorted photos. We estimated that the openCV function ‘warpPerspective’ for unfolding the distorted image was not worked well. 

The average accuracy of Method 2 is 97.7%. The accuracy of Method 2 is 3.95% higher than that of Method 1. Also it satisfies the requirement which is the error should be less than 3mm. With frontal photos, the accuracy of method 1 and 2 are almost the same. 

Therefore, when detecting the size of a 3D object, taking photos from the front is highly important. To obtain the real size even with a distorted image, we must find or create a method to unfold the distorted image into an exact flat image.

# Conclusion

As a result, we were able to measure the 3D dimensions of a rectangular object using a smartphone image and image processing with two methods. The second method, 'Size Detection with Frontal Photo,' provided more accurate detection.

For the first method, we attempted to measure the dimensions at any camera angle. The assumptions for this algorithm were that 'Users have to detect the four corners in the image processing,' and 'Users have to press 's' or 'S' when saving the ratio (pixel to [mm]).' However, the measurement error was more or less than 3 mm, which did not satisfy the accuracy requirement.

Therefore, we tried the second method. The assumption here was that 'Users have to take frontal photos.' This algorithm met the accuracy requirement.

To improve the algorithm of method 1, the function of OpenCV 'warpPerspective' should be more accurate and clear. We could try other functions such as 'getPerspectiveTransform' or 'warpAffine.' We could also explore other functions in OpenCV that convert the warped image to a real image. For method 2, detecting the corner of the target plane should be conducted at various angles for wider application.

In conclusion, assumptions like knowing the exact size of the reference object and taking a clear frontal photo are necessary when we want to determine the real size of an object using only phone photos.

---

# Appendix

### Mehtod 1 : Size detection with Manual Corner Matching

```cpp
/*------------------------------------------------------/
* Class   :  Image Proccessing with Deep Learning
* OpenCV  :  LAB2 : Dimension Measurement with 2D camera
* Created :  4/23/2024
* Name    :  Eunji Ko, Hyeonho Moon
* Number  :  22100034
* 
* Method 1:  Size detection with Manual Corner Matching
------------------------------------------------------*/

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "cameraParam.h"
#include <opencv.hpp>

using namespace std;
using namespace cv;

static void onMouse(int event, int x, int y, int, void*);

String windowTitle = "Source Image";
String labels[4] = { "TL","TR","BR","BL" };
vector< Point2f> roi_corners;
vector< Point2f> midpoints(4);
vector< Point2f> dst_corners(4);

bool dragging;
int selected_corner_index = 0;
bool validation_needed = true;

int main(int argc, char** argv)
{
    /* 0. Initialization */
    Mat original_image;
    Mat src;
    Mat image;    
    Mat warped_image;
    Mat M;

    double referenceWidth;
    float original_image_cols;
    float original_image_rows;

    bool endProgram = false;

    int width = 0;    
    int height = 0;
    int pixwidth = 0;
    int pixheight = 0;
    double ratio = 0;
    int Refpixwidth = 0;

    int capture_flag = false;

    /* 1. Read the Callibration Information */
    cameraParam param("calib_resource_moon.xml");   // Caliibration - Intrinsic Matrix of the phone

    /* 2. Read the Source Image */
    src = imread("Test.jpg");
    //src = imread("WH.jpg");
    original_image = param.undistort(src);

    /* 3. Reference Value from User */
    cout << "Enter the real reference Width: ";
    cin >> referenceWidth;

    /* 4. Corner detection of User */
    original_image_cols = (float)original_image.cols;
    original_image_rows = (float)original_image.rows;

    roi_corners.push_back(Point2f((float)(original_image_cols / 1.70), (float)(original_image_rows / 4.20)));
    roi_corners.push_back(Point2f((float)(original_image.cols / 1.15), (float)(original_image.rows / 3.32)));
    roi_corners.push_back(Point2f((float)(original_image.cols / 1.33), (float)(original_image.rows / 1.10)));
    roi_corners.push_back(Point2f((float)(original_image.cols / 1.93), (float)(original_image.rows / 1.36)));

    namedWindow(windowTitle, WINDOW_NORMAL);
    namedWindow("Warped Image", WINDOW_AUTOSIZE);
    moveWindow("Warped Image", 20, 20);
    moveWindow(windowTitle, 330, 20);

    setMouseCallback(windowTitle, onMouse, 0);
    

    while (!endProgram)
    {
        
        if (validation_needed && (roi_corners.size() < 4))
        {
            validation_needed = false;
            image = original_image.clone();

            for (size_t i = 0; i < roi_corners.size(); ++i)
            {
                circle(image, roi_corners[i], 5, Scalar(0, 255, 0), 3);

                if (i > 0)
                {
                    line(image, roi_corners[i - 1], roi_corners[(i)], Scalar(0, 0, 255), 2);
                    circle(image, roi_corners[i], 5, Scalar(0, 255, 0), 3);
                    putText(image, labels[i].c_str(), roi_corners[i], FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
                }
            }
            imshow(windowTitle, image);
        }

        if (validation_needed && (roi_corners.size() == 4))
        {
            image = original_image.clone();
            for (int i = 0; i < 4; ++i)
            {
                line(image, roi_corners[i], roi_corners[(i + 1) % 4], Scalar(0, 0, 255), 2);
                circle(image, roi_corners[i], 5, Scalar(0, 255, 0), 3);
                putText(image, labels[i].c_str(), roi_corners[i], FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
            }

            imshow(windowTitle, image);

            midpoints[0] = (roi_corners[0] + roi_corners[1]) / 2;
            midpoints[1] = (roi_corners[1] + roi_corners[2]) / 2;
            midpoints[2] = (roi_corners[2] + roi_corners[3]) / 2;
            midpoints[3] = (roi_corners[3] + roi_corners[0]) / 2;

            dst_corners[0].x = 0;
            dst_corners[0].y = 0;
            dst_corners[1].x = (float)norm(midpoints[1] - midpoints[3]);
            dst_corners[1].y = 0;
            dst_corners[2].x = dst_corners[1].x;
            dst_corners[2].y = (float)norm(midpoints[0] - midpoints[2]);
            dst_corners[3].x = 0;
            dst_corners[3].y = dst_corners[2].y;

            Size warped_image_size = Size(cvRound(dst_corners[2].x), cvRound(dst_corners[2].y));

            M = getPerspectiveTransform(roi_corners, dst_corners);

            Size imageSize;

            warpPerspective(original_image, warped_image, M, warped_image_size);     // Do Perspective Transformation
            imshow("Warped Image", warped_image);
            imageSize = warped_image.size();                                         // Get the pixel size of warped_image
            pixwidth = imageSize.width;                                              // Get the pixel width size of warped_image
            pixheight = imageSize.height;
            
            /* 6. Calculate the height or length [mm] */

            if (capture_flag == true) {                                              // If all of the reference size [mm] and [pixel] are entered, calculate the real size
                                        
                height = pixheight * ratio;                                          // [mm] height
                width = pixwidth * ratio;                                            // [mm] width

                /* 7. Print our the Ratio and calculated Height or Length */
                std::cout << "Ratio: " << ratio << std::endl;
                std::cout << "Width [mm]: " << width << ", Height or Length [mm]: " << height << std::endl;
            }
            

        }

        char c = (char)waitKey(300);

        if ((c == 'q') || (c == 'Q') || (c == 27))                                   // End 
        {
            endProgram = true;
        }
        /* 5. Find the Ratio - [pixel] to [mm] */

        if ((c == 's') || (c == 'S'))                                             
        {   
            
            capture_flag = true;                                                    // Change the flag to true
            Refpixwidth = pixwidth;                                                 // When user puch the 's' or 'S' that is the reference pixel width
            ratio = referenceWidth / (double)Refpixwidth;                           // Calculate the ratio -[pixel] to [mm]
            std::cout << "Ratio is Calculated!" << std::endl;
        }
        
    }

    return 0;
}

static void onMouse(int event, int x, int y, int, void*)
{
   
    if (roi_corners.size() == 4)                                                     // Action when left button is pressed
    {
        for (int i = 0; i < 4; ++i)
        {
            if ((event == EVENT_LBUTTONDOWN) && ((abs(roi_corners[i].x - x) < 10)) && (abs(roi_corners[i].y - y) < 10))
            {
                selected_corner_index = i;
                dragging = true;
            }
        }
    }
    else if (event == EVENT_LBUTTONDOWN)
    {
        roi_corners.push_back(Point2f((float)x, (float)y));
        validation_needed = true;
    }

    
    if (event == EVENT_LBUTTONUP)                                               // Action when left button is released
    {
        dragging = false;
    }

    
    if ((event == EVENT_MOUSEMOVE) && dragging)                                 // Action when left button is pressed and mouse has moved over the window
    {
        roi_corners[selected_corner_index].x = (float)x;
        roi_corners[selected_corner_index].y = (float)y;
        validation_needed = true;
    }
}
```

### Method 2 : Size Detection with Frontal Photo

```cpp
/*------------------------------------------------------/
* Class   :  Image Proccessing with Deep Learning
* OpenCV  :  LAB2 : Dimension Measurement with 2D camera
* Created :  4/23/2024
* Name    :  Eunji Ko, Hyeonho Moon
* Number  :  22100034
* 
* Method 2:  Size Detection with Frontal Photo 
------------------------------------------------------*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include "cameraParam.h"
#include <sstream>
#include <iomanip>

using namespace cv;
using namespace std;

/*  0. Initialization  */

// Global variables for Threshold
int threshold_value = 220;
int threshold_type = 1;
int morphology_type = 0;

// Global variables for Morphology
int element_shape = MORPH_RECT;		
int n = 3;
Mat element = getStructuringElement(element_shape, Size(n, n));

int const max_value = 255;
int const max_type = 8;
int const max_BINARY_value = 255;

double Ratio;

Mat src, dst, src1, src2, dst1, dst2, dst_norm_scaled1, dst_norm_scaled2, src_gray1, src_gray2;

void dataset(void);
void filtering(void);
void findCorners_z(const Mat& src_gray, Mat& corners, const Mat& original);
void findCorners_xy(const Mat& src_gray, Mat& corners, const Mat& original);
void showresult(void);

int main() {
    
    /* Read the Calibration Infromation and Source Image */
    dataset();

    /* Thresholding for Clear Corner Detection */
    filtering();

    /* Finding Corners through Contouring */
    findCorners_z(src_gray1, dst_norm_scaled1, dst1);
    findCorners_xy(src_gray2, dst_norm_scaled2, dst2);

    /* Results */
    showresult();

    waitKey(0);
    return 0;
}

void dataset(void) {

    /* 1. Read the Callibration Information */
    cameraParam param("calib_resource_moon.xml");                                       // Caliibration - Intrinsic Matrix of the phone

    /* 2. Read the Source Image and Conver to Gray-scale Image */
    src1 = imread("WL.jpg");
    src2 = imread("WH.jpg");

    dst1 = param.undistort(src1);
    dst2 = param.undistort(src2);

    cvtColor(dst1, src_gray1, COLOR_BGR2GRAY);
    cvtColor(dst2, src_gray2, COLOR_BGR2GRAY);
}

void filtering(void) {

    threshold(src_gray1, src_gray1, 220, max_BINARY_value, THRESH_BINARY); // Thresholding 1
    erode(src_gray1, src_gray1, element, Point(-1, -1), 3);
    dilate(src_gray1, src_gray1, element, Point(-1, -1), 4);

    threshold(src_gray2, src_gray2, 180, max_BINARY_value, THRESH_BINARY); // Thresholding 2
    erode(src_gray2, src_gray2, element, Point(-1, -1), 120);
    dilate(src_gray2, src_gray2, element, Point(-1, -1), 140);
}

void findCorners_z(const Mat& src_gray, Mat& corners, const Mat& original) {
    vector<vector<Point>> contours;
    findContours(src_gray, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    corners = original.clone();

    const double minArea = 100.0;                                          // Minimum contour area
    vector<Rect> boundingBoxes;                                            // Vector to store bounding boxes of contours

            
    for (const auto& contour : contours) {                                 // Extract bounding boxes for all contours that meet the minimum area requirement
        if (contourArea(contour) > minArea) {
            boundingBoxes.push_back(boundingRect(contour));
        }
    }

    if (boundingBoxes.size() < 2) {
        cerr << "Two objects are required for comparison!" << endl;
        return;
    }

    /* 3.1 Length : Reference Value from User */
    double referenceWidth;
    cout << "Enter the reference Width for the first object: ";             // Get a real reference Width from the user
    cin >> referenceWidth;

    /* 4.1 Length : Calculate the size - [pixel] to [mm] */
    double firstObjectWidth = boundingBoxes[0].width;                       // Calculate the dimensions of the first object
    double firstObjectLength = boundingBoxes[0].height;

    Ratio = referenceWidth / firstObjectWidth;

    firstObjectLength = boundingBoxes[0].height * Ratio;

   
    double secondObjectRealWidth = boundingBoxes[1].width * Ratio;          // Calculate the real dimensions of the second object based on the first object's reference width
    double secondObjectRealLength = boundingBoxes[1].height * Ratio;

    /* 5.1 Display  */
    Scalar color1 = Scalar(0, 255, 0);                                      // Display the first object with its reference dimensions

    stringstream ss1;

    ss1 << fixed << setprecision(2) << "W1: " << referenceWidth << ", L1: " << firstObjectLength; 
    
    putText(corners, ss1.str(), Point(boundingBoxes[0].x + 5, boundingBoxes[0].y + 20), FONT_HERSHEY_SIMPLEX, 2, color1, 3);

    Scalar color2 = Scalar(0, 0, 255);                                      // Display the second object with its calculated dimensions

    stringstream ss2;

    ss2 << fixed << setprecision(2) << "W2: " << secondObjectRealWidth << ", L2: " << secondObjectRealLength;

    putText(corners, ss2.str(), Point(boundingBoxes[1].x + 5, boundingBoxes[1].y + 20), FONT_HERSHEY_SIMPLEX, 2, color2, 3);

}

void findCorners_xy(const Mat& src_gray, Mat& corners, const Mat& original) {

    vector<vector<Point>> contours;
    findContours(src_gray, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    corners = original.clone();
    const double minArea = 100.0;                                            // Minimum contour area

    vector<Rect> boundingBoxes;                                              // For storing contour bounding boxes

    for (const auto& contour : contours) {                                   // Extracting bounding boxes for all contours
        if (contourArea(contour) > minArea) {
            boundingBoxes.push_back(boundingRect(contour));
        }
    }

    if (boundingBoxes.size() < 2) {
        cerr << "Two objects are required for comparison!" << endl;
        return;
    }

    /* 4.2 Height : Calculate the size - [pixel] to [mm] */
    double firstObjectRealWidth = boundingBoxes[0].width * Ratio;             // Calculate the dimensions of the first object

    double firstObjectRealHeight = boundingBoxes[0].height * Ratio;

    double secondObjectRealWidth = boundingBoxes[1].width * Ratio;            // Calculate the real dimensions of the second object based on the first object's reference width

    double secondObjectRealHeight = boundingBoxes[1].height * Ratio;

    /* 5.2 Height : Display  */
    Scalar color1 = Scalar(0, 255, 0);                                        // Display the first object with its reference dimensions

    stringstream ss1;

    ss1 << fixed << setprecision(2) << "W2: " << firstObjectRealWidth << ", H2: " << firstObjectRealHeight;

    putText(corners, ss1.str(), Point(boundingBoxes[0].x + 5, boundingBoxes[0].y + 20), FONT_HERSHEY_SIMPLEX, 2, color1, 3);

    Scalar color2 = Scalar(0, 0, 255);                                        // Display the second object with its calculated dimensions

    stringstream ss2;

    ss2 << fixed << setprecision(2) << "W1: " << secondObjectRealWidth << ", H1: " << secondObjectRealHeight;

    putText(corners, ss2.str(), Point(boundingBoxes[1].x + 5, boundingBoxes[1].y + 20), FONT_HERSHEY_SIMPLEX, 2, color2, 3);
}

void showresult(void) {

    namedWindow("filter", WINDOW_NORMAL);
    imshow("filter", src_gray1);

    namedWindow("filter1", WINDOW_NORMAL);
    imshow("filter1", src_gray2);

    namedWindow("corners_window1", WINDOW_NORMAL);
    imshow("corners_window1", dst_norm_scaled1);

    namedWindow("corners_window2", WINDOW_NORMAL);
    imshow("corners_window2", dst_norm_scaled2);
    
}
```
