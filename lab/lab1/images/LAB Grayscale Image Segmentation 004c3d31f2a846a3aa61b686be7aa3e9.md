# LAB: Grayscale Image Segmentation

**Date:**  2024-March-30

**Author:**  Eunji Ko 22100034

**Github:** 

[DLIP/LAB1 at main · eunjijuliako/DLIP](https://github.com/eunjijuliako/DLIP/tree/main/LAB1)

---

# Introduction

## 1. Objective

The purpose of this lab is to detect each different Nut and Bolt by Image Processing with all of the functions in OpenCV.

**Goal**: Count the number of nuts and bolts as the different types.

- M5 Bolt
- M6 Bolt
- M5 Hexa Nut
- M6 Hexa Nut
- M5 Rectangular Nut

## 2. Preparation

### Software Installation

- OpenCV 4.90, Visual Studio 2022

### Dataset

**Dataset link:** [Download the test image](https://github.com/eunjijuliako/DLIP/blob/main/LAB1/Image/grayscaled_source.png) 

# Algorithm

## 1. Overview

The overview of this project is the same as the below Figure 1. It is a summarization of all processes.

![Figure1. Overview Diagram](LAB%20Grayscale%20Image%20Segmentation%20004c3d31f2a846a3aa61b686be7aa3e9/KakaoTalk_Photo_2024-03-30-21-20-26_001.png)

Figure1. Overview Diagram

## 2. Procedure

### Histogram Analysis

![Figure2. Grayscaled Source Image ](https://github.com/eunjijuliako/DLIP/blob/main/lab/lab1/images/grayscaled_source.png?raw=true)

Figure2. Grayscaled Source Image 

![Figure3. Gray Scaled Histogram](https://github.com/eunjijuliako/DLIP/blob/main/lab/lab1/images/grayscaled_histogram.png?raw=true)

Figure3. Gray Scaled Histogram

Figure 3 is a histogram plot of a gray-scaled source image. The objects that we want to see are in a small area of intensity around 255. Most of the intensity of the image is around 0 to 128. To make the image clearer, and to make the contrast of intensity higher, we have to consider what filter will be used.

### Filtering

**Process :** Gaussian → Otsu → Median → Sobel

Since there are shadows and noises on the source image, filters should be applied. There are some types of smoothing filters, and we will compare those smoothing filters with histogram plots.

![Figure4. Gaussian and Averaging Filter](https://github.com/eunjijuliako/DLIP/blob/main/lab/lab1/images/Gau&Ave.png?raw=true)

Figure4. Gaussian and Averaging Filter

First, Gaussian blurring is a smoothing filter, which eliminates the noise of the image but maintains a similar image with the source image. (Figure4)

Second, the average blurring is a filter which makes the image smooth overall.

![Figure5.  Histogram of Gaussian and Averaging Filter](https://github.com/eunjijuliako/DLIP/blob/main/lab/lab1/images/Gau&Ave%20histo.png?raw=true)

Figure5.  Histogram of Gaussian and Averaging Filter

The intensity of the object is around 255, and the value of the Gaussian histogram around 255 is higher than filter 2D as Figure 5. Therefore, the Gaussian filter is an appropriate filter for making the bigger contrast of intensity of the source image.

### Thresholding and Morphology

- Thresholding

![Figure6. Global and Adaptative Thresholding](https://github.com/eunjijuliako/DLIP/blob/main/lab/lab1/images/Glob&Adap.png?raw=true)

Figure6. Global and Adaptative Thresholding

When we compare the Global and adaptive thresholding(Figure 6), we can see that the global thresholding is hard to remove the noise on the top of the image. However, dividing the object with the background is most important to segmentate the objects. Therefore, I applied the Otsu thresholding to maximize the class variation. 

As a result(Figure 7), most of the segments are divided well. Still, there are salt noises on the left top of the image. Therefore, I applied a Median filter to the image.

![Figure8. Threshold Otsu](https://github.com/eunjijuliako/DLIP/blob/main/lab/lab1/images/Otsu.png?raw=true)

Figure8. Threshold Otsu

![Figure9. Median Filter](https://github.com/eunjijuliako/DLIP/blob/main/lab/lab1/images/Medi.png?raw=true)

Figure9. Median Filter

We can check the salt noises on the top left corner of Figure 9 were eliminated after applying the median filter.

![Figure10. Sobel Filter](LAB%20Grayscale%20Image%20Segmentation%20004c3d31f2a846a3aa61b686be7aa3e9/Sobel1.png)

Figure10. Sobel Filter

To get the more clear image for separating the two rectangular nuts, I applied a Sobel filter. Sobel filter is a sharpening filter which use differentiation.

- Morphology

**Process :** n = 8 → Erode → Dilate → Erode → Sobel → n = 3→ Dilate → Erode → Dilate → Dilate → Dilate → Open→ Erode

The process is same as above, a total of 4 times erode, 5 times Dilate, and once Sobel and Open method to accurately calculate the area and length of the object in the next process, segmentation. Through this process, the two attached rectangular nuts in the lower left should be separated as possible, while at the same time preventing the pixels for the holes inside from being broken. Also, the intensity of the M5 hexa nuts underneath the two rectangular nuts should not reduced due to shadows. Moreover, I changed the size of the Kernel from 3 to 8, and from 8 to 3 to adjust the intensity of the filter or morphology.

![Figure11. Image Table of Morphology Process](LAB%20Grayscale%20Image%20Segmentation%20004c3d31f2a846a3aa61b686be7aa3e9/Table.png)

Figure11. Image Table of Morphology Process

### Contouring

**Process:** Count the number of Children → Separate two Square Bolt → Segmentation as the size and length of bolts

First, I used the hierarchy value obtained through the ‘find contour’ function to separate the attached rectangular nuts.

When you use the function, you can select options for how to perform contouring with the parameters. I chose the ‘RETR_TREE’ option, which contours both the outline and the inline, and recognizes the outline as the parent and the inline as the child. Additionally, with ‘CHAIN_APPROX_SIMPLE’, the straight-line component of the contour was saved by simplifying only the endpoints.

When two different objects are attached and recognized as one object, you can distinguish whether they are two different objects or one object by counting the number of holes in them, which is, the number of children. If the number of children is two or more, different objects are attached.

In the entire loop that contours the object, we first count the number of children of each object, and if there are more than two, we are focusing on two attached rectangular nuts, so we add the number of rectangular nuts equal to the number of children.

In the code, if the number of children is one, it moves on to the next conditional statement and distinguishes the object through characteristics (length and width) according to type.

In the entire loop that contours the object, we first count the number of children of each object, and if there are more than two, we can cosider as two attached rectangular nuts, so we add the number of rectangular nuts equal to the number of children.

**When the number of childeren is more than 2**

```cpp
for (int i = 0; i < contours.size(); i++) {								 // Counting the number of child
		int childIndex = hierarchy[i][2];											 // Index of first child
						int childCount = 0;														 // Initialization
			while (childIndex != -1) {													 // When it has child
								childCount++;															 // Count the num of child
	childIndex = hierarchy[childIndex][0];									 // Move to the next index of child
}

if (childCount >= 2) {                                     // If the num of child is more than 2
	drawContours(drawing, contours, i, grn, FILLED, LINE_4);
	SqNM5 = SqNM5 + childCount;                              // Add as the num of child (only for Sqare case)
}}
```

**When the number of childeren is one**

If the number of children is one, it moves on to the next conditional statement and distinguishes the object through characteristics (length and width) according to type.

- **Bolt M6**

First, the characteristic of BM6 is the long bolt length. First, we determined the relatively long length of BM6 through the length and width of all objects, and then used a conditional statement to recognize objects larger than that as BM6.

```cpp
int BM6_len = 645;

else if (arcLength(contours[i], true) > BM6_len) {
	drawContours(drawing, contours, i, red, FILLED, LINE_4);
	BM6++;
}
```

- **Bolt M5**

Although it is shorter than BM6, it is longer than other objects, so BM5 was also classified by length.

```cpp
else if (arcLength(contours[i], true) > BM5_len) {
	drawContours(drawing, contours, i, pnk, FILLED, LINE_4);
	BM5++;
}
```

- **Hexa Nut M5, Suare Nut M5, Hexa Nut M6**

The remaining objects had sequentially wide widths and long lengths, so they were classified according to the specific distinguishable values of each object.

```cpp
else if (arcLength(contours[i], true) > HxNM5_len && contourArea(contours[i]) > HxNM5_ara) {
	drawContours(drawing, contours, i, org, FILLED, LINE_4);
	HxNM5++;
}
else if (arcLength(contours[i], true) > SqNM5_len && contourArea(contours[i]) > Sq_ara) {
	drawContours(drawing, contours, i, grn, FILLED, LINE_4);
	SqNM5++;
}
else if (arcLength(contours[i], true) > min) {
	drawContours(drawing, contours, i, mnt, FILLED, LINE_4);
	HxNM6++;
}
```

# Result and Discussion

## 1. Final Result

![Figure12. Final Output of colored Segmentation](LAB%20Grayscale%20Image%20Segmentation%20004c3d31f2a846a3aa61b686be7aa3e9/Final_Result_P.png)

Figure12. Final Output of colored Segmentation

![Figure13. Final Output of Segmentation](LAB%20Grayscale%20Image%20Segmentation%20004c3d31f2a846a3aa61b686be7aa3e9/Final_Result_W.png)

Figure13. Final Output of Segmentation

**( M5 Bolt : Pink,**

  **M6 Bolt : Red,**

  **M5 Hexa Nut : Mint,**

  **M6 Hexa Nut : Orange,**

  **M5 Rectangular Nut : Green )**

The result of automatic part segmentation is shown with color in the figure left. Also, the counting output of each nut and bolt are shown in figure right.

The color of the M5 Bolt is Pink, and the number of it is 5. The color of the M6 Bolt is Red, and the number of it is 3. The color of the M5 Hexa Nut is Mint, and the number of it is 4. The color of the M6 Hexa Nut is orange, and the number of it is 4. The color of the M5 Rectangular Nut is green, and the number of it is 5. 

## 2. Discussion

| Items | True | Estimated | Accuracy |
| --- | --- | --- | --- |
| M5 Bolt | 5 | 5 | 100% |
| M6 Bolt | 3 | 3 | 100% |
| M5 Hex Nut | 4 | 4 | 100% |
| M6 Hex Nut | 4 | 4 | 100% |
| M5 Rec Nut | 5 | 5 | 100% |

Since this project’s objective segmentation accuracy is 100% for each item, the proposed algorithm is an appropriate one for this LAB. 

# Conclusion

**Summarize the Project Goal and Results.**

The goal of this project was to detect each different Nut and Bolt by Image Processing. There were two small goals of this project,  and the first one was to distinguish two attached objects which is recognized as one object to two different objects. The Second was preventing the disappearance of the M5 Hexa Nut because of the shadow. 

As a result, the accuracy of segmentation and detection was 100% for each item. The algorithm was successful because of detected the number of holes and morphology. Detecting the number of holes was held with the hierarchy arrays to separate the two attached rectangular nuts, and morphology was applied appropriately for the M5 Hexa Nut.

**Algorithm Problems and Expected Improvement**

Among my algorithms, when there are two or more children in a parent-child relationship, that is, when they are identified as one object but are attached and identified as one, there is a method that adds the number of holes to the number of objects. However, in my algorithm, if two holes are found in a rectangular while already knowing that two rectangular nuts are glued together, the number of holes is added to the number of rectangular.

However, to perfectly implement this algorithm, not only the rectangular case but also several cases must be considered. For example, when this algorithm is applied to a situation where two hexagonal bolts are attached, the number of hexagonal bolts will not be added but will be added to the rectangular nuts.

To improve this problem, it is necessary to determine the number of children, that is, what kind of polygon it is when the number of holes is more than two. Therefore, you may identify polygons using OpenCV’s ‘Harris Corner Detection’ function, which determines the number of corners. In addition, by using the’ CHAIN_APPROX_SIMPLE’ option in the ‘findContours()’ function, only the vertices of the border are returned, allowing you to distinguish between circles and polygons based on the ratio of the length and width of the border.

---

# Appendix

---

**Threshold code**

```cpp
int threshold_type_Otsu = 100; // Otsu
int const max_BINARY_value = 255;

threshold(dst_Gau, dst_Otsu, threshold_type_Otsu, max_BINARY_value, THRESH_OTSU);
```

parameters of threshold:

`input = dst_Gau` 

`output = dst_Otsu`

`thresh = threshold_type_Otsu` : reference threshold

`maxval = max_BINARY_value` : maximum threshold

`type = THRESH_OTSU`

There are some types of threshold, and we used the ‘THRESH_OTSU’ which thresholding with appropriate threshold and the otsu algorithm. We set the Otsu threshold value as 100 because when we analysis the histogram of the Gaussian, there is a gap between 0 and 255, and the dividing point seems can be estimated around 100. 

---

```cpp
/*------------------------------------------------------/
* Class   :  Image Proccessing with Deep Learning
* OpenCV  :  LAB1 : Grayscale Image Segmentation
* Created :  3/30/2024
* Name    :  Eunji Ko
* Number  :  22100034
------------------------------------------------------*/

#include "opencv.hpp"
#include <iostream>
#include <time.h>

using namespace cv;
using namespace std;

void plotHist(Mat, string, int, int);

String window_name = "Final Output";

vector<vector<Point>> contours;
vector<Vec4i> hierarchy;

int main(int argc, char* argv[]) {
	
	/* Initialize variables */

	int i = 3;
	int delta = 0;
	int ddepth = -1;
	int kernel_size = 5;
	Point anchor = Point(-1, -1);
	Mat src_gray,inv;
	Mat kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);

	// Filtering
	Mat dst_Adap, dst_Glob, dst_Otsu, dst_Medi, dst_Gau;
	int threshold_value = 100;
	int const max_value = 255;
	int const max_BINARY_value = 255;

	// Sobel
	int dx = 1;
	int dy = 1;
	Mat scaled;
	Mat addsc;
	Mat dst_Sob;
	int scale = 1;
	Mat Sob1;
	Mat dst_Sob1;
	Mat Sob2;
	Mat dst_Sob2;

	// Thresholding
	int threshold_type_Otsu = 100; 

	// Morphology
	int element_shape = MORPH_RECT;		
	int n = 3; // Kernel size
	Mat element = getStructuringElement(element_shape, Size(n, n));
	Mat dst_dila;
	Mat dst_erod1;
	Mat dst_dila1;
	Mat dst_erod2;
	Mat dst_dila2;
	Mat dst_erod3;
	Mat dst_dila3;
	Mat dst_open;
	Mat dst_dila4;
	Mat dst_dila5;
	Mat dst_erod4;

	//Mat dst_clos;

	// Contouring and Segmentation
	int BM6 = 0;
	int BM5 = 0;
	int SqNM5 = 0;
	int HxNM5 = 0;
	int HxNM6 = 0;

	// Colors
	Scalar red = cv::Scalar(0, 0, 255);
	Scalar pnk = cv::Scalar(255, 0, 255);
	Scalar grn = cv::Scalar(0, 255, 0);
	Scalar org = cv::Scalar(0, 100, 200);
	Scalar mnt = cv::Scalar(200, 200, 0);

	// Objects minimum length and Area
	int BM6_len = 645;
	int BM5_len = 500;
	int HxNM5_len = 200;
	int HxNM5_ara = 6000;
	int SqNM5_len = 300;
	int Sq_ara = 4700;
	int min = 200; // ignore the salt noises   

	/* Load the Image */

	src_gray = imread("Lab_GrayScale_TestImage.jpg", 0);

	/* Filtering and Thresholding */
	
	GaussianBlur(src_gray, dst_Gau, Size(i, i), 0, 0);								 // Gaussian Filtering
    threshold(dst_Gau, dst_Otsu, threshold_type_Otsu, max_BINARY_value, THRESH_OTSU);// Otsu Thresholding
	medianBlur(dst_Otsu, dst_Medi, i);											     // Midian Blurring
	
	Sobel(dst_Medi, Sob1, ddepth, dx, dy, kernel_size, scale, delta);				 // Sobel Filtering
	convertScaleAbs(Sob1, scaled);													 // Scale, converting back to CV_8U
	add(scaled, dst_Medi, addsc);												     // Add src and scaled images
	convertScaleAbs(addsc, dst_Sob1);												 // Scale 

	/* Morphology */

	n = n + 5;																		 // Increase the kernel size to 8
	erode(dst_Sob1, dst_erod1, element);
	dilate(dst_erod1, dst_dila1, element);											 // Closing again to connect
	erode(dst_dila1, dst_erod2, element);

	Sobel(dst_erod2, Sob2, ddepth, dx, dy, kernel_size, scale, delta);
	convertScaleAbs(Sob2, scaled);
	add(scaled, dst_erod2, addsc);										
	convertScaleAbs(addsc, dst_Sob2);

	n = 3;																			 // Decrease the kernel size to 3
	dilate(dst_Sob2, dst_dila2, element);
	namedWindow("5", WINDOW_NORMAL);
	imshow("5", dst_dila2);

	erode(dst_dila2, dst_erod3, element);
	namedWindow("6", WINDOW_NORMAL);
	imshow("6", dst_erod3);

	dilate(dst_erod3, dst_dila3, element);
	namedWindow("7", WINDOW_NORMAL);
	imshow("7", dst_dila3);

	dilate(dst_dila3, dst_dila4, element);
	namedWindow("8", WINDOW_NORMAL);
	imshow("8", dst_dila4);

	dilate(dst_dila4, dst_dila5, element);
	namedWindow("9", WINDOW_NORMAL);
	imshow("9", dst_dila5);

	morphologyEx(dst_dila5, dst_open, MORPH_OPEN, element,Point(0,0),3);				 // Opening to eliminate the salt noises
	namedWindow("10", WINDOW_NORMAL);
	imshow("10", dst_open);

	erode(dst_open, dst_erod4, element); 
	namedWindow("11", WINDOW_NORMAL);
	imshow("11", dst_erod4);

	/* Find the contours */

	findContours(dst_erod4, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);    // RETR_TREE for relationship between parent and child
	Mat drawing(dst_erod3.size(), CV_8UC3, Scalar(0, 0, 0));                         // Draw all contours excluding holes in black background
	
	/* Counting the number of Objects*/
	
	for (int i = 0; i < contours.size(); i++) {
																					 // Counting the number of child
		int childIndex = hierarchy[i][2];											 // Index of fisr child
		int childCount = 0;															 // Initialization
		while (childIndex != -1) {													 // When it has child
			childCount++;															 // Count the num of child
			childIndex = hierarchy[childIndex][0];									 // Move to the next index of child
		}

		if (childCount >= 2) {                                                       // If the num of child is more than 2
			drawContours(drawing, contours, i, grn, FILLED, LINE_4);
			SqNM5 = SqNM5 + childCount;                                              // Add as the num of child (only for Sqare case)
		}
		else if (arcLength(contours[i], true) > BM6_len) {
			drawContours(drawing, contours, i, red, FILLED, LINE_4);
			BM6++;
		}
		else if (arcLength(contours[i], true) > BM5_len) {
			drawContours(drawing, contours, i, pnk, FILLED, LINE_4);
			BM5++;
		}
		else if (arcLength(contours[i], true) > HxNM5_len && contourArea(contours[i]) > HxNM5_ara) {
			drawContours(drawing, contours, i, org, FILLED, LINE_4);
			HxNM5++;
		}
		else if (arcLength(contours[i], true) > SqNM5_len && contourArea(contours[i]) > Sq_ara) {
			drawContours(drawing, contours, i, grn, FILLED, LINE_4);
			SqNM5++;
		}
		else if (arcLength(contours[i], true) > min) {
			drawContours(drawing, contours, i, mnt, FILLED, LINE_4);
			HxNM6++;
		}
	}
	
	/* Results */

	cout << " M5 Bolt       =" << BM5 << endl;
	cout << " M6 Bolt       =" << BM6 << endl;
	cout << " M5 Hex Nut    =" << HxNM5 << endl;
	cout << " M6 Hex Nut    =" << HxNM6 << endl;
	cout << " M5 Rec Nut    =" << SqNM5 << endl;

	namedWindow(window_name, WINDOW_NORMAL);
	imshow(window_name, drawing);

	waitKey(0);
	return 0;
}

void plotHist(Mat src, string plotname, int width, int height) {

	/* Compute the histograms */ 

	Mat hist;
	int histSize = 256;								
	float range[] = { 0, 256 };						
	const float* histRange = { range };
	calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange);

	double min_val, max_val;
	cv::minMaxLoc(hist, &min_val, &max_val);
	Mat hist_normed = hist * height / max_val;		
	float bin_w = (float)width / histSize;			
	Mat histImage(height, width, CV_8UC1, Scalar(0));	
	for (int i = 0; i < histSize - 1; i++) {									
		line(histImage,																			
			Point((int)(bin_w * i), height - cvRound(hist_normed.at<float>(i, 0))),				
			Point((int)(bin_w * (i + 1)), height - cvRound(hist_normed.at<float>(i + 1, 0))),	
			Scalar(255), 2, 8, 0);													
	}
	
	histImage = Scalar(255, 255, 255) - histImage;
	namedWindow(plotname, WINDOW_NORMAL);
	imshow(plotname, histImage);
}

```

![https://github.com/eunjijuliako/DLIP/blob/main/LAB1/Image/Final_Result_P.png](https://github.com/eunjijuliako/DLIP/blob/main/LAB1/Image/Final_Result_P.png)