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

