///*------------------------------------------------------/
//* Class   :  Image Proccessing with Deep Learning
//* OpenCV  :  LAB2 : Dimension Measurement with 2D camera
//* Created :  4/23/2024
//* Name    :  Eunji Ko, Hyeonho Moon
//* Number  :  22100034
//* 
//* Method 2:  Size Detection with Frontal Photo 
//------------------------------------------------------*/
//
//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include "cameraParam.h"
//#include <sstream>
//#include <iomanip>
//
//using namespace cv;
//using namespace std;
//
///*  0. Initialization  */
//
//// Global variables for Threshold
//int threshold_value = 220;
//int threshold_type = 1;
//int morphology_type = 0;
//
//// Global variables for Morphology
//int element_shape = MORPH_RECT;		
//int n = 3;
//Mat element = getStructuringElement(element_shape, Size(n, n));
//
//int const max_value = 255;
//int const max_type = 8;
//int const max_BINARY_value = 255;
//
//double Ratio;
//
//Mat src, dst, src1, src2, dst1, dst2, dst_norm_scaled1, dst_norm_scaled2, src_gray1, src_gray2;
//
//void dataset(void);
//void filtering(void);
//void findCorners_z(const Mat& src_gray, Mat& corners, const Mat& original);
//void findCorners_xy(const Mat& src_gray, Mat& corners, const Mat& original);
//void showresult(void);
//
//int main() {
//    
//    /* Read the Calibration Infromation and Source Image */
//    dataset();
//
//    /* Thresholding for Clear Corner Detection */
//    filtering();
//
//    /* Finding Corners through Contouring */
//    findCorners_z(src_gray1, dst_norm_scaled1, dst1);
//    findCorners_xy(src_gray2, dst_norm_scaled2, dst2);
//
//    /* Results */
//    showresult();
//
//    waitKey(0);
//    return 0;
//}
//
//
//void dataset(void) {
//
//    /* 1. Read the Callibration Information */
//    cameraParam param("calib_resource_moon.xml");                                       // Caliibration - Intrinsic Matrix of the phone
//
//    /* 2. Read the Source Image and Conver to Gray-scale Image */
//    src1 = imread("WL.jpg");
//    src2 = imread("WH.jpg");
//
//    dst1 = param.undistort(src1);
//    dst2 = param.undistort(src2);
//
//    cvtColor(dst1, src_gray1, COLOR_BGR2GRAY);
//    cvtColor(dst2, src_gray2, COLOR_BGR2GRAY);
//}
//
//
//void filtering(void) {
//
//    threshold(src_gray1, src_gray1, 220, max_BINARY_value, THRESH_BINARY); // Thresholding 1
//    erode(src_gray1, src_gray1, element, Point(-1, -1), 3);
//    dilate(src_gray1, src_gray1, element, Point(-1, -1), 4);
//
//    threshold(src_gray2, src_gray2, 180, max_BINARY_value, THRESH_BINARY); // Thresholding 2
//    erode(src_gray2, src_gray2, element, Point(-1, -1), 120);
//    dilate(src_gray2, src_gray2, element, Point(-1, -1), 140);
//}
//
//
//void findCorners_z(const Mat& src_gray, Mat& corners, const Mat& original) {
//    vector<vector<Point>> contours;
//    findContours(src_gray, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//
//    corners = original.clone();
//
//    const double minArea = 100.0;                                          // Minimum contour area
//    vector<Rect> boundingBoxes;                                            // Vector to store bounding boxes of contours
//
//            
//    for (const auto& contour : contours) {                                 // Extract bounding boxes for all contours that meet the minimum area requirement
//        if (contourArea(contour) > minArea) {
//            boundingBoxes.push_back(boundingRect(contour));
//        }
//    }
//
//    if (boundingBoxes.size() < 2) {
//        cerr << "Two objects are required for comparison!" << endl;
//        return;
//    }
//
//    /* 3.1 Length : Reference Value from User */
//    double referenceWidth;
//    cout << "Enter the reference Width for the first object: ";             // Get a real reference Width from the user
//    cin >> referenceWidth;
//
//    /* 4.1 Length : Calculate the size - [pixel] to [mm] */
//    double firstObjectWidth = boundingBoxes[0].width;                       // Calculate the dimensions of the first object
//    double firstObjectLength = boundingBoxes[0].height;
//
//    Ratio = referenceWidth / firstObjectWidth;
//
//    firstObjectLength = boundingBoxes[0].height * Ratio;
//
//   
//    double secondObjectRealWidth = boundingBoxes[1].width * Ratio;          // Calculate the real dimensions of the second object based on the first object's reference width
//    double secondObjectRealLength = boundingBoxes[1].height * Ratio;
//
//    /* 5.1 Display  */
//    Scalar color1 = Scalar(0, 255, 0);                                      // Display the first object with its reference dimensions
//
//    stringstream ss1;
//
//    ss1 << fixed << setprecision(2) << "W1: " << referenceWidth << ", L1: " << firstObjectLength; 
//    
//    putText(corners, ss1.str(), Point(boundingBoxes[0].x + 5, boundingBoxes[0].y + 20), FONT_HERSHEY_SIMPLEX, 2, color1, 3);
//
//    Scalar color2 = Scalar(0, 0, 255);                                      // Display the second object with its calculated dimensions
//
//    stringstream ss2;
//
//    ss2 << fixed << setprecision(2) << "W2: " << secondObjectRealWidth << ", L2: " << secondObjectRealLength;
//
//    putText(corners, ss2.str(), Point(boundingBoxes[1].x + 5, boundingBoxes[1].y + 20), FONT_HERSHEY_SIMPLEX, 2, color2, 3);
//
//}
//
//void findCorners_xy(const Mat& src_gray, Mat& corners, const Mat& original) {
//
//    vector<vector<Point>> contours;
//    findContours(src_gray, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//
//    corners = original.clone();
//    const double minArea = 100.0;                                            // Minimum contour area
//
//    vector<Rect> boundingBoxes;                                              // For storing contour bounding boxes
//
//    for (const auto& contour : contours) {                                   // Extracting bounding boxes for all contours
//        if (contourArea(contour) > minArea) {
//            boundingBoxes.push_back(boundingRect(contour));
//        }
//    }
//
//    if (boundingBoxes.size() < 2) {
//        cerr << "Two objects are required for comparison!" << endl;
//        return;
//    }
//
//    /* 4.2 Height : Calculate the size - [pixel] to [mm] */
//    double firstObjectRealWidth = boundingBoxes[0].width * Ratio;             // Calculate the dimensions of the first object
//
//    double firstObjectRealHeight = boundingBoxes[0].height * Ratio;
//
//    double secondObjectRealWidth = boundingBoxes[1].width * Ratio;            // Calculate the real dimensions of the second object based on the first object's reference width
//
//    double secondObjectRealHeight = boundingBoxes[1].height * Ratio;
//
//    /* 5.2 Height : Display  */
//    Scalar color1 = Scalar(0, 255, 0);                                        // Display the first object with its reference dimensions
//
//    stringstream ss1;
//
//    ss1 << fixed << setprecision(2) << "W2: " << firstObjectRealWidth << ", H2: " << firstObjectRealHeight;
//
//    putText(corners, ss1.str(), Point(boundingBoxes[0].x + 5, boundingBoxes[0].y + 20), FONT_HERSHEY_SIMPLEX, 2, color1, 3);
//
//    Scalar color2 = Scalar(0, 0, 255);                                        // Display the second object with its calculated dimensions
//
//    stringstream ss2;
//
//    ss2 << fixed << setprecision(2) << "W1: " << secondObjectRealWidth << ", H1: " << secondObjectRealHeight;
//
//    putText(corners, ss2.str(), Point(boundingBoxes[1].x + 5, boundingBoxes[1].y + 20), FONT_HERSHEY_SIMPLEX, 2, color2, 3);
//}
//
//
//void showresult(void) {
//
//    namedWindow("filter", WINDOW_NORMAL);
//    imshow("filter", src_gray1);
//
//    namedWindow("filter1", WINDOW_NORMAL);
//    imshow("filter1", src_gray2);
//
//    namedWindow("corners_window1", WINDOW_NORMAL);
//    imshow("corners_window1", dst_norm_scaled1);
//
//    namedWindow("corners_window2", WINDOW_NORMAL);
//    imshow("corners_window2", dst_norm_scaled2);
//    
//}