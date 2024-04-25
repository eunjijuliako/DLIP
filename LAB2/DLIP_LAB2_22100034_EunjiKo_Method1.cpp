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
    src = imread("WL.jpg");
    //src = imread("WH.jpg");
    //src = imread("Test.jpg");

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