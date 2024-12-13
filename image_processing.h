#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H


#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


class image_processing
{
public:

    void bilatFilter(const string& imagePath);
    void gaussFilter(const string& imagePath);
    void medianFilter(const string& imagePath);
    static Mat sobelOperator(Mat img);

    cv::Mat prevFrame;

    void processOpticalFlow(cv::Mat &prevFrame, cv::Mat &frame, int waitTime);
    void processFarneBackOpticalFlow(const std::string& videoPath, int waitTime = 100);

    void object_detection(Mat &frame);
    void object_detection_hog(Mat &frame);
    void qrCodeDetection(Mat &frame);

};

#endif // IMAGE_PROCESSING_H
