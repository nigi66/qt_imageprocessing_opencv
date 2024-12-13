

#include "image_processing.h"


// //////////////////////////// image filtering

void image_processing::bilatFilter(const string& imagePath)
{
    Mat img = imread(imagePath, IMREAD_COLOR);

    Mat bilateralImg;
    bilateralFilter(img, bilateralImg, 15, 95,45);

    imshow("bilateral image", bilateralImg);
    imshow("image", img);
    waitKey(0);
}

void image_processing::gaussFilter(const string& imagePath)
{
    Mat img = imread(imagePath, IMREAD_COLOR);

    Mat gaussianImg;
    GaussianBlur(img, gaussianImg, Size(3,3),0);

    imshow("bilateral image", gaussianImg);
    imshow("image", img);
    waitKey(0);
}

void image_processing::medianFilter(const string& imagePath)
{
    Mat img = imread(imagePath, IMREAD_COLOR);

    Mat medianImg;
    medianBlur(img, medianImg, 15);

    imshow("bilateral image", medianImg);
    imshow("image", img);
    waitKey(0);
}

void kernel_2d()
{
    Mat_<float> custom(3,3);
    Mat_<float> kernel(2,2);


    Mat custom2, kernel2, filter2D, filter2D2;

    cv::filter2D(custom, filter2D, -1, kernel,Point(-1,-1));

    custom.convertTo(custom2, CV_8UC1);
    kernel.convertTo(kernel2, CV_8UC1);
    filter2D.convertTo(filter2D2, CV_8UC1);

}

// //////////////////////////// object tracking

void image_processing::processOpticalFlow(cv::Mat &prevFrame, cv::Mat &frame, int waitTime) {
    Ptr<DISOpticalFlow> disOpticalFlow = DISOpticalFlow::create();
    Mat  flow;


    // Configure DISOpticalFlow parameters
    disOpticalFlow->setFinestScale(5);
    disOpticalFlow->setGradientDescentIterations(16);
    disOpticalFlow->setPatchSize(8);
    disOpticalFlow->setPatchStride(1);
    disOpticalFlow->setUseMeanNormalization(1);
    disOpticalFlow->setUseSpatialPropagation(1);
    disOpticalFlow->setVariationalRefinementAlpha(1000);
    disOpticalFlow->setVariationalRefinementDelta(5);
    disOpticalFlow->setVariationalRefinementGamma(1);
    disOpticalFlow->setVariationalRefinementIterations(100);


       // cap >> frame; // Read the next frame
        if (frame.empty()) {
            cerr << "Frame is empty. Exiting." << endl;
            //break;
        }

        // Convert frame to grayscale
        try {
            cvtColor(frame, frame, COLOR_BGR2GRAY);
            cvtColor(prevFrame, prevFrame, COLOR_BGR2GRAY);
        } catch (const cv::Exception& e) {
            cerr << "OpenCV Error: " << e.what() << endl;
            //break;
        }

        // Compute Optical Flow if a previous frame exists
        if (!prevFrame.empty()) {
            disOpticalFlow->calc(prevFrame, frame, flow);

            // Draw the optical flow vectors
            for (int y = 0; y < frame.rows; y += 20) {
                for (int x = 0; x < frame.cols; x += 20) {
                    Point2f flowAtPoint = flow.at<Point2f>(y, x);

                    // Draw a line representing flow at this point
                    line(frame, Point(x, y), Point(cvRound(x + flowAtPoint.x), cvRound(y + flowAtPoint.y)), Scalar(0, 0, 0), 2);

                    // Draw a small circle at the starting point
                    circle(frame, Point(x, y), 1, Scalar(0, 255, 0), 1);
                }
            }
        }

}


void image_processing::processFarneBackOpticalFlow(const std::string& videoPath, int waitTime) {

    Ptr<FarnebackOpticalFlow> farneOpticalFlow = FarnebackOpticalFlow::create();
    Mat frame, prevFrame, flow;

    // Open the video file or webcam
    VideoCapture cap(videoPath); // Use 0 for webcam
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open video or webcam." << endl;
        return;
    }

    farneOpticalFlow->setPyrScale(0.5);
    farneOpticalFlow->setFlags(0);
    farneOpticalFlow->setNumIters(10);
    farneOpticalFlow->setWinSize(20); // default = 13
    farneOpticalFlow->setFastPyramids(0);
    farneOpticalFlow->setNumLevels(1);

    while (true) {
        cap >> frame; // Read the next frame
        if (frame.empty()) {
            cerr << "Frame is empty. Exiting." << endl;
            break;
        }

        try {
            cvtColor(frame, frame, COLOR_BGR2GRAY);
        } catch (const cv::Exception& e) {
            cerr << "OpenCV Error: " << e.what() << endl;
            break;
        }


        if (!prevFrame.empty()) {
            farneOpticalFlow->calc(prevFrame, frame, flow);

            // Draw the optical flow vectors
            for (int y = 0; y < frame.rows; y += 20) {
                for (int x = 0; x < frame.cols; x += 20) {
                    Point2f flowAtPoint = flow.at<Point2f>(y, x);

                    line(frame, Point(x, y), Point(cvRound(x + flowAtPoint.x), cvRound(y + flowAtPoint.y)), Scalar(0, 0, 0), 2);

                    circle(frame, Point(x, y), 1, Scalar(0, 255, 0), 1);
                }
            }
        }

        prevFrame = frame.clone();

        imshow("Optical Flow Output", frame);

        if (waitKey(waitTime) == 27) { // Exit on 'ESC' key
            break;
        }
    }
}


// //////////////////////////// object detection



void image_processing::object_detection(Mat &frame)
{
    //VideoCapture cap(0);
    CascadeClassifier cascade;

    cascade.load("C:\\opencv-4.5.4\\build\\install\\etc\\haarcascades\\haarcascade_frontalface_default.xml");

    Mat grayImg;

    //cap >> frame;
    cvtColor(frame, grayImg, COLOR_BGR2GRAY);
    vector<Rect> faces;

    cv::equalizeHist(grayImg, grayImg);
    cascade.detectMultiScale(grayImg, faces, 1.05, 3, 0, Size(), Size());

    for (size_t i=0; i<faces.size(); i++)
    {
        rectangle(frame, faces[i], Scalar(255,0,0), 2);
    }
}

void image_processing::object_detection_hog(Mat &frame)
{
    HOGDescriptor  hog;

    hog.setSVMDetector(hog.getDefaultPeopleDetector());

    vector<Rect> detectedRects;
    vector<double> detectedWeights;

    hog.detectMultiScale(frame, detectedRects, detectedWeights, 0, Size(), Size(), 1.01);

    for (size_t i=0; i<detectedRects.size(); i++)
    {
        rectangle(frame, detectedRects[i], Scalar(0,255,0), 2);
    }
}


void image_processing::qrCodeDetection(Mat &frame)
{

    QRCodeDetector qrDetector;

    Mat points, result;
    string qrData = qrDetector.detectAndDecode(frame, points, result);

    std::vector<cv::Rect> rects;
    for (int i = 0; i < points.cols; ++i) {
        rects.push_back(points.at<cv::Rect>(0, i));
    }

    for (size_t i=0; i<qrData.size(); i++)
    {
        rectangle(frame, rects[i], Scalar(0,255,0), 2);
    }


    cout << "points: " << points << endl;
    cout << "website: " << qrData << endl;
    cout << "result: " << result << endl;

}


