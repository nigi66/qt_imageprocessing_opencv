#include "mainwindow.h"

#include <QApplication>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <image_processing.h>

#include <iostream>
#include <memory>
#include <algorithm>

using namespace cv;
using namespace std;

/**
 * Function to process a video or webcam stream and compute DIS Optical Flow.
 * @param videoPath Path to the video file (use an empty string for webcam).
 * @param waitTime Time to wait between frames (ms).
 */



class TestClass {
public:

    TestClass(){
        cout << "Constructor invoked" << endl;
    }
    ~TestClass(){
        cout << "Destructor invoked" << endl;
    }
};

int add(int a, int b){
    return a+b;
}

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    MainWindow mainWindow;
    mainWindow.show();


    //cout << cv::cuda::getCudaEnabledDeviceCount() << endl;

    return app.exec();



    return 0;
}


