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
    QApplication app(argc, argv); // Initialize the Qt application

    MainWindow mainWindow;        // Create the main window
    mainWindow.show();            // Show the main window


    vector<int> v{2,3,4,5,6};
    for_each(v.begin(), v.end(), [](int x){cout << x << endl;});




    system("PAUSE");


    //cout << cv::cuda::getCudaEnabledDeviceCount() << endl;

    return app.exec();            // Enter the event loop



    return 0;
}


