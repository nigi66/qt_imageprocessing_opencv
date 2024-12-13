#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QFileDialog>
#include <QPixmap>
#include <QTimer>
#include <QThread>

using namespace cv;
using namespace std;


QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();


protected:
    void mousePressEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;

private:
    Ui::MainWindow *ui;

    void keyPressEvent(QKeyEvent *event);

    int dx = 1;                // Gradient in x-direction
    int dy = 1;                // Gradient in y-direction
    int sobelKernelSize = 3;   // Sobel kernel size
    int scaleFactor = 1;       // Scale factor
    int deltaValue = 0;        // Delta value

    cv::Mat originalImage;       // Store the original image
    cv::Mat currentImage;        // Store the current image after filtering

    void displayImage(const cv::Mat& image);
    cv::Mat applyFilter(const QString &filter); // Helper function to process filters

    // ///////////////////////// Video
    cv::VideoCapture videoCapture; // For video reading
    bool isVideoLoaded;            // To track if a video is loaded
    QString videoFilePath;         // Path to the loaded video
    bool isCameraActive;

    void displayFrame(const cv::Mat &frame); // Helper function to display video frames

    // ///////////////////////// Drawing
    bool isDrawing;              // To track if the user is drawing
    QPoint startPoint, endPoint; // Points for line or circle drawing
    QString selectedFilter;       // Current selection from ComboBox (e.g., Line, Circle)

    void drawShape();            // Draw the selected shape
    QPoint mapToImageCoordinates(const QPoint &widgetPoint);

private slots:
    void onBrowseImage(); // Slot to handle button click
    void onApplyFilter();        // Slot for applying the selected filter

    void onBrowseVideo();    // Slot to browse and load a video
    void onStartDetection(); // Slot to start object detection in the video

    void onConnectCamera();  // Slot to connect to the camera
    void onStopCamera();     // Slot to stop the camera feed
    void onObjectDetection();

};
#endif // MAINWINDOW_H

