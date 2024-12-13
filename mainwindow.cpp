#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QImage>
#include <QDebug>
#include <QPixmap>
#include <QMessageBox>
#include <QKeyEvent>
#include <image_processing.h>

using namespace std;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , isCameraActive(false)
    , isDrawing(false)
{
    ui->setupUi(this);

    //filter options
    ui->cmbFilters->addItem("None");
    ui->cmbFilters->addItem("Blur");
    ui->cmbFilters->addItem("Bilateral");
    ui->cmbFilters->addItem("Median");
    ui->cmbFilters->addItem("Gaussian Blur");
    ui->cmbFilters->addItem("Edge Detection (Sobel)");
    ui->cmbFilters->addItem("Edge Detection (Canny)");
    ui->cmbFilters->addItem("Laplacian");
    ui->cmbFilters->addItem("Sharpen");
    ui->cmbFilters->addItem("sobelOperator");
    ui->cmbFilters->addItem("SVD");

    // Draw Options
    ui->cmbFilters->addItem("Line");
    ui->cmbFilters->addItem("Rectangle");
    ui->cmbFilters->addItem("Circle");
    ui->cmbFilters->addItem("Ellipse");


    // Connect signals to slots
    connect(ui->btnBrowse, &QPushButton::clicked, this, &MainWindow::onBrowseImage);
    connect(ui->btnApplyFilter, &QPushButton::clicked, this, &MainWindow::onApplyFilter);


    // Connect buttons to their respective slots
    connect(ui->btnBrowseVideo, &QPushButton::clicked, this, &MainWindow::onBrowseVideo);
    connect(ui->btnStartDetection, &QPushButton::clicked, this, &MainWindow::onStartDetection);

    // camera connection
    connect(ui->btnStartCamera, &QPushButton::clicked, this, &MainWindow::onConnectCamera);
    connect(ui->btnStopCamera, &QPushButton::clicked, this, &MainWindow::onStopCamera);
    connect(ui->btnObjectDetect, &QPushButton::clicked, this, &MainWindow::onObjectDetection);
}


MainWindow::~MainWindow()
{
    delete ui;
}

// ///////////////////////////// Image

void MainWindow::onBrowseImage()
{
    QString filePath = QFileDialog::getOpenFileName(this, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)");

    if (filePath.isEmpty()) {
        return;
    }

    originalImage = cv::imread(filePath.toStdString(), cv::IMREAD_COLOR); // Load image
    if (originalImage.empty()) {
        QMessageBox::warning(this, "Error", "Failed to load the image!");
        return;
    }

    currentImage = originalImage.clone(); // Initialize currentImage with the original
    displayImage(currentImage);           // Display the original image
}

// Slot: Apply the selected filter
void MainWindow::onApplyFilter()
{
    if (originalImage.empty()) {
        QMessageBox::warning(this, "Error", "No image loaded!");
        return;
    }

    // Get the selected filter from the combo box
    selectedFilter = ui->cmbFilters->currentText();

    // Apply the filter to the image
    cv::Mat filteredImage = applyFilter(selectedFilter);

    // Update the current image and display it
    currentImage = filteredImage;
    displayImage(currentImage);
}

// Helper: Display an OpenCV image in QLabel
void MainWindow::displayImage(const cv::Mat &image)
{
    cv::Mat rgbImage;
    cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB); // Convert BGR to RGB

    QImage qImage(rgbImage.data, rgbImage.cols, rgbImage.rows, rgbImage.step, QImage::Format_RGB888);
    QPixmap pixmap = QPixmap::fromImage(qImage).scaled(ui->lblImage->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);

    ui->lblImage->setPixmap(pixmap);
}

// Helper: Apply the selected filter
cv::Mat MainWindow::applyFilter(const QString &filter)
{
    cv::Mat result = originalImage.clone();

    if (filter == "Blur") {
        cv::blur(originalImage, result, cv::Size(5, 5));
    }
    else if(filter == "Bilateral"){
        cv::bilateralFilter(originalImage, result, 20, 90, 45);
    }
    else if(filter == "Median"){
        cv::medianBlur(originalImage, result, 3);
    }
    else if (filter == "Gaussian Blur") {
        cv::GaussianBlur(originalImage, result, cv::Size(5, 5), 1.5);
    }
    else if (filter == "Edge Detection (Sobel)") {
        cv::Mat gray, gradX, gradY;
        cv::cvtColor(originalImage, gray, cv::COLOR_BGR2GRAY);
        cv::Sobel(gray, gradX, CV_16S, 1, 0, 3);
        cv::Sobel(gray, gradY, CV_16S, 0, 1, 3);
        cv::convertScaleAbs(gradX, gradX);
        cv::convertScaleAbs(gradY, gradY);
        cv::addWeighted(gradX, 0.5, gradY, 0.5, 0, result);
        cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
    }
    else if (filter == "Edge Detection (Canny)") {
        cv::Mat gray, edges;
        cv::cvtColor(originalImage, gray, cv::COLOR_BGR2GRAY);
        cv::Canny(gray, edges, 50, 150);
        cv::cvtColor(edges, result, cv::COLOR_GRAY2BGR);
    }
    else if (filter == "Laplacian") {
        cv::Mat gray, edges;
        cv::cvtColor(originalImage, gray, cv::COLOR_BGR2GRAY);
        cv::Laplacian(gray, edges, -1, 1, 1, 0);
        cv::cvtColor(edges, result, cv::COLOR_GRAY2BGR);
    }
    else if (filter == "Sharpen") {
        cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
                          0, -1,  0,
                         -1,  5, -1,
                          0, -1,  0);
        cv::filter2D(originalImage, result, -1, kernel);
    }
    else if (filter == "sobelOperator") {
        // Sobel Operator Parameters

        cv::Mat gray;
        cv::cvtColor(originalImage, gray, cv::COLOR_BGR2GRAY);

        // Apply Sobel filter
        cv::Mat gradX, gradY;
        cv::Sobel(gray, gradX, CV_16S, dx, 0, sobelKernelSize, scaleFactor, deltaValue);
        cv::Sobel(gray, gradY, CV_16S, 0, dy, sobelKernelSize, scaleFactor, deltaValue);
        cv::convertScaleAbs(gradX, gradX);
        cv::convertScaleAbs(gradY, gradY);

        // Combine gradients
        cv::addWeighted(gradX, 0.5, gradY, 0.5, 0, result);

        // Convert to 3-channel for display purposes
        cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
    }
    else if (filter == "SVD") {

        Mat U, W, Vt, imgGray;
        originalImage.convertTo(imgGray, CV_32F);
        cv::cvtColor(imgGray, imgGray, cv::COLOR_BGR2GRAY);

        cv::SVDecomp(imgGray, W, U, Vt);

        int threshold = 50;
        Mat W_compressed = cv::Mat::zeros(W.size(), W.type());

        for (int i=0; i<threshold; i++){
            W_compressed.at<double> (cv::Point(0,i)) = W.at<double>(Point(0,i));
        }


        result = U * cv::Mat::diag(W_compressed) * Vt;

        convertScaleAbs(result, result);
    }
    else if (filter == "None") {
        result = originalImage.clone();
    }
    return result;
}

void MainWindow::keyPressEvent(QKeyEvent *event) {
    switch (event->key()) {
        case Qt::Key_A:
            if (dx && dy)
                dx = 0;
            else
                dx = 1;
            qDebug() << "Toggled dx to " << dx;
            break;
        case Qt::Key_S:
            if (dx && dy)
                dy = 0;
            else
                dy = 1;
            qDebug() << "Toggled dy to " << dy;
            break;
        case Qt::Key_D:
            sobelKernelSize += 2;
            qDebug() << "Increased sobelKernelSize value to " << sobelKernelSize;
            break;
        case Qt::Key_F:
            if (sobelKernelSize > 1) sobelKernelSize -= 2;
            qDebug() << "Decreased sobelKernelSize factor to " << sobelKernelSize;
            break;
        case Qt::Key_Z:
            scaleFactor++; // Decrease scale factor
            qDebug() << "Increased scale factor to " << scaleFactor;
            break;
        case Qt::Key_X:
            scaleFactor--;
            qDebug() << "Decreased scale value to " << scaleFactor;
            break;
        case Qt::Key_C:
            deltaValue++; // Decrease delta
            qDebug() << "Increased delta value to " << deltaValue;
            break;
        case Qt::Key_V:
            deltaValue--;
            qDebug() << "Decreased delta value to " << deltaValue;
            break;
        default:
            QMainWindow::keyPressEvent(event); // Pass unhandled events to the parent class
    }

    // Apply filter and update UI
    currentImage = applyFilter("sobelOperatora");
    displayImage(currentImage); // Assuming you have a displayImage() function
}

// ////////////////////////////// Draw

void MainWindow::mousePressEvent(QMouseEvent *event)
{
    if (selectedFilter == "Line" || selectedFilter == "Circle" || selectedFilter == "Rectangle" || selectedFilter== "Ellipse") {
        isDrawing = true;
        startPoint = mapToImageCoordinates(event->pos()); // Record the starting point
    }
}

void MainWindow::mouseMoveEvent(QMouseEvent *event)
{
    if (isDrawing && (selectedFilter == "Line" || selectedFilter == "Circle" || selectedFilter == "Rectangle" || selectedFilter== "Ellipse")) {
        // Update end point while dragging
        endPoint = mapToImageCoordinates(event->pos());

        // Display a preview of the shape being drawn
        cv::Mat previewImage = currentImage.clone();

        if (selectedFilter == "Line") {
            cv::line(previewImage, cv::Point(startPoint.x(), startPoint.y()),
                     cv::Point(endPoint.x(), endPoint.y()), cv::Scalar(0, 255, 0), 2);
        } else if (selectedFilter == "Circle") {
            int radius = cv::norm(cv::Point(endPoint.x(), endPoint.y()) - cv::Point(startPoint.x(), startPoint.y()));
            cv::circle(previewImage, cv::Point(startPoint.x(), startPoint.y()), radius, cv::Scalar(255, 0, 0), 2);
        }
        else if (selectedFilter == "Rectangle"){
            cv::rectangle(previewImage, Point(startPoint.x(), startPoint.y()), Point(endPoint.x(), endPoint.y()), Scalar(0, 0, 255), 2);
        }
        else if (selectedFilter == "Ellipse"){
            int radius = cv::norm(cv::Point(endPoint.x(), endPoint.y()) - cv::Point(startPoint.x(), startPoint.y()));
            float angle = fastAtan2(endPoint.y() - startPoint.y(), endPoint.x()-startPoint.x());
            cv::ellipse(previewImage, Point(startPoint.x(), startPoint.y()), Size(radius, radius+50), angle, 0, 360, Scalar(0,255,0), 2);
        }

        displayImage(previewImage);
    }
}

void MainWindow::mouseReleaseEvent(QMouseEvent *event)
{
    if (isDrawing && (selectedFilter == "Line" || selectedFilter == "Circle" || selectedFilter == "Rectangle" || selectedFilter== "Ellipse")) {
        isDrawing = false;
        endPoint = mapToImageCoordinates(event->pos()); // Record the ending point

        // Draw the final shape on the image
        if (selectedFilter == "Line") {
            cv::line(currentImage, cv::Point(startPoint.x(), startPoint.y()),
                     cv::Point(endPoint.x(), endPoint.y()), cv::Scalar(0, 255, 0), 2);
        } else if (selectedFilter == "Circle") {
            int radius = cv::norm(cv::Point(endPoint.x(), endPoint.y()) - cv::Point(startPoint.x(), startPoint.y()));
            cv::circle(currentImage, cv::Point(startPoint.x(), startPoint.y()), radius, cv::Scalar(255, 0, 0), 2);
        }
        else if (selectedFilter == "Rectangle"){
            cv::rectangle(currentImage, Point(startPoint.x(), startPoint.y()), Point(endPoint.x(), endPoint.y()), Scalar(0, 0, 255), 2);
        }
        else if (selectedFilter == "Ellipse"){
            int radius = cv::norm(cv::Point(endPoint.x(), endPoint.y()) - cv::Point(startPoint.x(), startPoint.y()));
            float angle = fastAtan2(endPoint.y() - startPoint.y(), endPoint.x()-startPoint.x());
            cv::ellipse(currentImage, Point(startPoint.x(), startPoint.y()), Size(radius, radius+50), angle, 0, 360, Scalar(0,255,0), 2);
        }

        displayImage(currentImage); // Update the QLabel with the new image
    }
}


QPoint MainWindow::mapToImageCoordinates(const QPoint &widgetPoint)
{
    // Get the size of the QLabel and the original image
    QSize labelSize = ui->lblImage->size();
    QSize imageSize = QSize(originalImage.cols, originalImage.rows);

    // Calculate scale factors
    double scaleX = static_cast<double>(imageSize.width()) / labelSize.width();
    double scaleY = static_cast<double>(imageSize.height()) / labelSize.height();

    // Adjust for aspect ratio (letterboxing or pillarboxing)
    double aspectRatioLabel = static_cast<double>(labelSize.width()) / labelSize.height();
    double aspectRatioImage = static_cast<double>(imageSize.width()) / imageSize.height();

    double offsetX = 0, offsetY = 0;
    if (aspectRatioImage > aspectRatioLabel) {
        // Pillarboxing: Vertical black bars
        double scaledHeight = labelSize.width() / aspectRatioImage;
        offsetY = (labelSize.height() - scaledHeight) / 2.0 + 50;
        scaleY = scaleX;
    } else {
        // Letterboxing: Horizontal black bars
        double scaledWidth = labelSize.height() * aspectRatioImage;
        offsetX = (labelSize.width() - scaledWidth) / 2.0 ;
        scaleX = scaleY;
    }

    // Map the QLabel point to the original image point
    int x = (widgetPoint.x() - offsetX) * scaleX;
    int y = (widgetPoint.y() - offsetY) * scaleY;

    // Ensure the point is within image bounds
    x = (x < 0) ? 0 : (x >= originalImage.cols ? originalImage.cols - 1 : x);
    y = (y < 0) ? 0 : (y >= originalImage.rows ? originalImage.rows - 1 : y);

    QPoint qPoint = QPoint(x, y);

    return qPoint;
}


// ///////////////////////////// Video



// Slot: Browse for a video file
void MainWindow::onBrowseVideo()
{
    videoFilePath = QFileDialog::getOpenFileName(this, "Open Video File", "", "Videos (*.mp4 *.avi *.mkv)");

    if (videoFilePath.isEmpty()) {
        QMessageBox::warning(this, "Error", "No video file selected!");
        isVideoLoaded = false;
        return;
    }

    // Open the video using OpenCV
    videoCapture.open(videoFilePath.toStdString());
    if (!videoCapture.isOpened()) {
        QMessageBox::warning(this, "Error", "Failed to open video!");
        isVideoLoaded = false;
        return;
    }

    isVideoLoaded = true;
    QMessageBox::information(this, "Success", "Video loaded successfully!");
}

// Slot: Start video object detection
void MainWindow::onStartDetection()
{
    //if (!isVideoLoaded) {
        //QMessageBox::warning(this, "Error", "No video loaded!");
        //return;
    //}

    image_processing imgpro;
    cv::Mat frame, prevFrame;
    while (videoCapture.read(frame)) { // Read each frame
        if (frame.empty()) {
            break;
        }

        if (!prevFrame.empty()) {
            imgpro.processOpticalFlow(prevFrame, frame, 100);

        }

        displayFrame(frame);

        // Update prevFrame with the current frame
        prevFrame = frame.clone();

        // Wait for a short delay to simulate video playback speed
        cv::waitKey(30); // Approx. 30 FPS

    }

    QMessageBox::information(this, "Info", "Video processing complete!");

}

// Helper: Display a video frame in QLabel
void MainWindow::displayFrame(const cv::Mat &frame)
{
    // Convert BGR to RGB (Qt expects RGB format)
    cv::Mat rgbFrame;
    cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);

    // Convert OpenCV Mat to QImage
    QImage qImage(rgbFrame.data, rgbFrame.cols, rgbFrame.rows, rgbFrame.step, QImage::Format_RGB888);

    // Scale the image to fit the QLabel while maintaining aspect ratio
    QPixmap pixmap = QPixmap::fromImage(qImage).scaled(ui->lblVideoDisplay->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);

    ui->lblVideoDisplay->setPixmap(pixmap);

    // Update the UI to display the frame immediately
    QApplication::processEvents();
}


// /////////////////////////////////// camera


// Slot: Connect to the camera
void MainWindow::onConnectCamera()
{
    if (isCameraActive) {
        QMessageBox::warning(this, "Error", "Camera is already active!");
        return;
    }

    if (videoCapture.isOpened()) {
        videoCapture.release();
    }

    videoCapture.open(0); // Use camera index 0
    if (!videoCapture.isOpened()) {
        QMessageBox::warning(this, "Error", "Failed to connect to the camera!");
        return;
    }

    isCameraActive = true;
    //QMessageBox::information(this, "Success", "Camera connected successfully!");

    cv::Mat frame;
    while (isCameraActive) {
        videoCapture >> frame;
        if (frame.empty()) {
            continue;
        }

        displayFrame(frame);

        cv::waitKey(30); // Allow ~30 FPS
    }
}
// Slot: Stop the camera feed
void MainWindow::onStopCamera()
{
    if (!isCameraActive) {
        QMessageBox::warning(this, "Error", "Camera is not active!");
        return;
    }

    isCameraActive = false;
    videoCapture.release();
    //QMessageBox::information(this, "Success", "Camera feed stopped!");
}

void MainWindow::onObjectDetection()
{
    if (!isCameraActive){
        QMessageBox::warning(this, "Error", "Camera is not Active");
        return;
    }

    image_processing imgpro;
    cv::Mat frame;
    while (videoCapture.read(frame)) { // Read each frame
        if (frame.empty()) {
            break;
        }

        imgpro.qrCodeDetection(frame);

        displayFrame(frame);

        // Wait for a short delay to simulate video playback speed
        cv::waitKey(30); // Approx. 30 FPS

    }
}



