QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++14

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    image_processing.cpp \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    image_processing.h \
    mainwindow.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target


win32: LIBS += -L$$PWD/../../../../opencv-4.5.4/build/install/x64/mingw/lib/ -llibopencv_world454.dll

INCLUDEPATH += $$PWD/../../../../opencv-4.5.4/build/install/include
DEPENDPATH += $$PWD/../../../../opencv-4.5.4/build/install/include


