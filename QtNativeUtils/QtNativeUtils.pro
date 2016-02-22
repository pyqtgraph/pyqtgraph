#-------------------------------------------------
#
# Project created by QtCreator 2016-02-16T20:55:07
#
#-------------------------------------------------

TARGET = QtNativeUtils
TEMPLATE = lib
CONFIG += static

QMAKE_CXXFLAGS += -std=c++11 -Wno-deprecated  -Wno-unused-parameter -Wno-unused-function

INCLUDEPATH += src \
               src/mouseevents

SOURCES += src/QtNativeUtils.cpp \
    src/mouseevents/MouseClickEvent.cpp \
    src/mouseevents/HoverEvent.cpp \
    src/mouseevents/MouseDragEvent.cpp \
    src/mouseevents/MouseEvent.cpp \
    src/Point.cpp \
    src/internal/Numpy.cpp \
    src/QGraphicsObject2.cpp \
    src/QGraphicsWidget2.cpp \
    src/QGraphicsScene2.cpp \
    src/ExtendedItem.cpp \
    src/ViewBoxBase.cpp


HEADERS += src/QtNativeUtils.h \
    src/mouseevents/MouseEvent.h \
    src/mouseevents/HoverEvent.h \
    src/mouseevents/MouseDragEvent.h \
    src/mouseevents/MouseClickEvent.h \
    src/Point.h \
    src/internal/Numpy.h \
    src/QGraphicsObject2.h \
    src/QGraphicsWidget2.h \
    src/QGraphicsScene2.h \
    src/internal/point_utils.h \
    src/ExtendedItem.h \
    src/Interfaces.h \
    src/ViewBoxBase.h

#NUMPY_INCLUDE = /usr/local/lib/python2.7/dist-packages/numpy/core/include
NUMPY_INCLUDE = /usr/lib/python2.7/dist-packages/numpy/core/include
PYTHON_INCLUDE = /usr/include/python2.7

LIBS += python2.7

INCLUDEPATH += $$NUMPY_INCLUDE
INCLUDEPATH += $$PYTHON_INCLUDE

OTHER_FILES += sip/Exceptions.sip \
    sip/QtNativeUtils.sip \
    sip/MouseEvent.sip \
    sip/MouseClickEvent.sip \
    sip/HoverEvent.sip \
    sip/MouseDragEvent.sip \
    sip/Point.sip \
    sip/QGraphicsObject2.sip \
    sip/QGraphicsWidget2.sip \
    sip/Interfaces.sip \
    sip/QGraphicsScene2.sip \
    sip/ViewBoxBase.sip


