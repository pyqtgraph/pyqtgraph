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
    src/QGraphicsScene2.cpp \
    src/ExtendedItem.cpp \
    src/ViewBoxBase.cpp \
    src/PlotItemBase.cpp \
    src/GraphicsViewBase.cpp \
    src/GraphicsObject.cpp \
    src/GraphicsWidget.cpp \
    src/graphicsitems/ItemGroup.cpp \
    src/graphicsitems/ChildGroup.cpp \
    src/Range.cpp \
    src/graphicsitems/UIGraphicsItem.cpp


HEADERS += src/QtNativeUtils.h \
    src/mouseevents/MouseEvent.h \
    src/mouseevents/HoverEvent.h \
    src/mouseevents/MouseDragEvent.h \
    src/mouseevents/MouseClickEvent.h \
    src/Point.h \
    src/internal/Numpy.h \
    src/QGraphicsScene2.h \
    src/internal/point_utils.h \
    src/ExtendedItem.h \
    src/Interfaces.h \
    src/ViewBoxBase.h \
    src/PlotItemBase.h \
    src/ItemDefines.h \
    src/GraphicsViewBase.h \
    src/GraphicsObject.h \
    src/GraphicsWidget.h \
    src/graphicsitems/ItemGroup.h \
    src/graphicsitems/ChildGroup.h \
    src/Range.h \
    src/graphicsitems/UIGraphicsItem.h


NUMPY_INCLUDE_1 = /usr/local/lib/python2.7/dist-packages/numpy/core/include
NUMPY_INCLUDE_2 = /usr/lib/python2.7/dist-packages/numpy/core/include
PYTHON_INCLUDE = /usr/include/python2.7

LIBS += python2.7

INCLUDEPATH += $$NUMPY_INCLUDE_1 $$NUMPY_INCLUDE_2
INCLUDEPATH += $$PYTHON_INCLUDE

OTHER_FILES += sip/Exceptions.sip \
    sip/GraphicsObject.sip \
    sip/QtNativeUtils.sip \
    sip/MouseEvent.sip \
    sip/MouseClickEvent.sip \
    sip/HoverEvent.sip \
    sip/MouseDragEvent.sip \
    sip/Point.sip \
    sip/Interfaces.sip \
    sip/QGraphicsScene2.sip \
    sip/ViewBoxBase.sip \
    sip/PlotItemBase.sip \
    sip/GraphicsViewBase.sip \
    sip/ExtendedItem.sip \
    sip/GraphicsWidget.sip \
    sip/ChildGroup.sip \
    sip/Range.sip \
    sip/UIGraphicsItem.sip

DISTFILES += \
    sip/ItemGroup.sip


