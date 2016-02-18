#-------------------------------------------------
#
# Project created by QtCreator 2016-02-16T20:55:07
#
#-------------------------------------------------

TARGET = QtNativeUtils
TEMPLATE = lib
CONFIG += static

QMAKE_CXXFLAGS += -std=c++11

INCLUDEPATH += src

SOURCES += src/QtNativeUtils.cpp

HEADERS += src/QtNativeUtils.h

DISTFILES += sip/Exceptions.sip \
             sip/QtNativeUtils.sip

#NUMPY_INCLUDE = /usr/local/lib/python2.7/dist-packages/numpy/core/include
NUMPY_INCLUDE = /usr/lib/python2.7/dist-packages/numpy/core/include
PYTHON_INCLUDE = /usr/include/python2.7

INCLUDEPATH += $$NUMPY_INCLUDE
INCLUDEPATH += $$PYTHON_INCLUDE


