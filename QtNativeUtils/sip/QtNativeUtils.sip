
%Module(name=QtNativeUtils, keyword_arguments="Optional")


%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip


%PostInitialisationCode
init_package();
%End

%ModuleCode
#include "QtNativeUtils.h"
%End

%ModuleHeaderCode
#include <Python.h>
#include <limits>

static double pyAsDouble(PyObject* obj)
{
    if(obj==Py_None)
        return std::numeric_limits<double>::quiet_NaN();
    if(PyFloat_Check(obj))
        return PyFloat_AsDouble(obj);
    if(PyInt_Check(obj))
        return (double)PyInt_AsLong(obj);
    PyObject* num = PyNumber_Float(obj);
    if(num!=nullptr)
    {
        double val = PyFloat_AsDouble(num);
        Py_DecRef(num);
        return val;
    }

    return std::numeric_limits<double>::quiet_NaN();
}

%End


%Include Exceptions.sip

%Include ExtendedItem.sip
%Include Interfaces.sip

%Include Point.sip
%Include Range.sip
%Include MouseEvent.sip
%Include MouseClickEvent.sip
%Include HoverEvent.sip
%Include MouseDragEvent.sip

%Include GraphicsObject.sip
%Include GraphicsWidget.sip

%Include ItemGroup.sip
%Include ChildGroup.sip

%Include QGraphicsScene2.sip
%Include ViewBoxBase.sip
%Include PlotItemBase.sip
%Include GraphicsViewBase.sip
