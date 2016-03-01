
%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip


class Point: public QPointF
{

%TypeHeaderCode
#include <cmath>
#include "Point.h"
%End


%ConvertToTypeCode
    // See if we are just being asked to check the type of the Python
    // object.
    if (!sipIsErr)
    {
        if(PyInt_Check(sipPy) || PyFloat_Check(sipPy))
            return 1;

        if(sipCanConvertToType(sipPy, sipType_QPoint, SIP_NOT_NONE))
            return 1;

        if(sipCanConvertToType(sipPy, sipType_QPointF, SIP_NOT_NONE))
            return 1;

        if(sipCanConvertToType(sipPy, sipType_QSizeF, SIP_NOT_NONE))
            return 1;

        if(PyList_Check(sipPy))
        {
            int listIsOk = 1;
            for (int i = 0; i < 2; ++i)
            {
                PyObject* val = PyList_GetItem(sipPy, i);
                if (!PyInt_Check(val) && !PyFloat_Check(val))
                    listIsOk = 0;
            }
            return listIsOk;
        }

        if(PyTuple_Check(sipPy))
        {
            int tupleIsOk = 1;
            for (int i = 0; i < 2; ++i)
            {
                PyObject* val = PyTuple_GetItem(sipPy, i);
                if (!PyInt_Check(val) && !PyFloat_Check(val))
                    tupleIsOk = 0;
            }
            return tupleIsOk;
        }

        if(PyObject_HasAttrString(sipPy, "__getitem__")!=0)
        {
            PyObject* key = PyInt_FromLong(0);
            PyObject* value = PyObject_GetItem(sipPy, key);
            double x = PyFloat_AsDouble(value);
            Py_XDECREF(key);
            Py_XDECREF(value);
            if(x==-1.0 && PyErr_Occurred())
                return 0;

            key = PyInt_FromLong(1);
            value = PyObject_GetItem(sipPy, key);
            double y = PyFloat_AsDouble(value);
            Py_XDECREF(key);
            Py_XDECREF(value);
            if(y==-1.0 && PyErr_Occurred())
                return 0;
        }

        // The type is not valid.
        return 0;
    }

    double x = 0.0;
    double y = 0.0;
    int state;

    // Convert Python object to Point
    if(PyInt_Check(sipPy) || PyFloat_Check(sipPy))
    {
        x = PyFloat_AsDouble(sipPy);
        y = x;
    }
    else if(sipCanConvertToType(sipPy, sipType_QPoint, SIP_NOT_NONE))
    {
        QPoint* point = reinterpret_cast<QPoint*>(sipConvertToType(sipPy, sipType_QPoint, 0, 0, &state, sipIsErr));

        if(*sipIsErr)
        {
            sipReleaseType(point, sipType_QPoint, state);
            return 0;
        }

        x = point->x();
        y = point->y();

        sipReleaseType(point, sipType_QPoint, state);
    }
    else if(sipCanConvertToType(sipPy, sipType_QPointF, SIP_NOT_NONE))
    {
        QPointF* point = reinterpret_cast<QPointF*>(sipConvertToType(sipPy, sipType_QPointF, 0, 0, &state, sipIsErr));

        if(*sipIsErr)
        {
            sipReleaseType(point, sipType_QPointF, state);
            return 0;
        }

        x = point->x();
        y = point->y();

        sipReleaseType(point, sipType_QPointF, state);
    }
    else if(sipCanConvertToType(sipPy, sipType_QSizeF, SIP_NOT_NONE))
    {
        QSizeF* sf = reinterpret_cast<QSizeF*>(sipConvertToType(sipPy, sipType_QSizeF, 0, 0, &state, sipIsErr));

        if(*sipIsErr)
        {
            sipReleaseType(sf, sipType_QSizeF, state);
            return 0;
        }

        x = sf->width();
        y = sf->height();

        sipReleaseType(sf, sipType_QSizeF, state);
    }
    else if(PyList_Check(sipPy))
    {
        x = PyFloat_AsDouble(PyList_GetItem(sipPy, 0));
        y = PyFloat_AsDouble(PyList_GetItem(sipPy, 1));
    }
    else if(PyTuple_Check(sipPy))
    {
        x = PyFloat_AsDouble(PyTuple_GetItem(sipPy, 0));
        y = PyFloat_AsDouble(PyTuple_GetItem(sipPy, 1));
    }
    else if(PyObject_HasAttrString(sipPy, "__getitem__")!=0)
    {
        PyObject* key = PyInt_FromLong(0);
        PyObject* value = PyObject_GetItem(sipPy, key);
        x = PyFloat_AsDouble(value);
        Py_XDECREF(key);
        Py_XDECREF(value);
        if(x==-1.0 && PyErr_Occurred())
            return 0;

        key = PyInt_FromLong(1);
        value = PyObject_GetItem(sipPy, key);
        y = PyFloat_AsDouble(value);
        Py_XDECREF(key);
        Py_XDECREF(value);
        if(y==-1.0 && PyErr_Occurred())
            return 0;
    }

    *sipCppPtr = new Point(x, y);

    // The type is valid.
    return sipGetState(sipTransferObj);
%End


public:
    Point() /HoldGIL/;

    Point(const double xy) /HoldGIL/;

    Point(const double x, const double y) /HoldGIL/;

    Point(const QPointF& p) /HoldGIL/;

    Point(const QPoint& p) /HoldGIL/;

    Point(const QSizeF& s) /HoldGIL/;

    Point(SIP_PYOBJECT) throw(std::runtime_error) /HoldGIL/;
    %MethodCode
        try
        {
            PyObject* iterable = a0;
            if(PyObject_HasAttrString(iterable, "__getitem__")==0)
                throw std::runtime_error("Object is not iterable");

            PyObject* key = PyInt_FromLong(0);
            PyObject* value = PyObject_GetItem(iterable, key);
            double x = PyFloat_AsDouble(value);
            Py_XDECREF(key);
            Py_XDECREF(value);
            if(x==-1 && PyErr_Occurred())
                throw std::runtime_error("iterable[0] is not a float value");

            key = PyInt_FromLong(1);
            value = PyObject_GetItem(iterable, key);
            double y = PyFloat_AsDouble(value);
            Py_XDECREF(key);
            Py_XDECREF(value);
            if(x==-1 && PyErr_Occurred())
                throw std::runtime_error("iterable[1] is not a float value");

            sipCpp = new Point(x, y);
        }
        catch(std::runtime_error& e)
        {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            sipCpp = NULL;
        }
    %End

    void __setitem__(const int i, const double value) /HoldGIL/;
    %MethodCode
        try {
            (*sipCpp)[a0] = a1;
        } catch(...) {
            PyErr_SetString(PyExc_IndexError, "Index out of range");
            sipIsErr = 1;
        }
    %End

    double __getitem__(const int i) const /HoldGIL/;
    %MethodCode
        try {
            sipRes = (*sipCpp)[a0];
        } catch(...) {
            PyErr_SetString(PyExc_IndexError, "Index out of range");
            sipIsErr = 1;
        }
    %End

    Point __getitem__(SIP_PYSLICE) const /HoldGIL/;
    %MethodCode
        try {
            sipRes = new Point(*sipCpp);
        } catch(...) {
            PyErr_SetString(PyExc_IndexError, "Index out of range");
            sipIsErr = 1;
        }
    %End

    int __len__() const /HoldGIL/;
    %MethodCode
        sipRes = 2;
    %End

    Point __radd__(Point p) const /HoldGIL/;
    %MethodCode
        sipRes = new Point(a0->x() + sipCpp->x(), a0->y() + sipCpp->y());
    %End

    Point __add__(Point p) const /HoldGIL/;
    %MethodCode
        sipRes = new Point(a0->x() + a1->x(), a0->y() + a1->y());
    %End

    Point __rsub__(Point p) const /HoldGIL/;
    %MethodCode
        sipRes = new Point(a0->x() - sipCpp->x(), a0->y() - sipCpp->y());
    %End

    Point __sub__(Point p) const /HoldGIL/;
    %MethodCode
        sipRes = new Point(a0->x() - a1->x(), a0->y() - a1->y());
    %End

    Point __rmul__(Point p) const /HoldGIL/;
    %MethodCode
        sipRes = new Point(a0->x() * sipCpp->x(), a0->y() * sipCpp->y());
    %End

    Point __mul__(Point p) const /HoldGIL/;
    %MethodCode
        sipRes = new Point(a0->x() * a1->x(), a0->y() * a1->y());
    %End

    Point __rdiv__(Point p) const /HoldGIL/;
    %MethodCode
        sipRes = new Point(a0->x() / sipCpp->x(), a0->y() / sipCpp->y());
    %End

    Point __div__(Point p) const /HoldGIL/;
    %MethodCode
        sipRes = new Point(a0->x() / a1->x(), a0->y() / a1->y());
    %End

    Point __rtruediv__(Point p) const /HoldGIL/;
    %MethodCode
        sipRes = new Point(a0->x() / sipCpp->x(), a0->y() / sipCpp->y());
    %End

    Point __truediv__(Point p) const /HoldGIL/;
    %MethodCode
        sipRes = new Point(a0->x() / a1->x(), a0->y() / a1->y());
    %End

    Point __rpow__(Point p) const /HoldGIL/;
    %MethodCode
        sipRes = new Point(std::pow(sipCpp->x(), a0->x()), std::pow(sipCpp->x(), a0->x()));
    %End

    Point __pow__(Point p) const /HoldGIL/;
    %MethodCode
        sipRes = new Point(std::pow(sipCpp->x(), a0->x()), std::pow(sipCpp->x(), a0->x()));
    %End

    double length() const /HoldGIL/;
    Point norm() const /HoldGIL/;
    double angle(const Point& a) const /HoldGIL/;
    double dot(const Point& a) const /HoldGIL/;
    double cross(const Point& a) const /HoldGIL/;
    Point proj(const Point& b) const /HoldGIL/;
    double min() const /HoldGIL/;
    double max() const /HoldGIL/;
    Point copy() const /HoldGIL/;
    Point toQPoint() const /HoldGIL/;
};

