%ModuleCode
#include "Range.h"
%End


class Range /Supertype=sip.simplewrapper/
{

%TypeHeaderCode
#include <cmath>
#include <string>
#include <QSize>
#include "Range.h"
%End


%ConvertToTypeCode
    // See if we are just being asked to check the type of the Python
    // object.
    if (!sipIsErr)
    {
        if(sipPy==Py_None)
            return 1;

        if(sipCanConvertToType(sipPy, sipType_Range, SIP_NO_CONVERTORS))
            return 1;

        if(sipCanConvertToType(sipPy, sipType_QPoint, SIP_NOT_NONE))
            return 1;

        if(sipCanConvertToType(sipPy, sipType_QPointF, SIP_NOT_NONE))
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
            return 1;
        }

        // The type is not valid.
        return 0;
    }

    if (sipCanConvertToType(sipPy, sipType_Range, SIP_NO_CONVERTORS))
    {
        *sipCppPtr = reinterpret_cast<Range*>(sipConvertToType(sipPy, sipType_Range, sipTransferObj, SIP_NO_CONVERTORS, 0, sipIsErr));

        return 0;
    }


    double x = 0.0;
    double y = 0.0;
    int state;

    if(sipPy==Py_None)
    {
        x = Range::NO_LIMIT;
        y = Range::NO_LIMIT;
    }
    if(sipCanConvertToType(sipPy, sipType_QPoint, SIP_NOT_NONE))
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
    else if(PyList_Check(sipPy))
    {
        x = pyAsDouble(PyList_GetItem(sipPy, 0));
        y = pyAsDouble(PyList_GetItem(sipPy, 1));
    }
    else if(PyTuple_Check(sipPy))
    {
        x = pyAsDouble(PyTuple_GetItem(sipPy, 0));
        y = pyAsDouble(PyTuple_GetItem(sipPy, 1));
    }
    else if(PyObject_HasAttrString(sipPy, "__getitem__")!=0)
    {
        PyObject* key = PyInt_FromLong(0);
        PyObject* value = PyObject_GetItem(sipPy, key);
        x = pyAsDouble(value);
        Py_XDECREF(key);
        Py_XDECREF(value);

        key = PyInt_FromLong(1);
        value = PyObject_GetItem(sipPy, key);
        y = pyAsDouble(value);
        Py_XDECREF(key);
        Py_XDECREF(value);
    }

    *sipCppPtr = new Range(x, y);

    // The type is valid.
    return sipGetState(sipTransferObj);
%End


public:
    Range();

    Range(const double minVal, const double maxVal);

    Range(const Range& other);

    Range(SIP_PYOBJECT) throw(std::runtime_error) /HoldGIL/;
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

            sipCpp = new Range(x, y);
        }
        catch(std::runtime_error& e)
        {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            sipCpp = NULL;
        }
    %End

    ~Range();

    double min() const;
    double max() const;

    void setMin(const double minVal);
    void setMax(const double maxVal);

    bool hasMin() const;
    bool hasMax() const;

    bool isValid() const;
    bool isNull() const;


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

    SIP_PYOBJECT __repr__() const /DocType="str"/;
    %MethodCode
        std::string s = "QtNativeUtils.Range(";
        s += std::to_string(sipCpp->min());
        s += ",";
        s += std::to_string(sipCpp->max());
        s += ")";

        #if PY_MAJOR_VERSION >= 3
            sipRes = PyUnicode_FromString(s.c_str());
        #else
            sipRes = PyString_FromString(s.c_str());
        #endif

    %End
};

bool operator==(const Range& r1, const Range& r2);
bool operator!=(const Range& r1, const Range& r2);
