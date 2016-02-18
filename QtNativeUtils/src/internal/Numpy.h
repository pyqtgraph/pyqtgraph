#ifndef NUMPY_H
#define NUMPY_H

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdexcept>

template<size_t _NDIM>
class NumpyArray
{
public:
    NumpyArray(const NPY_TYPES dType):
        mDtype(dType),
        mNdArray(nullptr),
        mData(nullptr)
    {
        for(int i=0; i<_NDIM; ++i)
            mShape[i] = 0;
    }

    NumpyArray(PyObject* ndarray, const NPY_TYPES dType) throw(std::runtime_error):
        mDtype(dType),
        mNdArray(nullptr),
        mData(nullptr)
    {
        acquire(ndarray);
    }

    ~NumpyArray()
    {
        release();
    }

    void operator=(PyObject* ndarray) throw(std::runtime_error)
    {
        acquire(ndarray);
    }

    void release()
    {
        Py_XDECREF(mNdArray);
        mNdArray = nullptr;
        mData = nullptr;
        for(int i=0; i<_NDIM; ++i)
            mShape[i] = 0;
    }

    size_t ndim() const
    {
        return _NDIM;
    }

    size_t shape(const size_t i=0) const throw(std::runtime_error)
    {
        if(i>=_NDIM)
             throw std::runtime_error("Index error in ndarray.shape");
        return mShape[i];
    }

    template<typename _Tp>
    _Tp* data()
    {
        return static_cast<_Tp*>(mData);
    }

    template<typename _Tp>
    const _Tp* data() const
    {
        return static_cast<const _Tp*>(mData);
    }

protected:

    void acquire(PyObject* ndarray) throw(std::runtime_error)
    {
        release(); // release the previous data

        if(PyArray_IS_C_CONTIGUOUS((PyArrayObject*)ndarray)==0)
            throw std::runtime_error("Numpy array must be contiguous");

        if(PyArray_TYPE((PyArrayObject*)ndarray)!=mDtype)
            throw std::runtime_error("Numpy array dtype is not what expected");

        if(PyArray_NDIM((PyArrayObject*)ndarray)!=_NDIM)
            throw std::runtime_error("Numpy array ndim is not what expected");

        for(int i=0; i<_NDIM; ++i)
            mShape[i] = PyArray_DIM((PyArrayObject*)ndarray, i);

        mNdArray = ndarray;
        Py_IncRef(mNdArray);
    }


protected:
    NPY_TYPES mDtype;
    PyObject* mNdArray;
    size_t mShape[_NDIM];
    void* mData;
};

#endif // NUMPY_H
