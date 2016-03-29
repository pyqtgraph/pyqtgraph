#include "QtNativeUtils.h"


#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "Range.h"


void init_package()
{
    //int id = qRegisterMetaType<Range>("Range");
    Range::registerMetatype();
    std::cout<<"Registering Range metatype: "<<Range::metatypeId<<std::endl;

    int ret = _import_array();
    std::cout<<"Importing numpy: "<<ret<<std::endl;
}

