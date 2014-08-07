# -*- coding: utf-8 -*-
"""
Importing this module installs support for 16-bit images in PIL.
This works by patching objects in the PIL namespace; no files are
modified.
"""

from PIL import Image

if Image.VERSION == '1.1.7':
    Image._MODE_CONV["I;16"] = ('%su2' % Image._ENDIAN, None)
    Image._fromarray_typemap[((1, 1), "<u2")] = ("I", "I;16")
if Image.VERSION == '1.1.6':
    Image._MODE_CONV["I;16"] = ('%su2' % Image._ENDIAN, None)
    ## just a copy of fromarray() from Image.py with I;16 added in
    def fromarray(obj, mode=None):
        arr = obj.__array_interface__
        shape = arr['shape']
        ndim = len(shape)
        try:
            strides = arr['strides']
        except KeyError:
            strides = None
        if mode is None:
            typestr = arr['typestr']
            if not (typestr[0] == '|' or typestr[0] == Image._ENDIAN or
                    typestr[1:] not in ['u1', 'b1', 'i4', 'f4']):
                raise TypeError("cannot handle data-type")
            if typestr[0] == Image._ENDIAN:
                typestr = typestr[1:3]
            else:
                typestr = typestr[:2]
            if typestr == 'i4':
                mode = 'I'
            if typestr == 'u2':
                mode = 'I;16'
            elif typestr == 'f4':
                mode = 'F'
            elif typestr == 'b1':
                mode = '1'
            elif ndim == 2:
                mode = 'L'
            elif ndim == 3:
                mode = 'RGB'
            elif ndim == 4:
                mode = 'RGBA'
            else:
                raise TypeError("Do not understand data.")
        ndmax = 4
        bad_dims=0
        if mode in ['1','L','I','P','F']:
            ndmax = 2
        elif mode == 'RGB':
            ndmax = 3
        if ndim > ndmax:
            raise ValueError("Too many dimensions.")

        size = shape[:2][::-1]
        if strides is not None:
            obj = obj.tostring()

        return frombuffer(mode, size, obj, "raw", mode, 0, 1)
        
    Image.fromarray=fromarray