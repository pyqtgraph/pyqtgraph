import numpy

from .Qt import QtGui
from . import functions
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions


def _apply_lut_for_uint16_mono(xp, image, lut):
    # Note: compared to makeARGB(), we have already clipped the data to range
    augmented_alpha = False

    # if lut is 1d, then lut[image] is fastest
    # if lut is 2d, then lut.take(image, axis=0) is faster than lut[image]

    if not image.flags.c_contiguous:
        image = lut.take(image, axis=0)

        # if lut had dimensions (N, 1), then our resultant image would
        # have dimensions (h, w, 1)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]

        return image, augmented_alpha

    # if we are contiguous, we can take a faster codepath where we
    # ensure that the lut is 1d

    lut, augmented_alpha = _convert_2dlut_to_1dlut(xp, lut)

    fn_numba = getNumbaFunctions()
    if xp == numpy and fn_numba is not None:
        image = fn_numba.numba_take(lut, image)
    else:
        image = lut[image]

    if image.dtype == xp.uint32:
        image = image[..., xp.newaxis].view(xp.uint8)

    return image, augmented_alpha


def _convert_2dlut_to_1dlut(xp, lut):
    # converts:
    #   - uint8 (N, 1) to uint8 (N,)
    #   - uint8 (N, 3) or (N, 4) to uint32 (N,)
    # this allows faster lookup as 1d lookup is faster
    augmented_alpha = False

    if lut.ndim == 1:
        return lut, augmented_alpha

    if lut.shape[1] == 3:  # rgb
        # convert rgb lut to rgba so that it is 32-bits
        lut = xp.column_stack([lut, xp.full(lut.shape[0], 255, dtype=xp.uint8)])
        augmented_alpha = True
    if lut.shape[1] == 4:  # rgba
        lut = lut.view(xp.uint32)
    lut = lut.ravel()

    return lut, augmented_alpha


def _rescale_float_mono(xp, image, levels, lut):
    augmented_alpha = False

    # Decide on maximum scaled value
    if lut is not None:
        scale = lut.shape[0]
        num_colors = lut.shape[0]
    else:
        scale = 255.0
        num_colors = 256
    dtype = xp.min_scalar_type(num_colors - 1)

    minVal, maxVal = levels
    rng = maxVal - minVal
    rng = 1 if rng == 0 else rng

    fn_numba = getNumbaFunctions()
    if (
        xp == numpy
        and image.flags.c_contiguous
        and dtype == xp.uint16
        and fn_numba is not None
    ):
        lut, augmented_alpha = _convert_2dlut_to_1dlut(xp, lut)
        image = fn_numba.rescale_and_lookup1d(image, scale / rng, minVal, lut)
        if image.dtype == xp.uint32:
            image = image[..., xp.newaxis].view(xp.uint8)
        return image, None, None, augmented_alpha
    else:
        image = functions.rescaleData(
            image, scale / rng, offset=minVal, dtype=dtype, clip=(0, num_colors - 1)
        )

        levels = None

        if image.dtype == xp.uint16 and image.ndim == 2:
            image, augmented_alpha = _apply_lut_for_uint16_mono(xp, image, lut)
            lut = None

        # image is now of type uint8
        return image, levels, lut, augmented_alpha


def _try_combine_lut(xp, image, levels, lut):
    augmented_alpha = False

    if (
        image.dtype == xp.uint16
        and levels is None
        and image.ndim == 3
        and image.shape[2] == 3
    ):
        # uint16 rgb can't be directly displayed, so make it
        # pass through effective lut processing
        levels = [0, 65535]

    if levels is None and lut is None:
        # nothing to combine
        return image, levels, lut, augmented_alpha

    # distinguish between lut for levels and colors
    levels_lut = None
    colors_lut = lut

    eflsize = 2 ** (image.itemsize * 8)
    if levels is None:
        info = xp.iinfo(image.dtype)
        minlev, maxlev = info.min, info.max
    else:
        minlev, maxlev = levels
    levdiff = maxlev - minlev
    levdiff = 1 if levdiff == 0 else levdiff  # don't allow division by 0

    if colors_lut is None:
        if image.dtype == xp.ubyte and image.ndim == 2:
            # uint8 mono image
            ind = xp.arange(eflsize)
            levels_lut = functions.rescaleData(
                ind, scale=255.0 / levdiff, offset=minlev, dtype=xp.ubyte
            )
            # image data is not scaled. instead, levels_lut is used
            # as (grayscale) Indexed8 ColorTable to get the same effect.
            # due to the small size of the input to rescaleData(), we
            # do not bother caching the result
            return image, None, levels_lut, augmented_alpha
        else:
            # uint16 mono, uint8 rgb, uint16 rgb
            # rescale image data by computation instead of by memory lookup
            image = functions.rescaleData(
                image, scale=255.0 / levdiff, offset=minlev, dtype=xp.ubyte
            )
            return image, None, colors_lut, augmented_alpha
    else:
        num_colors = colors_lut.shape[0]
        effscale = num_colors / levdiff
        lutdtype = xp.min_scalar_type(num_colors - 1)

        if image.dtype == xp.ubyte or lutdtype != xp.ubyte:
            # combine if either:
            #   1) uint8 mono image
            #   2) colors_lut has more entries than will fit within 8-bits
            ind = xp.arange(eflsize)
            levels_lut = functions.rescaleData(
                ind,
                scale=effscale,
                offset=minlev,
                dtype=lutdtype,
                clip=(0, num_colors - 1),
            )
            efflut = colors_lut[levels_lut]

            # apply the effective lut early for the following types:
            if image.dtype == xp.uint16 and image.ndim == 2:
                image, augmented_alpha = _apply_lut_for_uint16_mono(xp, image, efflut)
                efflut = None
            return image, None, efflut, augmented_alpha
        else:
            # uint16 image with colors_lut <= 256 entries
            # don't combine, we will use QImage ColorTable
            image = functions.rescaleData(
                image,
                scale=effscale,
                offset=minlev,
                dtype=lutdtype,
                clip=(0, num_colors - 1),
            )
            return image, None, colors_lut, augmented_alpha


def try_make_qimage(image, *, levels, lut):
    """
    Internal function to make an QImage from an ndarray without going
    through the full generality of makeARGB().
    Only certain combinations of input arguments are supported.
    """

    # this function assumes that image has no nans.
    # checking for nans is an expensive operation; it is expected that
    # the caller would want to cache the result rather than have this
    # function check for nans unconditionally.

    cp = getCupy()
    xp = cp.get_array_module(image) if cp else numpy

    # float images always need levels
    if image.dtype.kind == "f" and levels is None:
        return None

    # can't handle multi-channel levels
    if levels is not None:
        levels = xp.asarray(levels)
        if levels.ndim != 1:
            return None

    if lut is not None and lut.dtype != xp.uint8:
        raise ValueError("lut dtype must be uint8")

    augmented_alpha = False

    if image.dtype.kind == "f":
        image, levels, lut, augmented_alpha = _rescale_float_mono(
            xp, image, levels, lut
        )
        # on return, we will have an uint8 image with levels None.
        # lut if not None will have <= 256 entries

    # if the image data is a small int, then we can combine levels + lut
    # into a single lut for better performance
    elif image.dtype in (xp.ubyte, xp.uint16):
        image, levels, lut, augmented_alpha = _try_combine_lut(xp, image, levels, lut)

    ubyte_nolvl = image.dtype == xp.ubyte and levels is None
    is_passthru8 = ubyte_nolvl and lut is None
    is_indexed8 = (
        ubyte_nolvl and image.ndim == 2 and lut is not None and lut.shape[0] <= 256
    )
    is_passthru16 = image.dtype == xp.uint16 and levels is None and lut is None
    can_grayscale16 = (
        is_passthru16
        and image.ndim == 2
        and hasattr(QtGui.QImage.Format, "Format_Grayscale16")
    )
    is_rgba64 = is_passthru16 and image.ndim == 3 and image.shape[2] == 4

    # bypass makeARGB for supported combinations
    supported = is_passthru8 or is_indexed8 or can_grayscale16 or is_rgba64
    if not supported:
        return None

    if xp == cp:
        image = image.get()

    # worthwhile supporting non-contiguous arrays
    image = numpy.ascontiguousarray(image)

    fmt = None
    ctbl = None
    if is_passthru8:
        # both levels and lut are None
        # these images are suitable for display directly
        if image.ndim == 2:
            fmt = QtGui.QImage.Format.Format_Grayscale8
        elif image.shape[2] == 3:
            fmt = QtGui.QImage.Format.Format_RGB888
        elif image.shape[2] == 4:
            if augmented_alpha:
                fmt = QtGui.QImage.Format.Format_RGBX8888
            else:
                fmt = QtGui.QImage.Format.Format_RGBA8888
    elif is_indexed8:
        # levels and/or lut --> lut-only
        fmt = QtGui.QImage.Format.Format_Indexed8
        if lut.ndim == 1 or lut.shape[1] == 1:
            ctbl = [QtGui.qRgb(x, x, x) for x in lut.ravel().tolist()]
        elif lut.shape[1] == 3:
            ctbl = [QtGui.qRgb(*rgb) for rgb in lut.tolist()]
        elif lut.shape[1] == 4:
            ctbl = [QtGui.qRgba(*rgba) for rgba in lut.tolist()]
    elif can_grayscale16:
        # single channel uint16
        # both levels and lut are None
        fmt = QtGui.QImage.Format.Format_Grayscale16
    elif is_rgba64:
        # uint16 rgba
        # both levels and lut are None
        fmt = QtGui.QImage.Format.Format_RGBA64  # endian-independent
    if fmt is None:
        raise ValueError("unsupported image type")
    qimage = functions.ndarray_to_qimage(image, fmt)
    if ctbl is not None:
        qimage.setColorTable(ctbl)
    return qimage
