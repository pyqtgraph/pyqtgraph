import numpy

from .Qt import QtGui
from . import functions
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions


def _apply_lut_for_uint(xp, image, lut):
    # Note: compared to makeARGB(), we have already clipped the data to range

    # if lut is 1d, then lut[image] is fastest
    # if lut is 2d, then lut.take(image, axis=0) is faster than lut[image]
    lut = _convert_2dlut_to_1dlut(xp, lut)

    if xp == numpy and (fn_numba := getNumbaFunctions()) is not None:
        # numba "take" supports only the 1st 2 arguments of np.take,
        # therefore we have to convert the lut to 1d.
        # "take" will output a c contiguous array regardless of its input.
        image = fn_numba.numba_take(lut, image)
    else:
        # advanced indexing is memory order aware.
        # its output can be either C or F contiguous.
        image = lut[image]

    if image.dtype == xp.uint32:
        # "view" requires c contiguous for numpy < 1.23
        image = xp.ascontiguousarray(image)
        image = image[..., xp.newaxis].view(xp.uint8)

    return image


def _convert_lut_to_rgba(xp, lut):
    # converts:
    #   - None to (256, 4)
    #   - uint8 (N,) to uint8 (N, 4)
    #   - uint8 (N, 1) to uint8 (N, 4)
    #   - uint8 (N, 3) to uint8 (N, 4)

    if not (
        lut is None
        or lut.ndim == 1
        or (
            lut.ndim == 2
            and lut.shape[1] in (1, 3, 4)
        )
    ):
        raise ValueError("unsupported lut shape")

    N = lut.shape[0] if lut is not None else 256

    if lut is None:
        lut = xp.arange(N, dtype=xp.uint8)

    # convert (N,) to (N, 1)
    if lut.ndim == 1:
        lut = lut[:, xp.newaxis]

    if lut.shape[1] == 4:
        return lut

    out = xp.full((N, 4), 255, dtype=xp.uint8)
    out[:, 0:3] = lut
    return out


def _convert_2dlut_to_1dlut(xp, lut):
    # converts:
    #   - uint8 (N, 1) to uint8 (N,)
    #   - uint8 (N, 3) or (N, 4) to uint32 (N,)
    # this allows faster lookup as 1d lookup is faster

    if lut.ndim == 1:
        return lut

    if lut.shape[1] == 3:  # rgb
        # convert rgb lut to rgba so that it is 32-bits
        lut = xp.column_stack([lut, xp.full(lut.shape[0], 255, dtype=xp.uint8)])
    if lut.shape[1] == 4:  # rgba
        lut = lut.view(xp.uint32)
    lut = lut.ravel()

    return lut


def _rescale_and_lookup_float(xp, image, levels, lut, *, forceApplyLut):
    # It is usually more performant to _not_ apply the lut and
    # instead use it as an Indexed8 ColorTable. This is only
    # applicable if the lut has <= 256 entries.

    if forceApplyLut and lut is None:
        raise ValueError("forceApplyLut True but lut not provided")

    # Decide on maximum scaled value
    if lut is not None:
        num_colors = lut.shape[0]
        max_scale_value = num_colors
    else:
        num_colors = 256
        max_scale_value = 255.0
    dtype = xp.min_scalar_type(num_colors - 1)

    # note: "dtype == uint16" ==> lut provided ==> mono-channel image
    #       i.e. multi-channel image ==> lut is None ==> dtype == uint8
    #
    #       the library defaults to using 256-entry luts, so
    #       "dtype == uint8" is the common case

    apply_lut = forceApplyLut or dtype == xp.uint16

    minVal, maxVal = levels
    rng = maxVal - minVal
    rng = 1 if rng == 0 else rng
    offset = minVal
    scale = max_scale_value / rng

    if xp == numpy and (fn_numba := getNumbaFunctions()) is not None:
        if apply_lut:
            # this path does rescale and apply lut in one step
            lut = _convert_2dlut_to_1dlut(xp, lut)
            image = fn_numba.rescale_and_lookup(image, scale, offset, lut)
            lut = None
            if image.dtype == xp.uint32:
                # "view" requires c contiguous for numpy < 1.23
                image = xp.ascontiguousarray(image)
                image = image[..., xp.newaxis].view(xp.uint8)
        else:
            image = fn_numba.rescale_and_clip(image, scale, offset, 0, num_colors - 1)
    else:
        image = functions.rescaleData(
            image, scale, offset, dtype=dtype, clip=(0, num_colors - 1)
        )
        if apply_lut:
            image = _apply_lut_for_uint(xp, image, lut)
            lut = None

    # image is now of type uint8
    return image, lut


def _combine_levels_and_lut(xp, image, levels, lut):
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
        return image, lut

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
    offset = minlev

    if colors_lut is None:
        scale = 255.0 / levdiff
        if image.dtype == xp.ubyte and image.ndim == 2:
            # uint8 mono image
            ind = xp.arange(eflsize)
            levels_lut = functions.rescaleData(ind, scale, offset, dtype=xp.ubyte)
            # image data is not scaled. instead, levels_lut is used
            # as (grayscale) Indexed8 ColorTable to get the same effect.
            # due to the small size of the input to rescaleData(), we
            # do not bother caching the result
            return image, levels_lut
        else:
            # uint16 mono, uint8 rgb, uint16 rgb
            # rescale image data by computation instead of by memory lookup
            if xp == numpy and (fn_numba := getNumbaFunctions()) is not None:
                image = fn_numba.rescale_and_clip(image, scale, offset, 0, 255)
            else:
                image = functions.rescaleData(image, scale, offset, dtype=xp.ubyte)
            return image, colors_lut
    else:
        num_colors = colors_lut.shape[0]
        scale = num_colors / levdiff
        lutdtype = xp.min_scalar_type(num_colors - 1)

        if image.dtype == xp.ubyte or lutdtype != xp.ubyte:
            # combine if either:
            #   1) uint8 mono image
            #   2) colors_lut has more entries than will fit within 8-bits
            ind = xp.arange(eflsize)
            levels_lut = functions.rescaleData(
                ind, scale, offset, dtype=lutdtype, clip=(0, num_colors - 1),
            )
            efflut = colors_lut[levels_lut]

            # apply the effective lut early for the following types:
            if image.dtype == xp.uint16 and image.ndim == 2:
                image = _apply_lut_for_uint(xp, image, efflut)
                efflut = None
            return image, efflut
        else:
            # uint16 image with colors_lut <= 256 entries
            # don't combine, we will use QImage ColorTable
            if xp == numpy and (fn_numba := getNumbaFunctions()) is not None:
                image = fn_numba.rescale_and_clip(image, scale, offset, 0, num_colors - 1)
            else:
                image = functions.rescaleData(
                    image, scale, offset, dtype=lutdtype, clip=(0, num_colors - 1),
                )
            return image, colors_lut


def try_make_qimage(image, *, levels, lut, transparentLocations=None):
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

    if levels is not None:
        levels = xp.asarray(levels)

        # can't handle multi-channel levels
        if levels.ndim != 1:
            return None

        # if levels is provided, multi-channel images must be 3 channels only.
        # (because it doesn't make sense to scale a 4th alpha channel.)
        if image.ndim == 3 and image.shape[2] != 3:
            return None

    if lut is not None and lut.dtype != xp.uint8:
        raise ValueError("lut dtype must be uint8")

    alpha_channel_required = (
        (   # image itself has alpha channel
            image.ndim == 3
            and image.shape[2] == 4
        )
        or
        (    # lut has alpha channel
            lut is not None
            and lut.ndim == 2
            and lut.shape[1] == 4
        )
    )

    if image.dtype.kind == "f":
        if image.ndim == 2:
            # mono float images
            if transparentLocations is None:
                image, lut = _rescale_and_lookup_float(
                    xp, image, levels, lut, forceApplyLut=False
                )
                levels = None
                # on return, we will have an uint8 image.
                # lut if not None will have <= 256 entries
            else:
                # this path creates an alpha channel
                lut = _convert_lut_to_rgba(xp, lut)
                alpha_channel_required = True

                image, lut = _rescale_and_lookup_float(
                    xp, image, levels, lut, forceApplyLut=True
                )
                levels = None
                assert lut is None
                image[..., 3][transparentLocations] = 0
        else:
            # RGB float images
            # lut can only be None for RGB images
            image, lut = _rescale_and_lookup_float(
                xp, image, levels, lut, forceApplyLut=False
            )
            levels = None

            if transparentLocations is not None:
                alpha_channel_required = True
                mask = xp.full(image.shape[:2], 255, dtype=xp.uint8)
                mask[transparentLocations] = 0
                image = xp.dstack((image, mask))

    # if the image data is a small int, then we can combine levels + lut
    # into a single lut for better performance
    elif image.dtype in (xp.ubyte, xp.uint16):
        image, lut = _combine_levels_and_lut(xp, image, levels, lut)
        levels = None

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
            if alpha_channel_required:
                fmt = QtGui.QImage.Format.Format_RGBA8888
            else:
                fmt = QtGui.QImage.Format.Format_RGBX8888
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
