# -*- coding: utf-8 -*-
from __future__ import division

import numpy

from .GraphicsObject import GraphicsObject
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtGui, QtCore
from ..util.cupy_helper import getCupy

try:
    from collections.abc import Callable
except ImportError:
    # fallback for python < 3.3
    from collections import Callable

translate = QtCore.QCoreApplication.translate

__all__ = ['ImageItem']


class ImageItem(GraphicsObject):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`

    GraphicsObject displaying an image. Optimized for rapid update (ie video display).
    This item displays either a 2D numpy array (height, width) or
    a 3D array (height, width, RGBa). This array is optionally scaled (see
    :func:`setLevels <pyqtgraph.ImageItem.setLevels>`) and/or colored
    with a lookup table (see :func:`setLookupTable <pyqtgraph.ImageItem.setLookupTable>`)
    before being displayed.

    ImageItem is frequently used in conjunction with
    :class:`HistogramLUTItem <pyqtgraph.HistogramLUTItem>` or
    :class:`HistogramLUTWidget <pyqtgraph.HistogramLUTWidget>` to provide a GUI
    for controlling the levels and lookup table used to display the image.
    """

    sigImageChanged = QtCore.Signal()
    sigRemoveRequested = QtCore.Signal(object)  # self; emitted when 'remove' is selected from context menu

    def __init__(self, image=None, **kargs):
        """
        See :func:`setImage <pyqtgraph.ImageItem.setImage>` for all allowed initialization arguments.
        """
        GraphicsObject.__init__(self)
        self.menu = None
        self.image = None   ## original image data
        self.qimage = None  ## rendered image for display

        self.paintMode = None
        self.levels = None  ## [min, max] or [[redMin, redMax], ...]
        self.lut = None
        self.autoDownsample = False
        self._lastDownsample = (1, 1)
        self._processingBuffer = None
        self._displayBuffer = None
        self._renderRequired = True
        self._unrenderable = False
        self._xp = None  # either numpy or cupy, to match the image data
        self._defferedLevels = None

        self.axisOrder = getConfigOption('imageAxisOrder')

        # In some cases, we use a modified lookup table to handle both rescaling
        # and LUT more efficiently
        self._effectiveLut = None

        self.drawKernel = None
        self.border = None
        self.removable = False

        if image is not None:
            self.setImage(image, **kargs)
        else:
            self.setOpts(**kargs)

    def setCompositionMode(self, mode):
        """Change the composition mode of the item (see QPainter::CompositionMode
        in the Qt documentation). This is useful when overlaying multiple ImageItems.

        ============================================  ============================================================
        **Most common arguments:**
        QtGui.QPainter.CompositionMode_SourceOver     Default; image replaces the background if it
                                                      is opaque. Otherwise, it uses the alpha channel to blend
                                                      the image with the background.
        QtGui.QPainter.CompositionMode_Overlay        The image color is mixed with the background color to
                                                      reflect the lightness or darkness of the background.
        QtGui.QPainter.CompositionMode_Plus           Both the alpha and color of the image and background pixels
                                                      are added together.
        QtGui.QPainter.CompositionMode_Multiply       The output is the image color multiplied by the background.
        ============================================  ============================================================
        """
        self.paintMode = mode
        self.update()

    def setBorder(self, b):
        self.border = fn.mkPen(b)
        self.update()

    def width(self):
        if self.image is None:
            return None
        axis = 0 if self.axisOrder == 'col-major' else 1
        return self.image.shape[axis]

    def height(self):
        if self.image is None:
            return None
        axis = 1 if self.axisOrder == 'col-major' else 0
        return self.image.shape[axis]

    def channels(self):
        if self.image is None:
            return None
        return self.image.shape[2] if self.image.ndim == 3 else 1

    def boundingRect(self):
        if self.image is None:
            return QtCore.QRectF(0., 0., 0., 0.)
        return QtCore.QRectF(0., 0., float(self.width()), float(self.height()))

    def setLevels(self, levels, update=True):
        """
        Set image scaling levels. Can be one of:

        * [blackLevel, whiteLevel]
        * [[minRed, maxRed], [minGreen, maxGreen], [minBlue, maxBlue]]

        Only the first format is compatible with lookup tables. See :func:`makeARGB <pyqtgraph.makeARGB>`
        for more details on how levels are applied.
        """
        if self._xp is None:
            self.levels = levels
            self._defferedLevels = levels
            return
        if levels is not None:
            levels = self._xp.asarray(levels)
        self.levels = levels
        self._effectiveLut = None
        if update:
            self.updateImage()

    def getLevels(self):
        return self.levels
        #return self.whiteLevel, self.blackLevel

    def setLookupTable(self, lut, update=True):
        """
        Set the lookup table (numpy array) to use for this image. (see
        :func:`makeARGB <pyqtgraph.makeARGB>` for more information on how this is used).
        Optionally, lut can be a callable that accepts the current image as an
        argument and returns the lookup table to use.

        Ordinarily, this table is supplied by a :class:`HistogramLUTItem <pyqtgraph.HistogramLUTItem>`
        or :class:`GradientEditorItem <pyqtgraph.GradientEditorItem>`.
        """
        if lut is not self.lut:
            self.lut = lut
            self._effectiveLut = None
            if update:
                self.updateImage()

    def setAutoDownsample(self, ads):
        """
        Set the automatic downsampling mode for this ImageItem.

        Added in version 0.9.9
        """
        self.autoDownsample = ads
        self._renderRequired = True
        self.update()

    def setOpts(self, update=True, **kargs):
        if 'axisOrder' in kargs:
            val = kargs['axisOrder']
            if val not in ('row-major', 'col-major'):
                raise ValueError('axisOrder must be either "row-major" or "col-major"')
            self.axisOrder = val
        if 'lut' in kargs:
            self.setLookupTable(kargs['lut'], update=update)
        if 'levels' in kargs:
            self.setLevels(kargs['levels'], update=update)
        #if 'clipLevel' in kargs:
            #self.setClipLevel(kargs['clipLevel'])
        if 'opacity' in kargs:
            self.setOpacity(kargs['opacity'])
        if 'compositionMode' in kargs:
            self.setCompositionMode(kargs['compositionMode'])
        if 'border' in kargs:
            self.setBorder(kargs['border'])
        if 'removable' in kargs:
            self.removable = kargs['removable']
            self.menu = None
        if 'autoDownsample' in kargs:
            self.setAutoDownsample(kargs['autoDownsample'])
        if update:
            self.update()

    def setRect(self, rect):
        """Scale and translate the image to fit within rect (must be a QRect or QRectF)."""
        tr = QtGui.QTransform()
        tr.translate(rect.left(), rect.top())
        tr.scale(rect.width() / self.width(), rect.height() / self.height())
        self.setTransform(tr)

    def clear(self):
        self.image = None
        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        self.update()

    def _buildQImageBuffer(self, shape):
        self._displayBuffer = numpy.empty(shape[:2] + (4,), dtype=numpy.ubyte)
        if self._xp == getCupy():
            self._processingBuffer = self._xp.empty(shape[:2] + (4,), dtype=self._xp.ubyte)
        else:
            self._processingBuffer = self._displayBuffer
        self.qimage = fn.makeQImage(self._displayBuffer, transpose=False, copy=False)

    def setImage(self, image=None, autoLevels=None, **kargs):
        """
        Update the image displayed by this item. For more information on how the image
        is processed before displaying, see :func:`makeARGB <pyqtgraph.makeARGB>`

        =================  =========================================================================
        **Arguments:**
        image              (numpy array) Specifies the image data. May be 2D (width, height) or
                           3D (width, height, RGBa). The array dtype must be integer or floating
                           point of any bit depth. For 3D arrays, the third dimension must
                           be of length 3 (RGB) or 4 (RGBA). See *notes* below.
        autoLevels         (bool) If True, this forces the image to automatically select
                           levels based on the maximum and minimum values in the data.
                           By default, this argument is true unless the levels argument is
                           given.
        lut                (numpy array) The color lookup table to use when displaying the image.
                           See :func:`setLookupTable <pyqtgraph.ImageItem.setLookupTable>`.
        levels             (min, max) The minimum and maximum values to use when rescaling the image
                           data. By default, this will be set to the minimum and maximum values
                           in the image. If the image array has dtype uint8, no rescaling is necessary.
        opacity            (float 0.0-1.0)
        compositionMode    See :func:`setCompositionMode <pyqtgraph.ImageItem.setCompositionMode>`
        border             Sets the pen used when drawing the image border. Default is None.
        autoDownsample     (bool) If True, the image is automatically downsampled to match the
                           screen resolution. This improves performance for large images and
                           reduces aliasing. If autoDownsample is not specified, then ImageItem will
                           choose whether to downsample the image based on its size.
        =================  =========================================================================


        **Notes:**

        For backward compatibility, image data is assumed to be in column-major order (column, row).
        However, most image data is stored in row-major order (row, column) and will need to be
        transposed before calling setImage()::

            imageitem.setImage(imagedata.T)

        This requirement can be changed by calling ``image.setOpts(axisOrder='row-major')`` or
        by changing the ``imageAxisOrder`` :ref:`global configuration option <apiref_config>`.


        """
        profile = debug.Profiler()

        gotNewData = False
        if image is None:
            if self.image is None:
                return
        else:
            old_xp = self._xp
            cp = getCupy()
            self._xp = cp.get_array_module(image) if cp else numpy
            gotNewData = True
            processingSubstrateChanged = old_xp != self._xp
            if processingSubstrateChanged:
                self._processingBuffer = None
            shapeChanged = (processingSubstrateChanged or self.image is None or image.shape != self.image.shape)
            image = image.view()
            if self.image is None or image.dtype != self.image.dtype:
                self._effectiveLut = None
            self.image = image
            if self.image.shape[0] > 2**15-1 or self.image.shape[1] > 2**15-1:
                if 'autoDownsample' not in kargs:
                    kargs['autoDownsample'] = True
            if shapeChanged:
                self.prepareGeometryChange()
                self.informViewBoundsChanged()

        profile()

        if autoLevels is None:
            if 'levels' in kargs:
                autoLevels = False
            else:
                autoLevels = True
        if autoLevels:
            img = self.image
            while img.size > 2**16:
                img = img[::2, ::2]
            mn, mx = self._xp.nanmin(img), self._xp.nanmax(img)
            # mn and mx can still be NaN if the data is all-NaN
            if mn == mx or self._xp.isnan(mn) or self._xp.isnan(mx):
                mn = 0
                mx = 255
            kargs['levels'] = [mn,mx]

        profile()

        self.setOpts(update=False, **kargs)

        profile()

        self._renderRequired = True
        self.update()

        profile()

        if gotNewData:
            self.sigImageChanged.emit()
        if self._defferedLevels is not None:
            levels = self._defferedLevels
            self._defferedLevels = None
            self.setLevels((levels))

    def dataTransform(self):
        """Return the transform that maps from this image's input array to its
        local coordinate system.

        This transform corrects for the transposition that occurs when image data
        is interpreted in row-major order.
        """
        # Might eventually need to account for downsampling / clipping here
        tr = QtGui.QTransform()
        if self.axisOrder == 'row-major':
            # transpose
            tr.scale(1, -1)
            tr.rotate(-90)
        return tr

    def inverseDataTransform(self):
        """Return the transform that maps from this image's local coordinate
        system to its input array.

        See dataTransform() for more information.
        """
        tr = QtGui.QTransform()
        if self.axisOrder == 'row-major':
            # transpose
            tr.scale(1, -1)
            tr.rotate(-90)
        return tr

    def mapToData(self, obj):
        tr = self.inverseDataTransform()
        return tr.map(obj)

    def mapFromData(self, obj):
        tr = self.dataTransform()
        return tr.map(obj)

    def quickMinMax(self, targetSize=1e6):
        """
        Estimate the min/max values of the image data by subsampling.
        """
        data = self.image
        while data.size > targetSize:
            ax = self._xp.argmax(data.shape)
            sl = [slice(None)] * data.ndim
            sl[ax] = slice(None, None, 2)
            data = data[sl]
        return self._xp.nanmin(data), self._xp.nanmax(data)

    def updateImage(self, *args, **kargs):
        ## used for re-rendering qimage from self.image.

        ## can we make any assumptions here that speed things up?
        ## dtype, range, size are all the same?
        defaults = {
            'autoLevels': False,
        }
        defaults.update(kargs)
        return self.setImage(*args, **defaults)

    def render(self):
        # Convert data to QImage for display.
        self._unrenderable = True
        if self.image is None or self.image.size == 0:
            return

        # Request a lookup table if this image has only one channel
        if self.image.ndim == 2 or self.image.shape[2] == 1:
            if isinstance(self.lut, Callable):
                lut = self.lut(self.image)
            else:
                lut = self.lut
        else:
            lut = None

        if self.autoDownsample:
            xds, yds = self._computeDownsampleFactors()
            if xds is None:
                return

            axes = [1, 0] if self.axisOrder == 'row-major' else [0, 1]
            image = fn.downsample(self.image, xds, axis=axes[0])
            image = fn.downsample(image, yds, axis=axes[1])
            self._lastDownsample = (xds, yds)

            # Check if downsampling reduced the image size to zero due to inf values.
            if image.size == 0:
                return
        else:
            image = self.image

        # Convert single-channel image to 2D array
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]

        # Assume images are in column-major order for backward compatibility
        # (most images are in row-major order)
        if self.axisOrder == 'col-major':
            image = image.swapaxes(0, 1)

        levels = self.levels
        augmented_alpha = False

        if image.dtype.kind == 'f':
            image, levels, lut, augmented_alpha = self._try_rescale_float(image, levels, lut)
            # if we succeeded, we will have an uint8 image with levels None.
            # lut if not None will have <= 256 entries

        # if the image data is a small int, then we can combine levels + lut
        # into a single lut for better performance
        elif image.dtype in (self._xp.ubyte, self._xp.uint16):
            image, levels, lut, augmented_alpha = self._try_combine_lut(image, levels, lut)

        qimage = self._try_make_qimage(image, levels, lut, augmented_alpha)

        if qimage is not None:
            self._processingBuffer = None
            self._displayBuffer = None
            self.qimage = qimage
            self._renderRequired = False
            self._unrenderable = False
            return

        if self._processingBuffer is None or self._processingBuffer.shape[:2] != image.shape[:2]:
            self._buildQImageBuffer(image.shape)

        fn.makeARGB(image, lut=lut, levels=levels, output=self._processingBuffer)
        if self._xp == getCupy():
            self._processingBuffer.get(out=self._displayBuffer)

        self._renderRequired = False
        self._unrenderable = False

    def _try_rescale_float(self, image, levels, lut):
        xp = self._xp
        augmented_alpha = False

        can_handle = False
        while True:
            if levels is None or levels.ndim != 1:
                # float images always need levels
                # can't handle multi-channel levels
                break

            # awkward, but fastest numpy native nan evaluation
            if xp.isnan(image.min()):
                # don't handle images with nans
                # this should be an uncommon case
                break

            can_handle = True
            break

        if not can_handle:
            return image, levels, lut, augmented_alpha

        # Decide on maximum scaled value
        if lut is not None:
            scale = lut.shape[0]
            num_colors = lut.shape[0]
        else:
            scale = 255.
            num_colors = 256
        dtype = xp.min_scalar_type(num_colors-1)

        minVal, maxVal = levels
        if minVal == maxVal:
            maxVal = xp.nextafter(maxVal, 2*maxVal)
        rng = maxVal - minVal
        rng = 1 if rng == 0 else rng
        image = fn.rescaleData(image, scale/rng, offset=minVal, dtype=dtype, clip=(0, num_colors-1))

        levels = None

        if image.dtype == xp.uint16 and image.ndim == 2:
            image, augmented_alpha = self._apply_lut_for_uint16_mono(image, lut)
            lut = None

        # image is now of type uint8
        return image, levels, lut, augmented_alpha

    def _try_combine_lut(self, image, levels, lut):
        augmented_alpha = False
        xp = self._xp

        can_handle = False
        while True:
            if levels is not None and levels.ndim != 1:
                # can't handle multi-channel levels
                break
            if image.dtype == xp.uint16 and levels is None and \
                    image.ndim == 3 and image.shape[2] == 3:
                # uint16 rgb can't be directly displayed, so make it
                # pass through effective lut processing
                levels = [0, 65535]
            if levels is None and lut is None:
                # nothing to combine
                break

            can_handle = True
            break

        if not can_handle:
            return image, levels, lut, augmented_alpha

        # distinguish between lut for levels and colors
        levels_lut = None
        colors_lut = lut
        lut = None

        eflsize = 2**(image.itemsize*8)
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
                levels_lut = fn.rescaleData(ind, scale=255./levdiff,
                                        offset=minlev, dtype=xp.ubyte)
                # image data is not scaled. instead, levels_lut is used
                # as (grayscale) Indexed8 ColorTable to get the same effect.
                # due to the small size of the input to rescaleData(), we
                # do not bother caching the result
                return image, None, levels_lut, augmented_alpha
            else:
                # uint16 mono, uint8 rgb, uint16 rgb
                # rescale image data by computation instead of by memory lookup
                image = fn.rescaleData(image, scale=255./levdiff,
                                    offset=minlev, dtype=xp.ubyte)
                return image, None, colors_lut, augmented_alpha
        else:
            num_colors = colors_lut.shape[0]
            effscale = num_colors / levdiff
            lutdtype = xp.min_scalar_type(num_colors - 1)

            if image.dtype == xp.ubyte or lutdtype != xp.ubyte:
                # combine if either:
                #   1) uint8 mono image
                #   2) colors_lut has more entries than will fit within 8-bits
                if self._effectiveLut is None:
                    ind = xp.arange(eflsize)
                    levels_lut = fn.rescaleData(ind, scale=effscale,
                                    offset=minlev, dtype=lutdtype, clip=(0, num_colors-1))
                    efflut = colors_lut[levels_lut]
                    levels_lut = None
                    colors_lut = None
                    self._effectiveLut = efflut
                efflut = self._effectiveLut

                # apply the effective lut early for the following types:
                if image.dtype == xp.uint16 and image.ndim == 2:
                    image, augmented_alpha = self._apply_lut_for_uint16_mono(image, efflut)
                    efflut = None
                return image, None, efflut, augmented_alpha
            else:
                # uint16 image with colors_lut <= 256 entries
                # don't combine, we will use QImage ColorTable
                image = fn.rescaleData(image, scale=effscale,
                                offset=minlev, dtype=lutdtype, clip=(0, num_colors-1))
                return image, None, colors_lut, augmented_alpha

    def _apply_lut_for_uint16_mono(self, image, lut):
        # Note: compared to makeARGB(), we have already clipped the data to range

        xp = self._xp
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

        if lut.ndim == 2:
            if lut.shape[1] == 3:   # rgb
                # convert rgb lut to rgba so that it is 32-bits
                lut = xp.column_stack([lut, xp.full(lut.shape[0], 255, dtype=xp.uint8)])
                augmented_alpha = True
            if lut.shape[1] == 4:   # rgba
                lut = lut.view(xp.uint32)

        image = lut.ravel()[image]
        lut = None
        # now both levels and lut are None
        if image.dtype == xp.uint32:
            image = image.view(xp.uint8).reshape(image.shape + (4,))

        return image, augmented_alpha

    def _try_make_qimage(self, image, levels, lut, augmented_alpha):
        xp = self._xp

        ubyte_nolvl = image.dtype == xp.ubyte and levels is None
        is_passthru8 = ubyte_nolvl and lut is None
        is_indexed8 = ubyte_nolvl and image.ndim == 2 and \
            lut is not None and lut.shape[0] <= 256
        is_passthru16 = image.dtype == xp.uint16 and levels is None and lut is None
        can_grayscale16 = is_passthru16 and image.ndim == 2 and \
            hasattr(QtGui.QImage.Format, 'Format_Grayscale16')
        is_rgba64 = is_passthru16 and image.ndim == 3 and image.shape[2] == 4

        # bypass makeARGB for supported combinations
        supported = is_passthru8 or is_indexed8 or can_grayscale16 or is_rgba64
        if not supported:
            return None

        if self._xp == getCupy():
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
                ctbl = [QtGui.qRgb(x,x,x) for x in lut.ravel().tolist()]
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
            fmt = QtGui.QImage.Format.Format_RGBA64 # endian-independent
        if fmt is None:
            raise ValueError("unsupported image type")
        qimage = fn._ndarray_to_qimage(image, fmt)
        if ctbl is not None:
            qimage.setColorTable(ctbl)
        return qimage

    def paint(self, p, *args):
        profile = debug.Profiler()
        if self.image is None:
            return
        if self._renderRequired:
            self.render()
            if self._unrenderable:
                return
            profile('render QImage')
        if self.paintMode is not None:
            p.setCompositionMode(self.paintMode)
            profile('set comp mode')

        shape = self.image.shape[:2] if self.axisOrder == 'col-major' else self.image.shape[:2][::-1]
        p.drawImage(QtCore.QRectF(0,0,*shape), self.qimage)
        profile('p.drawImage')
        if self.border is not None:
            p.setPen(self.border)
            p.drawRect(self.boundingRect())

    def save(self, fileName, *args):
        """Save this image to file. Note that this saves the visible image (after scale/color changes), not the original data."""
        if self._renderRequired:
            self.render()
        self.qimage.save(fileName, *args)

    def getHistogram(self, bins='auto', step='auto', perChannel=False, targetImageSize=200,
                     targetHistogramSize=500, **kwds):
        """Returns x and y arrays containing the histogram values for the current image.
        For an explanation of the return format, see numpy.histogram().

        The *step* argument causes pixels to be skipped when computing the histogram to save time.
        If *step* is 'auto', then a step is chosen such that the analyzed data has
        dimensions roughly *targetImageSize* for each axis.

        The *bins* argument and any extra keyword arguments are passed to
        self.xp.histogram(). If *bins* is 'auto', then a bin number is automatically
        chosen based on the image characteristics:

        * Integer images will have approximately *targetHistogramSize* bins,
          with each bin having an integer width.
        * All other types will have *targetHistogramSize* bins.

        If *perChannel* is True, then the histogram is computed once per channel
        and the output is a list of the results.

        This method is also used when automatically computing levels.
        """
        if self.image is None or self.image.size == 0:
            return None, None
        if step == 'auto':
            step = (max(1, int(self._xp.ceil(self.image.shape[0] / targetImageSize))),
                    max(1, int(self._xp.ceil(self.image.shape[1] / targetImageSize))))
        if self._xp.isscalar(step):
            step = (step, step)
        stepData = self.image[::step[0], ::step[1]]

        if isinstance(bins, str) and bins == 'auto':
            mn = self._xp.nanmin(stepData).item()
            mx = self._xp.nanmax(stepData).item()
            if mx == mn:
                # degenerate image, arange will fail
                mx += 1
            if self._xp.isnan(mn) or self._xp.isnan(mx):
                # the data are all-nan
                return None, None
            if stepData.dtype.kind in "ui":
                # For integer data, we select the bins carefully to avoid aliasing
                step = int(self._xp.ceil((mx - mn) / 500.))
                bins = []
                if step > 0.0:
                    bins = self._xp.arange(mn, mx + 1.01 * step, step, dtype=self._xp.int)
            else:
                # for float data, let numpy select the bins.
                bins = self._xp.linspace(mn, mx, 500)

            if len(bins) == 0:
                bins = self._xp.asarray((mn, mx))

        kwds['bins'] = bins

        cp = getCupy()
        if perChannel:
            hist = []
            for i in range(stepData.shape[-1]):
                stepChan = stepData[..., i]
                stepChan = stepChan[self._xp.isfinite(stepChan)]
                h = self._xp.histogram(stepChan, **kwds)
                if cp:
                    hist.append((cp.asnumpy(h[1][:-1]), cp.asnumpy(h[0])))
                else:
                    hist.append((h[1][:-1], h[0]))
            return hist
        else:
            stepData = stepData[self._xp.isfinite(stepData)]
            hist = self._xp.histogram(stepData, **kwds)
            if cp:
                return cp.asnumpy(hist[1][:-1]), cp.asnumpy(hist[0])
            else:
                return hist[1][:-1], hist[0]

    def setPxMode(self, b):
        """
        Set whether the item ignores transformations and draws directly to screen pixels.
        If True, the item will not inherit any scale or rotation transformations from its
        parent items, but its position will be transformed as usual.
        (see GraphicsItem::ItemIgnoresTransformations in the Qt documentation)
        """
        self.setFlag(self.ItemIgnoresTransformations, b)

    def setScaledMode(self):
        self.setPxMode(False)

    def getPixmap(self):
        if self._renderRequired:
            self.render()
            if self._unrenderable:
                return None
        return QtGui.QPixmap.fromImage(self.qimage)

    def pixelSize(self):
        """return scene-size of a single pixel in the image"""
        br = self.sceneBoundingRect()
        if self.image is None:
            return 1,1
        return br.width()/self.width(), br.height()/self.height()

    def viewTransformChanged(self):
        if self.autoDownsample:
            xds, yds = self._computeDownsampleFactors()
            if xds is None:
                self._renderRequired = True
                self._unrenderable = True
                return
            if (xds, yds) != self._lastDownsample:
                self._renderRequired = True
                self.update()

    def _computeDownsampleFactors(self):
        # reduce dimensions of image based on screen resolution
        o = self.mapToDevice(QtCore.QPointF(0, 0))
        x = self.mapToDevice(QtCore.QPointF(1, 0))
        y = self.mapToDevice(QtCore.QPointF(0, 1))
        # scene may not be available yet
        if o is None:
            return None, None
        w = Point(x - o).length()
        h = Point(y - o).length()
        if w == 0 or h == 0:
            return None, None
        return max(1, int(1.0 / w)), max(1, int(1.0 / h))

    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        elif self.drawKernel is not None:
            ev.accept()
            self.drawAt(ev.pos(), ev)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            if self.raiseContextMenu(ev):
                ev.accept()
        if self.drawKernel is not None and ev.button() == QtCore.Qt.LeftButton:
            self.drawAt(ev.pos(), ev)

    def raiseContextMenu(self, ev):
        menu = self.getMenu()
        if menu is None:
            return False
        menu = self.scene().addParentContextMenus(self, menu, ev)
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(pos.x(), pos.y()))
        return True

    def getMenu(self):
        if self.menu is None:
            if not self.removable:
                return None
            self.menu = QtGui.QMenu()
            self.menu.setTitle(translate("ImageItem", "Image"))
            remAct = QtGui.QAction(translate("ImageItem", "Remove image"), self.menu)
            remAct.triggered.connect(self.removeClicked)
            self.menu.addAction(remAct)
            self.menu.remAct = remAct
        return self.menu

    def hoverEvent(self, ev):
        if not ev.isExit() and self.drawKernel is not None and ev.acceptDrags(QtCore.Qt.LeftButton):
            ev.acceptClicks(QtCore.Qt.LeftButton) ## we don't use the click, but we also don't want anyone else to use it.
            ev.acceptClicks(QtCore.Qt.RightButton)
        elif not ev.isExit() and self.removable:
            ev.acceptClicks(QtCore.Qt.RightButton)  ## accept context menu clicks

    def tabletEvent(self, ev):
        pass
        #print(ev.device())
        #print(ev.pointerType())
        #print(ev.pressure())

    def drawAt(self, pos, ev=None):
        pos = [int(pos.x()), int(pos.y())]
        dk = self.drawKernel
        kc = self.drawKernelCenter
        sx = [0,dk.shape[0]]
        sy = [0,dk.shape[1]]
        tx = [pos[0] - kc[0], pos[0] - kc[0]+ dk.shape[0]]
        ty = [pos[1] - kc[1], pos[1] - kc[1]+ dk.shape[1]]

        for i in [0,1]:
            dx1 = -min(0, tx[i])
            dx2 = min(0, self.image.shape[0]-tx[i])
            tx[i] += dx1+dx2
            sx[i] += dx1+dx2

            dy1 = -min(0, ty[i])
            dy2 = min(0, self.image.shape[1]-ty[i])
            ty[i] += dy1+dy2
            sy[i] += dy1+dy2

        ts = (slice(tx[0],tx[1]), slice(ty[0],ty[1]))
        ss = (slice(sx[0],sx[1]), slice(sy[0],sy[1]))
        mask = self.drawMask
        src = dk

        if isinstance(self.drawMode, Callable):
            self.drawMode(dk, self.image, mask, ss, ts, ev)
        else:
            src = src[ss]
            if self.drawMode == 'set':
                if mask is not None:
                    mask = mask[ss]
                    self.image[ts] = self.image[ts] * (1-mask) + src * mask
                else:
                    self.image[ts] = src
            elif self.drawMode == 'add':
                self.image[ts] += src
            else:
                raise Exception("Unknown draw mode '%s'" % self.drawMode)
            self.updateImage()

    def setDrawKernel(self, kernel=None, mask=None, center=(0,0), mode='set'):
        self.drawKernel = kernel
        self.drawKernelCenter = center
        self.drawMode = mode
        self.drawMask = mask

    def removeClicked(self):
        ## Send remove event only after we have exited the menu event handler
        self.removeTimer = QtCore.QTimer()
        self.removeTimer.timeout.connect(self.emitRemoveRequested)
        self.removeTimer.start(0)

    def emitRemoveRequested(self):
        self.removeTimer.timeout.disconnect(self.emitRemoveRequested)
        self.sigRemoveRequested.emit(self)
