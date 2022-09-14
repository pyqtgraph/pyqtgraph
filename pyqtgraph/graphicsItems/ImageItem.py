import warnings
from collections.abc import Callable

import numpy

from .. import colormap
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..util.cupy_helper import getCupy
from .GraphicsObject import GraphicsObject

translate = QtCore.QCoreApplication.translate

__all__ = ['ImageItem']


class ImageItem(GraphicsObject):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`
    """
    # Overall description of ImageItem (including examples) moved to documentation text
    sigImageChanged = QtCore.Signal()
    sigRemoveRequested = QtCore.Signal(object)  # self; emitted when 'remove' is selected from context menu

    def __init__(self, image=None, **kargs):
        """
        See :func:`~pyqtgraph.ImageItem.setOpts` for further keyword arguments and 
        and :func:`~pyqtgraph.ImageItem.setImage` for information on supported formats.

        Parameters
        ----------
            image: np.ndarray, optional
                Image data
        """
        GraphicsObject.__init__(self)
        self.menu = None
        self.image = None   ## original image data
        self.qimage = None  ## rendered image for display

        self.paintMode = None
        self.levels = None  ## [min, max] or [[redMin, redMax], ...]
        self.lut = None
        self.autoDownsample = False
        self._colorMap = None # This is only set if a color map is assigned directly
        self._lastDownsample = (1, 1)
        self._processingBuffer = None
        self._displayBuffer = None
        self._renderRequired = True
        self._unrenderable = False
        self._xp = None  # either numpy or cupy, to match the image data
        self._defferedLevels = None

        self.axisOrder = getConfigOption('imageAxisOrder')
        self._dataTransform = self._inverseDataTransform = None
        self._update_data_transforms( self.axisOrder ) # install initial transforms

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
        """
        Change the composition mode of the item. This is useful when overlaying
        multiple items.
        
        Parameters
        ----------
        mode : ``QtGui.QPainter.CompositionMode``
            Composition of the item, often used when overlaying items.  Common
            options include:

            ``QPainter.CompositionMode.CompositionMode_SourceOver`` (Default)
            Image replaces the background if it is opaque. Otherwise, it uses
            the alpha channel to blend the image with the background.

            ``QPainter.CompositionMode.CompositionMode_Overlay`` Image color is
            mixed with the background color to reflect the lightness or
            darkness of the background

            ``QPainter.CompositionMode.CompositionMode_Plus`` Both the alpha
            and color of the image and background pixels are added together.

            ``QPainter.CompositionMode.CompositionMode_Plus`` The output is the
            image color multiplied by the background.

            See ``QPainter::CompositionMode`` in the Qt Documentation for more
            options and details
        """
        self.paintMode = mode
        self.update()

    def setBorder(self, b):
        """
        Defines the border drawn around the image. Accepts all arguments supported by 
        :func:`~pyqtgraph.mkPen`.
        """
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
        Sets image scaling levels. 
        See :func:`makeARGB <pyqtgraph.makeARGB>` for more details on how levels are applied.
        
        Parameters
        ----------
            levels: array_like
                - ``[blackLevel, whiteLevel]`` 
                  sets black and white levels for monochrome data and can be used with a lookup table.
                - ``[[minR, maxR], [minG, maxG], [minB, maxB]]``
                  sets individual scaling for RGB values. Not compatible with lookup tables.
            update: bool, optional
                Controls if image immediately updates to reflect the new levels.
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
        """
        Returns the list representing the current level settings. See :func:`~setLevels`.
        When ``autoLevels`` is active, the format is ``[blackLevel, whiteLevel]``.
        """
        return self.levels
        
    def setColorMap(self, colorMap):
        """
        Sets a color map for false color display of a monochrome image.

        Parameters
        ----------
        colorMap : :class:`~pyqtgraph.ColorMap` or `str`
            A string argument will be passed to :func:`colormap.get() <pyqtgraph.colormap.get>`
        """
        if isinstance(colorMap, colormap.ColorMap):
            self._colorMap = colorMap
        elif isinstance(colorMap, str):
            self._colorMap = colormap.get(colorMap)
        else:
            raise TypeError("'colorMap' argument must be ColorMap or string")
        self.setLookupTable( self._colorMap.getLookupTable(nPts=256) )
        
    def getColorMap(self):
        """
        Returns the assigned :class:`pyqtgraph.ColorMap`, or `None` if not available
        """
        return self._colorMap
        
    def setLookupTable(self, lut, update=True):
        """
        Sets lookup table ``lut`` to use for false color display of a monochrome image. See :func:`makeARGB <pyqtgraph.makeARGB>` for more 
        information on how this is used. Optionally, `lut` can be a callable that accepts the current image as an
        argument and returns the lookup table to use.

        Ordinarily, this table is supplied by a :class:`~pyqtgraph.HistogramLUTItem`,
        :class:`~pyqtgraph.GradientEditorItem` or :class:`~pyqtgraph.ColorBarItem`.
        
        Setting ``update = False`` avoids an immediate image update.
        """
        if lut is not self.lut:
            if self._xp is not None:
                lut = self._ensure_proper_substrate(lut, self._xp)
            self.lut = lut
            self._effectiveLut = None
            if update:
                self.updateImage()

    @staticmethod
    def _ensure_proper_substrate(data, substrate):
        if data is None or isinstance(data, Callable) or isinstance(data, substrate.ndarray):
            return data
        cupy = getCupy()
        if substrate == cupy and not isinstance(data, cupy.ndarray):
            data = cupy.asarray(data)
        elif substrate == numpy:
            if cupy is not None and isinstance(data, cupy.ndarray):
                data = data.get()
            else:
                data = numpy.asarray(data)
        return data

    def setAutoDownsample(self, active=True):
        """
        Controls automatic downsampling for this ImageItem.

        If `active` is `True`, the image is automatically downsampled to match the
        screen resolution. This improves performance for large images and
        reduces aliasing. If `autoDownsample` is not specified, then ImageItem will
        choose whether to downsample the image based on its size.
        
        `False` disables automatic downsampling.
        """
        self.autoDownsample = active
        self._renderRequired = True
        self.update()

    def setOpts(self, update=True, **kargs):
        """
        Sets display and processing options for this ImageItem. :func:`~pyqtgraph.ImageItem.__init__` and 
        :func:`~pyqtgraph.ImageItem.setImage` support all keyword arguments listed here.

        Parameters
        ----------
            autoDownsample: bool
                See :func:`~pyqtgraph.ImageItem.setAutoDownsample`.
            axisOrder: str
                | `'col-major'`: The shape of the array represents (width, height) of the image. This is the default.
                | `'row-major'`: The shape of the array represents (height, width).
            border: bool
                Sets a pen to draw to draw an image border. See :func:`~pyqtgraph.ImageItem.setBorder`.
            compositionMode:
                See :func:`~pyqtgraph.ImageItem.setCompositionMode`
            colorMap: :class:`~pyqtgraph.ColorMap` or `str`
                Sets a color map. A string will be passed to :func:`colormap.get() <pyqtgraph.colormap.get()>`
            lut: array_like
                Sets a color lookup table to use when displaying the image.
                See :func:`~pyqtgraph.ImageItem.setLookupTable`.
            levels: array_like
                Shape of (min, max). Sets minimum and maximum values to use when
                rescaling the image data. By default, these will be set to the
                estimated minimum and maximum values in the image. If the image array
                has dtype uint8, no rescaling is necessary. See
                :func:`~pyqtgraph.ImageItem.setLevels`.
            opacity: float
                Overall opacity for an RGB image. Between 0.0-1.0.
            rect: :class:`QRectF`, :class:`QRect` or array_like
                Displays the current image within the specified rectangle in plot
                coordinates. If ``array_like``, should be of the of ``floats 
                (`x`,`y`,`w`,`h`)`` . See :func:`~pyqtgraph.ImageItem.setRect`.
            update : bool, optional
                Controls if image immediately updates to reflect the new options.
        """
        if 'axisOrder' in kargs:
            val = kargs['axisOrder']            
            if val not in ('row-major', 'col-major'):
                raise ValueError("axisOrder must be either 'row-major' or 'col-major'")
            self.axisOrder = val
            self._update_data_transforms(self.axisOrder) # update cached transforms
        if 'colorMap' in kargs:
            self.setColorMap(kargs['colorMap'])
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
        if 'rect' in kargs:
            self.setRect(kargs['rect'])
        if update:
            self.update()

    def setRect(self, *args):
        """
        setRect(rect) or setRect(x,y,w,h)
        
        Sets translation and scaling of this ImageItem to display the current image within the rectangle given
        as ``rect`` (:class:`QtCore.QRect` or :class:`QtCore.QRectF`), or described by parameters `x, y, w, h`, 
        defining starting position, width and height.

        This method cannot be used before an image is assigned.
        See the :ref:`examples <ImageItem_examples>` for how to manually set transformations.
        """
        if len(args) == 0:
            self.resetTransform() # reset scaling and rotation when called without argument
            return
        if isinstance(args[0], (QtCore.QRectF, QtCore.QRect)):
            rect = args[0] # use QRectF or QRect directly
        else:
            if hasattr(args[0],'__len__'):
                args = args[0] # promote tuple or list of values
            rect = QtCore.QRectF( *args ) # QRectF(x,y,w,h), but also accepts other initializers
        tr = QtGui.QTransform()
        tr.translate(rect.left(), rect.top())
        tr.scale(rect.width() / self.width(), rect.height() / self.height())
        self.setTransform(tr)

    def clear(self):
        """
        Clears the assigned image.
        """
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
        self.qimage = None

    def setImage(self, image=None, autoLevels=None, **kargs):
        """
        Updates the image displayed by this ImageItem. For more information on how the image
        is processed before displaying, see :func:`~pyqtgraph.makeARGB`.
        
        For backward compatibility, image data is assumed to be in column-major order (column, row) by default.
        However, most data is stored in row-major order (row, column). It can either be transposed before assignment::

            imageitem.setImage(imagedata.T)
        
        or the interpretation of the data can be changed locally through the ``axisOrder`` keyword or by changing the 
        `imageAxisOrder` :ref:`global configuration option <apiref_config>`
        
        All keywords supported by :func:`~pyqtgraph.ImageItem.setOpts` are also allowed here.

        Parameters
        ----------
        image: np.ndarray, optional
            Image data given as NumPy array with an integer or floating
            point dtype of any bit depth. A 2-dimensional array describes single-valued (monochromatic) data.
            A 3-dimensional array is used to give individual color components. The third dimension must
            be of length 3 (RGB) or 4 (RGBA).
        rect: QRectF or QRect or array_like, optional
            If given, sets translation and scaling to display the image within the
            specified rectangle. If ``array_like`` should be the form of floats
            ``[x, y, w, h]`` See :func:`~pyqtgraph.ImageItem.setRect`
        autoLevels: bool, optional
            If `True`, ImageItem will automatically select levels based on the maximum and minimum values encountered 
            in the data. For performance reasons, this search subsamples the images and may miss individual bright or
            or dark points in the data set.
            
            If `False`, the search will be omitted.

            The default is `False` if a ``levels`` keyword argument is given, and `True` otherwise.
        levelSamples: int, default 65536
            When determining minimum and maximum values, ImageItem
            only inspects a subset of pixels no larger than this number.
            Setting this larger than the total number of pixels considers all values.
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
            level_samples = kargs.pop('levelSamples', 2**16) 
            mn, mx = self.quickMinMax( targetSize=level_samples )
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

    def _update_data_transforms(self, axisOrder='col-major'):
        """ Sets up the transforms needed to map between input array and display """
        self._dataTransform = QtGui.QTransform()
        self._inverseDataTransform = QtGui.QTransform()
        if self.axisOrder == 'row-major': # transpose both
            self._dataTransform.scale(1, -1)
            self._dataTransform.rotate(-90)
            self._inverseDataTransform.scale(1, -1)
            self._inverseDataTransform.rotate(-90)

    def dataTransform(self):
        """
        Returns the transform that maps from this image's input array to its
        local coordinate system.

        This transform corrects for the transposition that occurs when image data
        is interpreted in row-major order.
        
        :meta private:
        """
        # Might eventually need to account for downsampling / clipping here
        # transforms are updated in setOpts call.
        return self._dataTransform

    def inverseDataTransform(self):
        """Return the transform that maps from this image's local coordinate
        system to its input array.

        See dataTransform() for more information.

        :meta private:
        """
        # transforms are updated in setOpts call.
        return self._inverseDataTransform

    def mapToData(self, obj):
        return self._inverseDataTransform.map(obj)

    def mapFromData(self, obj):
        return self._dataTransform.map(obj)

    def quickMinMax(self, targetSize=1e6):
        """
        Estimates the min/max values of the image data by subsampling.
        Subsampling is performed at regular strides chosen to evaluate a number of samples
        equal to or less than `targetSize`.
        
        Returns (`min`, `max`).
        """
        data = self.image
        if targetSize < 2: # keep at least two pixels
            targetSize = 2
        while True:
            h, w = data.shape[:2]
            if h * w <= targetSize: break
            if h > w:
                data = data[::2, ::] # downsample first axis
            else:
                data = data[::, ::2] # downsample second axis
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
            self.lut = self._ensure_proper_substrate(self.lut, self._xp)
            if isinstance(self.lut, Callable):
                lut = self._ensure_proper_substrate(self.lut(self.image), self._xp)
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

        if lut is not None and lut.dtype != self._xp.uint8:
            # Both _try_rescale_float() and _try_combine_lut() assume that
            # lut is of type uint8. It is considered a usage error if that
            # is not the case.
            # However, the makeARGB() codepath has previously allowed such
            # a usage to work. Rather than fail outright, we delegate this
            # case to makeARGB().
            warnings.warn(
                "Using non-uint8 LUTs is an undocumented accidental feature and may "
                "be removed at some point in the future. Please open an issue if you "
                "instead believe this to be worthy of protected inclusion in pyqtgraph.",
                DeprecationWarning, stacklevel=2)
        elif image.dtype.kind == 'f':
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
        self.qimage = fn.ndarray_to_qimage(self._displayBuffer, QtGui.QImage.Format.Format_ARGB32)

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

        fn_numba = fn.getNumbaFunctions()
        if xp == numpy and image.flags.c_contiguous and dtype == xp.uint16 and fn_numba is not None:
            lut, augmented_alpha = self._convert_2dlut_to_1dlut(lut)
            image = fn_numba.rescale_and_lookup1d(image, scale/rng, minVal, lut)
            if image.dtype == xp.uint32:
                image = image[..., xp.newaxis].view(xp.uint8)
            return image, None, None, augmented_alpha
        else:
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

        lut, augmented_alpha = self._convert_2dlut_to_1dlut(lut)

        fn_numba = fn.getNumbaFunctions()
        if xp == numpy and fn_numba is not None:
            image = fn_numba.numba_take(lut, image)
        else:
            image = lut[image]

        if image.dtype == xp.uint32:
            image = image[..., xp.newaxis].view(xp.uint8)

        return image, augmented_alpha

    def _convert_2dlut_to_1dlut(self, lut):
        # converts:
        #   - uint8 (N, 1) to uint8 (N,)
        #   - uint8 (N, 3) or (N, 4) to uint32 (N,)
        # this allows faster lookup as 1d lookup is faster
        xp = self._xp
        augmented_alpha = False

        if lut.ndim == 1:
            return lut, augmented_alpha

        if lut.shape[1] == 3:   # rgb
            # convert rgb lut to rgba so that it is 32-bits
            lut = xp.column_stack([lut, xp.full(lut.shape[0], 255, dtype=xp.uint8)])
            augmented_alpha = True
        if lut.shape[1] == 4:   # rgba
            lut = lut.view(xp.uint32)
        lut = lut.ravel()

        return lut, augmented_alpha

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
        qimage = fn.ndarray_to_qimage(image, fmt)
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
        """
        Saves this image to file. Note that this saves the visible image (after scale/color changes), not the 
        original data.
        """
        if self._renderRequired:
            self.render()
        self.qimage.save(fileName, *args)

    def getHistogram(self, bins='auto', step='auto', perChannel=False, targetImageSize=200,
                     targetHistogramSize=500, **kwds):
        """
        Returns `x` and `y` arrays containing the histogram values for the current image.
        For an explanation of the return format, see :func:`numpy.histogram()`.

        The `step` argument causes pixels to be skipped when computing the histogram to save time.
        If `step` is 'auto', then a step is chosen such that the analyzed data has
        dimensions approximating `targetImageSize` for each axis.

        The `bins` argument and any extra keyword arguments are passed to
        :func:`numpy.histogram()`. If `bins` is `auto`, a bin number is automatically
        chosen based on the image characteristics:

          * Integer images will have approximately `targetHistogramSize` bins,
            with each bin having an integer width.
          * All other types will have `targetHistogramSize` bins.

        If `perChannel` is `True`, then a histogram is computed for each channel, 
        and the output is a list of the results.
        """
        # This method is also used when automatically computing levels.
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
                    bins = self._xp.arange(mn, mx + 1.01 * step, step, dtype=int)
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
        Sets whether the item ignores transformations and draws directly to screen pixels.
        If `True`, the item will not inherit any scale or rotation transformations from its
        parent items, but its position will be transformed as usual.
        (see ``GraphicsItem::ItemIgnoresTransformations`` in the Qt documentation)
        """
        self.setFlag(self.GraphicsItemFlag.ItemIgnoresTransformations, b)

    def setScaledMode(self):
        self.setPxMode(False)

    def getPixmap(self):
        if self._renderRequired:
            self.render()
            if self._unrenderable:
                return None
        return QtGui.QPixmap.fromImage(self.qimage)

    def pixelSize(self):
        """
        Returns the scene-size of a single pixel in the image
        """
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
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        elif self.drawKernel is not None:
            ev.accept()
            self.drawAt(ev.pos(), ev)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            if self.raiseContextMenu(ev):
                ev.accept()
        if self.drawKernel is not None and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.drawAt(ev.pos(), ev)

    def raiseContextMenu(self, ev):
        menu = self.getMenu()
        if menu is None:
            return False
        menu = self.scene().addParentContextMenus(self, menu, ev)
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(int(pos.x()), int(pos.y())))
        return True

    def getMenu(self):
        if self.menu is None:
            if not self.removable:
                return None
            self.menu = QtWidgets.QMenu()
            self.menu.setTitle(translate("ImageItem", "Image"))
            remAct = QtGui.QAction(translate("ImageItem", "Remove image"), self.menu)
            remAct.triggered.connect(self.removeClicked)
            self.menu.addAction(remAct)
            self.menu.remAct = remAct
        return self.menu

    def hoverEvent(self, ev):
        if not ev.isExit() and self.drawKernel is not None and ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton):
            ev.acceptClicks(QtCore.Qt.MouseButton.LeftButton) ## we don't use the click, but we also don't want anyone else to use it.
            ev.acceptClicks(QtCore.Qt.MouseButton.RightButton)
        elif not ev.isExit() and self.removable:
            ev.acceptClicks(QtCore.Qt.MouseButton.RightButton)  ## accept context menu clicks

    def tabletEvent(self, ev):
        pass
        #print(ev.device())
        #print(ev.pointerType())
        #print(ev.pressure())

    def drawAt(self, pos, ev=None):
        if self.axisOrder == "col-major":
            pos = [int(pos.x()), int(pos.y())]
        else:
            pos = [int(pos.y()), int(pos.x())]
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
