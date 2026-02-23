import os
import pathlib
import warnings
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from .. import colormap
from .. import debug as debug
from .. import functions as fn
from .. import functions_qimage
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..util.cupy_helper import getCupy
from .GraphicsObject import GraphicsObject

translate = QtCore.QCoreApplication.translate

__all__ = ['ImageItem']


class ImageItem(GraphicsObject):
    """
    Graphics object used to display image data.

    ImageItem can render images with 1, 3 or 4 channels, use lookup tables to apply
    false colors to images, and users can either set levels limits, or rely on
    the auto-sampling.

    Performance can vary wildly based on the attributes of the inputs provided, see
    :ref:`performance <ImageItem_performance>` for guidance if performance is an 
    important factor.

    There is optional `numba` and `cupy` support.

    **Bases:** :class:`pyqtgraph.GraphicsObject`

    Parameters
    ----------
    image : np.ndarray or None, default None
        Image data.
    **kargs : dict, optional
        Arguments directed to `setImage` and `setOpts`, refer to each method for
        documentation for possible arguments.

    Signals
    -------
    sigImageChanged: :class:`Signal`
        Emitted when the image is changed.
    sigRemoveRequested: :class:`Signal`
        Emitted when there is a request to remove the image. Signal emits the instance
        of the :class:`~pyqtgraph.ImageItem` whose removal is requested.

    See Also
    --------
    setImage :
        For descriptions of available keyword arguments.
    setOpts :
        For information on supported formats.
    """
    sigImageChanged = QtCore.Signal()
    sigRemoveRequested = QtCore.Signal(object) 

    def __init__(self, image: np.ndarray | None=None, **kargs):
        super().__init__()
        self.menu = None
        self.image = None   ## original image data
        self.qimage = None  ## rendered image for display

        self.paintMode = None
        self.levels = None  ## [min, max] or [[redMin, redMax], ...]
        self.lut = None
        self.autoDownsample = False
        self._nanPolicy = 'propagate'
        self._colorMap = None # This is only set if a color map is assigned directly
        self._lastDownsample = (1, 1)
        self._processingBuffer = None
        self._displayBuffer = None
        self._renderRequired = True
        self._unrenderable = False
        self._xp = None  # either numpy or cupy, to match the image data
        self._defferedLevels = None
        self._imageHasNans = None    # None : not yet known
        self._imageNanLocations = None
        self._defaultAutoLevels = True

        self.axisOrder = getConfigOption('imageAxisOrder')
        self._dataTransform = self._inverseDataTransform = QtGui.QTransform()
        self._update_data_transforms( self.axisOrder ) # install initial transforms

        self.drawKernel = None
        self.border = None
        self.removable = False

        if image is not None:
            self.setImage(image, **kargs)
        else:
            self.setOpts(**kargs)

    def setCompositionMode(self, mode: QtGui.QPainter.CompositionMode):
        """
        Change the composition mode of the item, useful when overlaying multiple items.
        
        Parameters
        ----------
        mode : :class:`QPainter.CompositionMode <QPainter.CompositionMode>`
            Composition of the item, often used when overlaying items.  Common
            options include:

            * `QPainter.CompositionMode.CompositionMode_SourceOver`
              Image replaces the background if it is opaque. Otherwise, it uses the
              alpha channel to blend the image with the background, default.
            * `QPainter.CompositionMode.CompositionMode_Overlay` Image color is
              mixed with the background color to reflect the lightness or darkness of
              the background.
            * `QPainter.CompositionMode.CompositionMode_Plus` Both the alpha and
              color of the image and background pixels are added together.
            * `QPainter.CompositionMode.CompositionMode_Plus` The output is the
              image color multiplied by the background.

            See :class:`QPainter.CompositionMode <QPainter.CompositionMode>` in the Qt
            documentation for more options and details.
        
        See Also
        --------
        :class:`QPainter.CompositionMode <QPainter.CompositionMode>` :
            Details all the possible composition mode options accepted.
        """
        self.paintMode = mode
        self.update()

    def setBorder(self, b):
        """
        Define the color of the border drawn around the image.

        Parameters
        ----------
        b : color_like
            Accepts all arguments supported by :func:`~pyqtgraph.mkPen`.
        """
        self.border = fn.mkPen(b)
        self.update()

    def width(self) -> int | None:
        if self.image is None:
            return None
        axis = 0 if self.axisOrder == 'col-major' else 1
        return self.image.shape[axis]

    def height(self) -> int | None:
        if self.image is None:
            return None
        axis = 1 if self.axisOrder == 'col-major' else 0
        return self.image.shape[axis]

    def channels(self) -> int | None:
        if self.image is None:
            return None
        return self.image.shape[2] if self.image.ndim == 3 else 1

    def boundingRect(self) -> QtCore.QRectF:
        if self.image is None:
            return QtCore.QRectF(0., 0., 0., 0.)
        if (width := self.width()) is None:
            width = 0.
        if (height := self.height()) is None:
            height = 0.
        return QtCore.QRectF(0., 0., float(width), float(height))

    def setAutoLevels(self, bState: bool):
        """
        Controls whether automatic image scaling takes place for this ImageItem,
        if not otherwise overridden by ``autoLevels`` or ``levels`` keyword
        arguments in a call to :func:`~pyqtgraph.ImageItem.setImage`.
        """
        self._defaultAutoLevels = bState

    def setLevels(self, levels: npt.ArrayLike | None, update: bool=True):
        """
        Set image scaling levels.

        Calling this method, even with ``levels=None`` will disable auto leveling 
        which is equivalent to :meth:`setImage` with ``autoLevels=False``.
        
        Parameters
        ----------
        levels : array_like or None
            Sets the numerical values that correspond to the limits of the color range.

            * ``[blackLevel, whiteLevel]`` 
                sets black and white levels for monochrome data and can be used with a
                lookup table.
            * ``[[minR, maxR], [minG, maxG], [minB, maxB]]``
                sets individual scaling for RGB values. Not compatible with lookup 
                tables.
            * ``None``
                Disables the application of levels, but setting to ``None`` prevents
                the auto-levels mechanism from sampling the image.  Not compatible with
                images that use floating point dtypes.
        update : bool, default True
            Update the image immediately to reflect the new levels.

        See Also
        --------
        pyqtgraph.functions.makeARGB
            For more details on how levels are applied.
        """
        if self._xp is None:
            self.levels = levels
            self._defferedLevels = levels
            return
        if levels is not None:
            levels = self._xp.asarray(levels)
        self.levels = levels
        if update:
            self.updateImage()

    def getLevels(self) -> np.ndarray | None:
        """
        Return the array representing the current level settings.

        See :meth:`setLevels`. When `autoLevels` is active, the format is
        ``[blackLevel, whiteLevel]``.

        Returns
        -------
        np.ndarray or None
            The value that the levels are set to.
        """
        return self.levels
        
    def setColorMap(self, colorMap: colormap.ColorMap | str):
        """
        Set a color map for false color display of a monochrome image.

        Parameters
        ----------
        colorMap : :class:`~pyqtgraph.ColorMap` or `str`
            A string argument will be passed to
            :func:`colormap.get() <pyqtgraph.colormap.get>`.
        
        Raises
        ------
        TypeError
            Raised when `colorMap` is not of type `str` or :class:`~pyqtgraph.ColorMap`.
        """
        if isinstance(colorMap, colormap.ColorMap):
            self._colorMap = colorMap
        elif isinstance(colorMap, str):
            self._colorMap = colormap.get(colorMap)
        else:
            raise TypeError("'colorMap' argument must be ColorMap or string")
        self.setLookupTable( self._colorMap.getLookupTable(nPts=256) )
        
    def getColorMap(self) -> colormap.ColorMap | None:
        """
        Retrieve the :class:`~pyqtgraph.ColorMap` object currently used.

        Returns
        -------
        ColorMap or None 
            The assigned :class:`~pyqtgraph.ColorMap`, or `None` if not available.
        """
        return self._colorMap
        
    def setLookupTable(self, lut: npt.ArrayLike | Callable, update: bool=True):
        """
        Set lookup table `lut` to use for false color display of a monochrome image.

        Ordinarily, this table is supplied by a :class:`~pyqtgraph.HistogramLUTItem`,
        :class:`~pyqtgraph.GradientEditorItem` or :class:`~pyqtgraph.ColorBarItem`.

        Parameters
        ----------
        lut : array_like or callable
            If `lut` is an np.ndarray, ensure the dtype is `np.uint8`. Alternatively
            can be a callable that accepts the current image as an argument and
            returns the lookup table to use. Support for callable will be removed
            in a future version of pyqtgraph.
        update : bool, default True
            Update the intermediate image.
    
        See Also
        --------
        :func:`pyqtgraph.functions.makeARGB`
            See this function for more information on how this is used.
        :meth:`ColorMap.getLookupTable <pyqtgraph.ColorMap.getLookupTable>`
            Can construct a lookup table from a :class:`~pyqtgraph.ColorMap` object.

        Notes
        -----
        For performance reasons, if not passing a callable, every effort should be made
        to keep the number of entries to `<= 256`.
        """

        if lut is not self.lut:
            if self._xp is not None:
                lut = self._ensure_proper_substrate(lut, self._xp)
            self.lut = lut
            if update:
                self.updateImage()

    @staticmethod
    def _ensure_proper_substrate(data: Callable | npt.ArrayLike, substrate) -> np.ndarray:
        if data is None or isinstance(data, (Callable, substrate.ndarray)):
            return data
        cupy = getCupy()
        if substrate == cupy and not isinstance(data, cupy.ndarray):
            data = cupy.asarray(data)
        elif substrate == np:
            if cupy is not None and isinstance(data, cupy.ndarray):
                data = data.get()
            else:
                data = np.asarray(data)
        return data

    def setAutoDownsample(self, active: bool=True):
        """
        Control automatic downsampling for this ImageItem.

        Parameters
        ----------
        active : bool, default True
            If `active` is ``True``, the image is automatically downsampled to match
            the screen resolution. This improves performance for large images and
            reduces aliasing. If `autoDownsample` is not specified, then ImageItem will
            choose whether to downsample the image based on its size. ``False``
            disables automatic downsampling.
        """
        self.autoDownsample = active
        self._renderRequired = True
        self.update()

    def nanPolicy(self) -> str:
        """
        Retrieve the string representing the current NaN policy.

        See :meth:setNanPolicy.

        Returns
        -------
        { 'propagate', 'omit' }
            The NaN policy that this ImageItem uses during downsampling.
        """
        return self._nanPolicy

    def setNanPolicy(self, nanPolicy: str):
        """
        Control how NaN values are handled during downsampling for this ImageItem.

        Parameters
        ----------
        nanPolicy : { 'propagate', 'omit' }
            If 'nanPolicy' is 'ignore', NaNs are automatically ignored during
            downsampling, at the expense of performance. If 'nanPolicy' is 'propagate',
            NaNs are kept during downsampling. Unless a different policy was specified,
            a new ImageItem is created with ``nanPolicy='propagate'``.
        """
        if nanPolicy not in ['propagate', 'omit']:
            raise ValueError(f"{nanPolicy=} must be one of {'propagate', 'omit'}")
        self._nanPolicy = nanPolicy
        self._renderRequired = True
        self.update()

    def setOpts(self, update: bool=True, **kwargs):
        """
        Set display and processing options for this ImageItem.

        :class:`~pyqtgraph.ImageItem` and :meth:`setImage` support all keyword
        arguments listed here.

        Parameters
        ----------
        update : bool, default True
            Controls if image immediately updates to reflect the new options.

        **kwargs : dict, optional
            Extra arguments that are directed to the respective methods.  Expected
            keys include:

            * `autoDownsample` whose value is directed to :meth:`setAutoDownsample`
            * `nanPolicy` whose value is directed to :meth:`setNanPolicy`
            * `axisOrder`, which needs to be one of {'row-major', 'col-major'},
              determines the relationship between the numpy axis and visual axis
              of the data.
            * `border`, whose value is directed to :meth:`setBorder`
            * `colorMap`, whose value is directed to :meth:`setColorMap`
            * `compositionMode`, whose value is directed to :meth:`setCompositionMode`
            * `levels` whose value is directed to :meth:`setLevels`
            * `lut`, whose value  is directed to :meth:`setLookupTable`
            * `opacify` whose value is directed to
              :meth:`QGraphicsItem.setOpacity <QGraphicsItem.setOpacity>`
            * `rect` whose value is directed to :meth:`setRect`
            * `removable` boolean, determines if the context menu is available
        
        See Also
        --------
        :meth:`setAutoDownsample`
            Accepts the value of ``kwargs['autoDownsample']``.
        :meth:`setAutoLevels`
            Accepts the value of ``kwargs['autoLevels']``.
        :meth:`setNanPolicy`
            Accepts the value of ``kwargs['nanPolicy']``.
        :meth:`setBorder`
            Accepts the value of ``kwargs['border']``.
        :meth:`setColorMap`
            Accepts the value of ``kwargs['colorMap']``.
        :meth:`setCompositionMode`
            Accepts the value of ``kwargs['compositionMode']``.
        :meth:`setImage`
            Accepts the value of ``kwargs['image']``.
        :meth:`setLevels`
            Accepts the value of ``kwargs['levels']``.
        :meth:`setLookupTable`
            Accepts the value of ``kwargs['lut']``.
        :meth:`QGraphicsItem.setOpacity <QGraphicsItem.setOpacity>`
            Accepts the value of ``kwargs['opacity']``.
        :meth:`setRect`
            Accepts the value of ``kwargs['rect']``.
        """
        if 'axisOrder' in kwargs:
            val = kwargs['axisOrder']            
            if val not in ('row-major', 'col-major'):
                raise ValueError("axisOrder must be either 'row-major' or 'col-major'")
            self.axisOrder = val
            self._update_data_transforms(self.axisOrder) # update cached transforms
        if 'colorMap' in kwargs:
            self.setColorMap(kwargs['colorMap'])
        if 'lut' in kwargs:
            self.setLookupTable(kwargs['lut'], update=update)
        if 'levels' in kwargs:
            self.setLevels(kwargs['levels'], update=update)
        #if 'clipLevel' in kargs:
            #self.setClipLevel(kargs['clipLevel'])
        if 'opacity' in kwargs:
            self.setOpacity(kwargs['opacity'])
        if 'compositionMode' in kwargs:
            self.setCompositionMode(kwargs['compositionMode'])
        if 'border' in kwargs:
            self.setBorder(kwargs['border'])
        if 'removable' in kwargs:
            self.removable = kwargs['removable']
            self.menu = None
        if 'autoDownsample' in kwargs:
            self.setAutoDownsample(kwargs['autoDownsample'])
        if 'autoLevels' in kwargs:
            self.setAutoLevels(kwargs['autoLevels'])
        if 'nanPolicy' in kwargs:
            self.setNanPolicy(kwargs['nanPolicy'])
        if 'rect' in kwargs:
            self.setRect(kwargs['rect'])
        if update:
            self.update()

    def setRect(self, *args):
        """
        Set view rectangle for the :class:`~pyqtgraph.ImageItem` to occupy.

        In addition to accepting a :class:`QRectF`, you can pass the numerical values
        representing the  `x, y, w, h`, where `x, y` represent the x, y coordinates
        of the top left corner, and `w` and `h` represent the width and height
        respectively.

        Parameters
        ----------
        *args : tuple
            Contains one of :class:`QRectF`, :class:`QRect`, or arguments that can be
            used to construct :class:`QRectF`.

        See Also
        --------
        :class:`QRectF` :
            See constructor methods for allowable `*args`.

        Notes
        -----
        This method cannot be used before an image is assigned. See the
        :ref:`examples <ImageItem_examples>` for how to manually set transformations.
        """
        if not args:
            # reset scaling and rotation when called without argument
            self.resetTransform()
            return
        if isinstance(args[0], (QtCore.QRectF, QtCore.QRect)):
            rect = args[0] # use QRectF or QRect directly
        else:
            if hasattr(args[0],'__len__'):
                args = args[0] # promote tuple or list of values
            # QRectF(x,y,w,h), but also accepts other initializers
            rect = QtCore.QRectF( *args ) 
        tr = QtGui.QTransform()
        tr.translate(rect.left(), rect.top())

        if (width := self.width()) is None:
            width = 1.

        if (height := self.height()) is None:
            height = 1.

        tr.scale(rect.width() / width, rect.height() / height)
        self.setTransform(tr)

    def clear(self):
        """
        Clear the assigned image.
        """
        self.image = None
        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        self.update()

    def _buildQImageBuffer(self, shape: tuple[int, int, int]):
        self._displayBuffer = np.empty(shape[:2] + (4,), dtype=np.ubyte)
        if self._xp == getCupy():
            self._processingBuffer = self._xp.empty(
                shape[:2] + (4,),
                dtype=self._xp.ubyte
            )
        else:
            self._processingBuffer = self._displayBuffer
        self.qimage = None

    def setImage(
            self,
            image: np.ndarray | None=None,
            autoLevels: bool | None=None,
            levelSamples: int = 65536,
            **kwargs
        ):
        """
        Update the image displayed by this ImageItem.
        
        All keywords supported by :meth:`setOpts` are also allowed here.

        Parameters
        ----------
        image : np.ndarray or None, default None
            Image data given as NumPy array with an integer or floating point dtype of
            any bit depth. A 2-dimensional array describes single-valued
            (monochromatic) data. A 3-dimensional array is used to give individual
            color components. The third dimension must be of length 3 (RGB) or 4
            (RGBA). ``np.nan`` values are treated as transparent pixels.
        autoLevels : bool or None, default None
            If ``True``, ImageItem will automatically select levels based on the maximum
            and minimum values encountered in the data. For performance reasons, this
            search sub-samples the images and may miss individual bright or dark points
            in the data set. If ``False``, the search will be omitted. If ``None``, the
            value set by :func:`~pyqtgraph.ImageItem.setOpts` is used, unless a ``levels``
            keyword argument is given, which implies `False`.
        levelSamples : int, default 65536
            Only used when ``autoLevels is None``.  When determining minimum and
            maximum values, ImageItem only inspects a subset of pixels no larger than
            this number. Setting this larger than the total number of pixels considers
            all values. See `quickMinMax`.
        **kwargs : dict, optional
            Extra arguments that are passed to `setOpts`.

        See Also
        --------
        quickMinMax
            See this method for how levelSamples value is utilized.
        :func:`pyqtgraph.functions.makeARGB`
            See this function for how image data is modified prior to rendering.

        Notes
        -----
        For backward compatibility, image data is assumed to be in column-major order
        (column, row) by default. However, most data is stored in row-major order
        (row, column). It can either be transposed before assignment
        
        .. code-block:: python

            imageitem.setImage(imagedata.T)
        
        or the interpretation of the data can be changed locally through the
        `axisOrder` keyword or by changing the `imageAxisOrder`
        :ref:`global configuration option <apiref_config>`.
        """
        profile = debug.Profiler()

        gotNewData = False
        if image is None:
            if self.image is None:
                return
        else:
            old_xp = self._xp
            cp = getCupy()
            self._xp = cp.get_array_module(image) if cp else np
            gotNewData = True
            processingSubstrateChanged = old_xp != self._xp
            if processingSubstrateChanged:
                self._processingBuffer = None
            shapeChanged = (
                processingSubstrateChanged or
                self.image is None or
                image.shape != self.image.shape
            )
            image = image.view()
            self.image = image
            self._imageHasNans = None
            self._imageNanLocations = None
            if 'autoDownsample' not in kwargs and (
                self.image.shape[0] > 2**15-1 or self.image.shape[1] > 2**15-1
            ):
                kwargs['autoDownsample'] = True
            if shapeChanged:
                self.prepareGeometryChange()
                self.informViewBoundsChanged()

        profile()

        if autoLevels is None:
            autoLevels = False if 'levels' in kwargs else self._defaultAutoLevels

        if autoLevels:
            mn, mx = self.quickMinMax( targetSize=levelSamples )
            # mn and mx can still be NaN if the data is all-NaN
            if mn == mx or self._xp.isnan(mn) or self._xp.isnan(mx):
                mn = 0
                mx = 255
            kwargs['levels'] = self._xp.asarray((mn,mx))

        profile()

        self.setOpts(update=False, **kwargs)

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

    def _update_data_transforms(self, axisOrder: str='col-major'):
        """
        Set up the transforms needed to map between input array and display.

        Parameters
        ----------
        axisOrder : { 'col-major', 'row-major' }
            The axis order to update the data transformation to.
        """
        self._dataTransform = QtGui.QTransform()
        self._inverseDataTransform = QtGui.QTransform()
        if self.axisOrder == 'row-major': # transpose both
            self._dataTransform.scale(1, -1)
            self._dataTransform.rotate(-90)
            self._inverseDataTransform.scale(1, -1)
            self._inverseDataTransform.rotate(-90)

    def dataTransform(self):
        """
        Get the transform mapping image array to local coordinate system.

        This transform corrects for the transposition that occurs when image data is
        interpreted in row-major order.
        
        :meta private:

        Returns
        -------
        :class:`QTransform`
            The transform that is used for mapping.
        """
        # Might eventually need to account for downsampling / clipping here
        # transforms are updated in setOpts call.
        return self._dataTransform

    def inverseDataTransform(self) -> QtGui.QTransform:
        """
        Get the transform mapping local coordinate system to image array.

        :meta private:

        Returns
        -------
        :class:`QTransform`
            The transform that is used for mapping.
        
        See Also
        --------
        dataTransform
            See dataTransform() for more information. 
        """
        # transforms are updated in setOpts call.
        return self._inverseDataTransform

    def mapToData(self, obj):
        return self._inverseDataTransform.map(obj)

    def mapFromData(self, obj):
        return self._dataTransform.map(obj)

    def quickMinMax(self, targetSize: int=1_000_000) -> tuple[float, float]:
        """
        Estimate the min and max values of the image data by sub-sampling.

        Sampling is performed at regular strides chosen to evaluate a number of 
        samples equal to or less than `targetSize`.  Returns the estimated min and max
        values of the image data.

        Parameters
        ----------
        targetSize : int, default 1_000_000
            The number of pixels to downsample the image to.

        Returns
        -------
        float, float
            Estimated minimum and maximum values of the image data.
        """

        data = self.image
        if data is None:
            # image hasn't been set yet
            return 0., 0.
        targetSize = max(targetSize, 2) # keep at least 2 pixels
        while True:
            h, w = data.shape[:2]
            if h * w <= targetSize: break
            data = data[::2, ::] if h > w else data[::, ::2]
        return self._xp.nanmin(data), self._xp.nanmax(data)

    def updateImage(self, *args, **kargs):
        defaults = {
            'autoLevels': False,
        } | kargs
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
                lut = self._ensure_proper_substrate(
                    self.lut(self.image, 256),
                    self._xp
                )
            else:
                lut = self.lut
        else:
            lut = None

        if self._imageHasNans is None:
            # awkward, but fastest numpy native nan evaluation
            self._imageHasNans = (
                self.image.dtype.kind == 'f' and
                self._xp.isnan(self.image.min())
            )
            self._imageNanLocations = None

        image = self.image

        if self.autoDownsample:
            xds, yds = self._computeDownsampleFactors()
            if xds is None:
                return

            axes = [1, 0] if self.axisOrder == 'row-major' else [0, 1]
            nan_policy = self._nanPolicy if self._imageHasNans else 'propagate'
            image = fn.downsample(image, xds, axis=axes[0], nanPolicy=nan_policy)
            image = fn.downsample(image, yds, axis=axes[1], nanPolicy=nan_policy)
            self._lastDownsample = (xds, yds)

            # changes in view transform cause changes in downsampling factors,
            # which invalidates any previously calculated nan locations
            self._imageNanLocations = None

            # Check if downsampling reduced the image size to zero due to inf values.
            if image.size == 0:
                return

        # Convert single-channel image to 2D array
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]

        # Assume images are in column-major order for backward compatibility
        # (most images are in row-major order)
        if self.axisOrder == 'col-major':
            image = image.swapaxes(0, 1)

        levels = self.levels


        qimage = None

        if lut is not None and lut.dtype != self._xp.uint8:
            # try_make_image() assumes that lut is of type uint8.
            # It is considered a usage error if that is not the case.
            # However, the makeARGB() code-path has previously allowed such
            # a usage to work. Rather than fail outright, we delegate this
            # case to makeARGB().
            warnings.warn(
                ("Using non-uint8 LUTs is an undocumented accidental feature and may "
                "be removed at some point in the future. Please open an issue if you "
                "instead believe this to be worthy of protected inclusion in "
                "pyqtgraph."),
                DeprecationWarning,
                stacklevel=2
            )

        elif not self._imageHasNans:
            qimage = functions_qimage.try_make_qimage(image, levels=levels, lut=lut)

        elif image.ndim in (2, 3):
            # float images with nans
            if self._imageNanLocations is None:
                # the number of nans is expected to be small
                nanmask = self._xp.isnan(image)
                if nanmask.ndim == 3:
                    nanmask = nanmask.any(axis=2)
                self._imageNanLocations = nanmask.nonzero()
            qimage = functions_qimage.try_make_qimage(
                image,
                levels=levels,
                lut=lut,
                transparentLocations=self._imageNanLocations
            )

        if qimage is not None:
            self._processingBuffer = None
            self._displayBuffer = None
            self.qimage = qimage
            self._renderRequired = False
            self._unrenderable = False
            return

        if (
            self._processingBuffer is None or 
            self._processingBuffer.shape[:2] != image.shape[:2]
        ):
            self._buildQImageBuffer(image.shape)

        fn.makeARGB(image, lut=lut, levels=levels, output=self._processingBuffer)
        if self._xp == getCupy():
            self._processingBuffer.get(out=self._displayBuffer)
        self.qimage = fn.ndarray_to_qimage(
            self._displayBuffer,
            QtGui.QImage.Format.Format_ARGB32
        )

        self._renderRequired = False
        self._unrenderable = False

    def paint(self, painter, *args):
        profile = debug.Profiler()
        if self.image is None:
            return
        if self._renderRequired:
            self.render()
            if self._unrenderable:
                return
            profile('render QImage')
        if self.paintMode is not None:
            painter.setCompositionMode(self.paintMode)
            profile('set comp mode')

        shape = (
            self.image.shape[:2]
            if self.axisOrder == 'col-major'
            else self.image.shape[:2][::-1]
        )
        painter.drawImage(QtCore.QRectF(0,0,*shape), self.qimage)
        profile('p.drawImage')
        if self.border is not None:
            painter.setPen(self.border)
            painter.drawRect(self.boundingRect())

    def save(self, fileName: str | pathlib.Path, *args) -> None:
        """
        Save this image to file.

        Note that this saves the visible image, after scale/color changes, not the
        original data.

        Parameters
        ----------
        fileName : os.PathLike
            File path to save the image data to.
        *args : tuple
            Arguments that are passed to :meth:`QImage.save <QImage.save>`.
            
        See Also
        --------
        :meth:`QImage.save <QImage.save>` :
            ``*args`` is relayed to this method.
        """
        if self.qimage is None:
            return None

        if self._renderRequired:
            self.render()

        self.qimage.save(os.fsdecode(fileName), *args)

    def getHistogram(
        self,
        bins: str | int='auto',
        step: str | np.generic='auto',
        perChannel: bool=False,
        targetImageSize: int=200,
        **kwargs
    ) -> list[tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """
        Generate arrays containing the histogram values.

        Similar to :func:`numpy.histogram`

        Parameters
        ----------
        bins : int or str, default 'auto'
            The `bins` argument and any extra keyword arguments are passed to
            :func:`numpy.histogram()`. If ``bins == 'auto'``, a bin number is
            automatically chosen based on the image characteristics.
        step : int or str, default 'auto'
            The `step` argument causes pixels to be skipped when computing the
            histogram to save time. If `step` is 'auto', then a step is chosen such
            that the analyzed data has dimensions approximating `targetImageSize`
            for each axis.
        perChannel : bool, default False
            If ``True``, then a histogram is computed for each channel, and the output
            is a list of the results.
        targetImageSize : int, default 200
            This parameter is used if ``step == 'auto'``, If so, the `step` size is
            calculated by ``step = ceil(image.shape[0] / targetImageSize)``.
        **kwargs : dict, optional
            Dictionary of arguments passed to :func:`numpy.histogram()`.
        
        Returns
        -------
        numpy.ndarray, numpy.ndarray or None, None or list of tuple of numpy.ndarray, numpy.ndarray
            Returns `x` and `y` arrays containing the histogram values for the current
            image. For an explanation of the return format, see
            :func:`numpy.histogram()`.
            Returns ``[(numpy.ndarray, numpy.ndarray),...]`` if ``perChannel=True``, one
            element per channel.
            Returns ``(None, None)`` is there is no image, or image size is 0.

        Warns
        -----
        RuntimeWarning
            Emits when `targetHistogramSize` argument is passed in, which does nothing.

        See Also
        --------
        numpy.histogram :
            Describes return format in greater detail.
        numpy.histogram_bin_edges:
            Details the different string values accepted as the `bins` parameter.
        """        

        if 'targetHistogramSize' in kwargs:
            warnings.warn(
                "'targetHistogramSize' option is not used",
                RuntimeWarning,
                stacklevel=2
            )

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

        kwargs['bins'] = bins

        cp = getCupy()
        if perChannel:
            hist = []
            for i in range(stepData.shape[-1]):
                stepChan = stepData[..., i]
                stepChan = stepChan[self._xp.isfinite(stepChan)]
                h = self._xp.histogram(stepChan, **kwargs)
                if cp:
                    hist.append((cp.asnumpy(h[1][:-1]), cp.asnumpy(h[0])))
                else:
                    hist.append((h[1][:-1], h[0]))
            return hist
        else:
            stepData = stepData[self._xp.isfinite(stepData)]
            hist = self._xp.histogram(stepData, **kwargs)
            if cp:
                return cp.asnumpy(hist[1][:-1]), cp.asnumpy(hist[0])
            else:
                return hist[1][:-1], hist[0]

    def setPxMode(self, b: bool):
        """
        Set whether item ignores transformations and draws directly to screen pixels.

        Parameters
        ----------
        b : bool
            If ``True``, the item will not inherit any scale or rotation
            transformations from its parent items, but its position will be transformed
            as usual.

        See Also
        --------
        :class:`QGraphicsItem.GraphicsItemFlag <QGraphicsItem.GraphicsItemFlag>` :
            Read the description of `ItemIgnoresTransformations` for more information.
        """
        self.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
            b
        )

    def setScaledMode(self):
        self.setPxMode(False)

    def getPixmap(self) -> QtGui.QPixmap | None:
        if self._renderRequired:
            self.render()
            if self._unrenderable:
                return None
        if self.qimage is None:
            return QtGui.QPixmap()
        return QtGui.QPixmap.fromImage(self.qimage)

    def pixelSize(self) -> tuple[float, float]:
        """
        Get the `x` and `y` size of each pixel in the view coordinate system.

        Returns
        -------
        float, float
            The `x` and `y` size of each pixel in scene space.
        """
        br = self.sceneBoundingRect()
        if self.image is None:
            return 1.,1.

        if (width := self.width()) is None:
            width = 0.
        if (height := self.height()) is None:
            height = 0.

        return br.width() / width, br.height() / height

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

    def _computeDownsampleFactors(self) -> tuple[int, int]:
        # reduce dimensions of image based on screen resolution
        o = self.mapToDevice(QtCore.QPointF(0, 0))
        x = self.mapToDevice(QtCore.QPointF(1, 0))
        y = self.mapToDevice(QtCore.QPointF(0, 1))
        # scene may not be available yet
        if o is None:
            return 1, 1
        w = Point(x - o).length()
        h = Point(y - o).length()
        if w == 0 or h == 0:
            return 1, 1
        return max(1, int(1.0 / w)), max(1, int(1.0 / h))

    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        elif self.drawKernel is not None:
            ev.accept()
            self.drawAt(ev.pos(), ev)

    def mouseClickEvent(self, ev):
        if (
            ev.button() == QtCore.Qt.MouseButton.RightButton and
            self.raiseContextMenu(ev)
        ):
            ev.accept()
        if self.drawKernel is not None and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.drawAt(ev.pos(), ev)

    def raiseContextMenu(self, ev):
        menu = self.getMenu()
        if menu is None:
            return False
        if self.scene() is None:
            warnings.warn(
                (
                    "Attempting to raise a context menu with the GraphicsScene has "
                    "not been set. Returning None"
                ),
                RuntimeWarning,
                stacklevel=2
            )
            return None
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
        if not ev.isExit():
            if self.drawKernel is not None and ev.acceptDrags(
                QtCore.Qt.MouseButton.LeftButton
            ):
                # we don't use the click, but we also don't want anyone else to use it
                ev.acceptClicks(QtCore.Qt.MouseButton.LeftButton)
                ev.acceptClicks(QtCore.Qt.MouseButton.RightButton)
            elif self.removable:
                # accept context menu clicks
                ev.acceptClicks(QtCore.Qt.MouseButton.RightButton)

    def tabletEvent(self, ev):
        pass

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
                raise ValueError(f"Unknown draw mode '{self.drawMode}'")
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
