Displaying images and video
===========================

Pyqtgraph displays 2D numpy arrays as images and provides tools for determining how to translate between the numpy data type and RGB values on the screen. If you want to display data from common image and video file formats, you will need to load the data first using another library (PIL works well for images and built-in numpy conversion). 

The easiest way to display 2D or 3D data is using the :func:`pyqtgraph.image` function::
    
    import pyqtgraph as pg
    pg.image(imageData)
    
This function will accept any floating-point or integer data types and displays a single :class:`~pyqtgraph.ImageView` widget containing your data. This widget includes controls for determining how the image data will be converted to 32-bit RGBa values. Conversion happens in two steps (both are optional):
    
1. Scale and offset the data (by selecting the dark/light levels on the displayed histogram)
2. Convert the data to color using a lookup table (determined by the colors shown in the gradient editor)

If the data is 3D (time, x, y), then a time axis will be shown with a slider that can set the currently displayed frame. (if the axes in your data are ordered differently, use numpy.transpose to rearrange them)

There are a few other methods for displaying images as well:
   
* The :class:`~pyqtgraph.ImageView` class can also be instantiated directly and embedded in Qt applications.
* Instances of :class:`~pyqtgraph.ImageItem` can be used inside a :class:`ViewBox <pyqtgraph.ViewBox>` or :class:`GraphicsView <pyqtgraph.GraphicsView>`.
* For higher performance, use :class:`~pyqtgraph.RawImageWidget`.

Any of these classes are acceptable for displaying video by calling setImage() to display a new frame. To increase performance, the image processing system uses scipy.weave to produce compiled libraries. If your computer has a compiler available, weave will automatically attempt to build the libraries it needs on demand. If this fails, then the slower pure-python methods will be used instead. 

For more information, see the classes listed above and the 'VideoSpeedTest', 'ImageItem', 'ImageView', and 'HistogramLUT' :ref:`examples`.