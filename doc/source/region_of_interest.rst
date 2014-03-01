Interactive Data Selection Controls
===================================

PyQtGraph includes graphics items which allow the user to select and mark regions of data.

Linear Selection and Marking
----------------------------

Two classes allow marking and selecting 1-dimensional data: :class:`LinearRegionItem <pyqtgraph.LinearRegionItem>` and :class:`InfiniteLine <pyqtgraph.InfiniteLine>`. The first class, :class:`LinearRegionItem <pyqtgraph.LinearRegionItem>`, may be added to any ViewBox or PlotItem to mark either a horizontal or vertical region. The region can be dragged and its bounding edges can be moved independently. The second class, :class:`InfiniteLine <pyqtgraph.InfiniteLine>`, is usually used to mark a specific position along the x or y axis. These may be dragged by the user.


2D Selection and Marking
------------------------

To select a 2D region from an image, pyqtgraph uses the :class:`ROI <pyqtgraph.ROI>` class or any of its subclasses. By default, :class:`ROI <pyqtgraph.ROI>` simply displays a rectangle which can be moved by the user to mark a specific region (most often this will be a region of an image, but this is not required). To allow the ROI to be resized or rotated, there are several methods for adding handles (:func:`addScaleHandle <pyqtgraph.ROI.addScaleHandle>`, :func:`addRotateHandle <pyqtgraph.ROI.addRotateHandle>`, etc.) which can be dragged by the user. These handles may be placed at any location relative to the ROI and may scale/rotate the ROI around any arbitrary center point. There are several ROI subclasses with a variety of shapes and modes of interaction.

To automatically extract a region of image data using an ROI and an ImageItem, use :func:`ROI.getArrayRegion <pyqtgraph.ROI.getArrayRegion>`. ROI classes use the :func:`affineSlice <pyqtgraph.affineSlice>` function to perform this extraction.

ROI can also be used as a control for moving/rotating/scaling items in a scene similar to most vetctor graphics editing applications.

See the ROITypes example for more information.


