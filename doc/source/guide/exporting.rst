Exporting
=========

PyQtGraph provides a variety of export formats for all 2D graphics. For 3D graphics, see `Exporting 3D Graphics`_ below.

Exporting from the GUI
----------------------

Any 2D graphics can be exported by right-clicking on the graphic, then selecting 'export' from the context menu. 
This will display the export dialog in which the user must:

#. Select an item (or the entire scene) to export. Selecting an item will cause the item to be hilighted in the original 
   graphic window (but this hilight will not be displayed in the exported file). 
#. Select an export format.
#. Change any desired export options.
#. Click the 'export' button.

Export Formats
--------------

* Image - PNG is the default format. The exact set of image formats supported will depend on your Qt libraries. However, 
  common formats such as PNG, JPG, and TIFF are almost always available. 
* SVG - Graphics exported as SVG are targeted to work as well as possible with both Inkscape and 
  Adobe Illustrator. For high quality SVG export, please use PyQtGraph version 0.9.3 or later.
  This is the preferred method for generating publication graphics from PyQtGraph.
* CSV - Exports plotted data as CSV. This exporter _only_ works if a PlotItem is selected for export.
* Matplotlib - This exporter opens a new window and attempts to re-plot the
  data using matplotlib (if available). Note that some graphic features are either not implemented
  for this exporter or not available in matplotlib. This exporter _only_ works if a PlotItem is selected
  for export.
* Printer - Exports to the operating system's printing service. This exporter is provided for completeness, 
  but is not well supported due to problems with Qt's printing system.



Exporting from the API
----------------------

To export a file programatically, follow this example::

    import pyqtgraph as pg
    
    # generate something to export
    plt = pg.plot([1,5,2,4,3])

    # create an exporter instance, as an argument give it
    # the item you wish to export
    exporter = pg.exporters.ImageExporter.ImageExporter(plt.plotItem)

    # set export parameters if needed
    exporter.parameters()['width'] = 100   # (note this also affects height parameter)
    
    # save to file
    exporter.export('fileName.png')
    

Exporting 3D Graphics
---------------------

The exporting functionality described above is not yet available for 3D graphics. However, it is possible to 
generate an image from a GLViewWidget by using QGLWidget.grabFrameBuffer or QGLWidget.renderPixmap::

    glview.grabFrameBuffer().save('fileName.png')

See the Qt documentation for more information. 

    
