from __future__ import division, print_function, absolute_import
import os
from pyqtgraph.pgcollections import OrderedDict
from pyqtgraph.python2_3 import basestring

path = os.path.abspath(os.path.dirname(__file__))


examples = OrderedDict([
    ('Command-line usage', 'CLIexample.py'),
    ('Basic Plotting', 'Plotting.py'),
    ('ImageView', 'ImageView.py'),
    ('ParameterTree', 'parametertree.py'),
    ('Crosshair / Mouse interaction', 'crosshair.py'),
    ('Data Slicing', 'DataSlicing.py'),
    ('Plot Customization', 'customPlot.py'),
    ('Timestamps on x axis', 'DateAxisItem.py'),
    ('Image Analysis', 'imageAnalysis.py'),
    ('ViewBox Features', 'ViewBoxFeatures.py'),
    ('Dock widgets', 'dockarea.py'),
    ('Console', 'ConsoleWidget.py'),
    ('Histograms', 'histogram.py'),
    ('Beeswarm plot', 'beeswarm.py'),
    ('Symbols', 'Symbols.py'),
    ('Auto-range', 'PlotAutoRange.py'),
    ('Remote Plotting', 'RemoteSpeedTest.py'),
    ('Scrolling plots', 'scrollingPlots.py'),
    ('HDF5 big data', 'hdf5.py'),
    ('Demos', OrderedDict([
        ('Optics', 'optics_demos.py'),
        ('Special relativity', 'relativity_demo.py'),
        ('Verlet chain', 'verlet_chain_demo.py'),
        ('Koch Fractal', 'fractal.py'),
    ])),
    ('GraphicsItems', OrderedDict([
        ('Scatter Plot', 'ScatterPlot.py'),
        #('PlotItem', 'PlotItem.py'),
        ('IsocurveItem', 'isocurve.py'),
        ('GraphItem', 'GraphItem.py'),
        ('ErrorBarItem', 'ErrorBarItem.py'),
        ('FillBetweenItem', 'FillBetweenItem.py'),
        ('ImageItem - video', 'ImageItem.py'),
        ('ImageItem - draw', 'Draw.py'),
        ('Non-uniform Image', 'NonUniformImage.py'),
        ('Region-of-Interest', 'ROIExamples.py'),
        ('Bar Graph', 'BarGraphItem.py'),
        ('GraphicsLayout', 'GraphicsLayout.py'),
        ('LegendItem', 'Legend.py'),
        ('Text Item', 'text.py'),
        ('Linked Views', 'linkedViews.py'),
        ('Arrow', 'Arrow.py'),
        ('ViewBox', 'ViewBoxFeatures.py'),
        ('Custom Graphics', 'customGraphicsItem.py'),
        ('Labeled Graph', 'CustomGraphItem.py'),
        ('PColorMeshItem', 'PColorMeshItem.py'),
    ])),
    ('Benchmarks', OrderedDict([
        ('Video speed test', 'VideoSpeedTest.py'),
        ('Line Plot update', 'PlotSpeedTest.py'),
        ('Scatter Plot update', 'ScatterPlotSpeedTest.py'),
        ('Multiple plots', 'MultiPlotSpeedTest.py'),
    ])),
    ('3D Graphics', OrderedDict([
        ('Volumetric', 'GLVolumeItem.py'),
        ('Isosurface', 'GLIsosurface.py'),
        ('Surface Plot', 'GLSurfacePlot.py'),
        ('Scatter Plot', 'GLScatterPlotItem.py'),
        ('Shaders', 'GLshaders.py'),
        ('Line Plot', 'GLLinePlotItem.py'),
        ('Mesh', 'GLMeshItem.py'),
        ('Image', 'GLImageItem.py'),
    ])),
    ('Widgets', OrderedDict([
        ('PlotWidget', 'PlotWidget.py'),
        ('SpinBox', 'SpinBox.py'),
        ('ConsoleWidget', 'ConsoleWidget.py'),
        ('Histogram / lookup table', 'HistogramLUT.py'),
        ('TreeWidget', 'TreeWidget.py'),
        ('ScatterPlotWidget', 'ScatterPlotWidget.py'),
        ('DataTreeWidget', 'DataTreeWidget.py'),
        ('GradientWidget', 'GradientWidget.py'),
        ('TableWidget', 'TableWidget.py'),
        ('ColorButton', 'ColorButton.py'),
        #('CheckTable', '../widgets/CheckTable.py'),
        #('VerticalLabel', '../widgets/VerticalLabel.py'),
        ('JoystickButton', 'JoystickButton.py'),
    ])),
    ('Flowcharts', 'Flowchart.py'),
    ('Custom Flowchart Nodes', 'FlowchartCustomNode.py'),
])


def buildFileList(examples, files=None):
    if files == None:
        files = []
    for key, val in examples.items():
        #item = QtGui.QTreeWidgetItem([key])
        if isinstance(val, basestring):
            #item.file = val
            files.append((key,val))
        else:
            buildFileList(val, files)
    return files
