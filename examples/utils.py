from collections import OrderedDict
from argparse import Namespace

examples = OrderedDict([
    ('Command-line usage', 'CLIexample.py'),
    ('Basic Plotting', Namespace(filename='Plotting.py', recommended=True)),
    ('ImageView', 'ImageView.py'),
    ('ParameterTree', 'parametertree.py'),
    ('Crosshair / Mouse interaction', 'crosshair.py'),
    ('Data Slicing', 'DataSlicing.py'),
    ('Plot Customization', 'customPlot.py'),
    ('Timestamps on x axis', 'DateAxisItem.py'),
    ('Image Analysis', 'imageAnalysis.py'),
    ('ViewBox Features', Namespace(filename='ViewBoxFeatures.py', recommended=True)),
    ('Dock widgets', 'dockarea.py'),
    ('Console', 'ConsoleWidget.py'),
    ('Histograms', 'histogram.py'),
    ('Beeswarm plot', 'beeswarm.py'),
    ('Symbols', 'Symbols.py'),
    ('Auto-range', 'PlotAutoRange.py'),
    ('Remote Plotting', 'RemoteSpeedTest.py'),
    ('Scrolling plots', 'scrollingPlots.py'),
    ('Palette adjustment','PaletteApplicationExample.py'),
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


# don't care about ordering
# but actually from Python 3.7, dict is ordered
others = dict([
    ('logAxis', 'logAxis.py'),
    ('PanningPlot', 'PanningPlot.py'),
    ('MultiplePlotAxes', 'MultiplePlotAxes.py'),
    ('ROItypes', 'ROItypes.py'),
    ('ScaleBar', 'ScaleBar.py'),
    ('InfiniteLine', 'InfiniteLine.py'),
    ('ViewBox', 'ViewBox.py'),
    ('GradientEditor', 'GradientEditor.py'),
    ('GLBarGraphItem', 'GLBarGraphItem.py'),
    ('GLViewWidget', 'GLViewWidget.py'),
    ('DiffTreeWidget', 'DiffTreeWidget.py'),
    ('MultiPlotWidget', 'MultiPlotWidget.py'),
    ('RemoteGraphicsView', 'RemoteGraphicsView.py'),
    ('colorMaps', 'colorMaps.py'),
    ('contextMenu', 'contextMenu.py'),
    ('designerExample', 'designerExample.py'),
    ('DateAxisItem_QtDesigner', 'DateAxisItem_QtDesigner.py'),
    ('GraphicsScene', 'GraphicsScene.py'),
    ('MouseSelection', 'MouseSelection.py'),
])


# examples that are subsumed into other examples
trivial = dict([
    ('SimplePlot', 'SimplePlot.py'),    # Plotting.py
    ('LogPlotTest', 'LogPlotTest.py'),  # Plotting.py
    ('ViewLimits', 'ViewLimits.py'),    # ViewBoxFeatures.py
])

# examples that are not suitable for CI testing
skiptest = dict([
    ('ProgressDialog', 'ProgressDialog.py'),    # modal dialog
])
