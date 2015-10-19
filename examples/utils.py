from __future__ import division, print_function, absolute_import
import subprocess
import time
import os
import sys
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
    ('Image Analysis', 'imageAnalysis.py'),
    ('Dock widgets', 'dockarea.py'),
    ('Console', 'ConsoleWidget.py'),
    ('Histograms', 'histogram.py'),
    ('Beeswarm plot', 'beeswarm.py'),
    ('Auto-range', 'PlotAutoRange.py'),
    ('Remote Plotting', 'RemoteSpeedTest.py'),
    ('Scrolling plots', 'scrollingPlots.py'),
    ('HDF5 big data', 'hdf5.py'),
    ('Demos', OrderedDict([
        ('Optics', 'optics_demos.py'),
        ('Special relativity', 'relativity_demo.py'),
        ('Verlet chain', 'verlet_chain_demo.py'),
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
        ('Region-of-Interest', 'ROIExamples.py'),
        ('Bar Graph', 'BarGraphItem.py'),
        ('GraphicsLayout', 'GraphicsLayout.py'),
        ('LegendItem', 'Legend.py'),
        ('Text Item', 'text.py'),
        ('Linked Views', 'linkedViews.py'),
        ('Arrow', 'Arrow.py'),
        ('ViewBox', 'ViewBox.py'),
        ('Custom Graphics', 'customGraphicsItem.py'),
        ('Labeled Graph', 'CustomGraphItem.py'),
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

def testFile(name, f, exe, lib, graphicsSystem=None):
    global path
    fn = os.path.join(path,f)
    #print "starting process: ", fn
    os.chdir(path)
    sys.stdout.write(name)
    sys.stdout.flush()

    import1 = "import %s" % lib if lib != '' else ''
    import2 = os.path.splitext(os.path.split(fn)[1])[0]
    graphicsSystem = '' if graphicsSystem is None else "pg.QtGui.QApplication.setGraphicsSystem('%s')" % graphicsSystem
    code = """
try:
    %s
    import initExample
    import pyqtgraph as pg
    %s
    import %s
    import sys
    print("test complete")
    sys.stdout.flush()
    import time
    while True:  ## run a little event loop
        pg.QtGui.QApplication.processEvents()
        time.sleep(0.01)
except:
    print("test failed")
    raise

""" % (import1, graphicsSystem, import2)

    if sys.platform.startswith('win'):
        process = subprocess.Popen([exe], stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        process.stdin.write(code.encode('UTF-8'))
        process.stdin.close()
    else:
        process = subprocess.Popen(['exec %s -i' % (exe)], shell=True, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        process.stdin.write(code.encode('UTF-8'))
        process.stdin.close() ##?
    output = ''
    fail = False
    while True:
        c = process.stdout.read(1).decode()
        output += c
        #sys.stdout.write(c)
        #sys.stdout.flush()
        if output.endswith('test complete'):
            break
        if output.endswith('test failed'):
            fail = True
            break
    time.sleep(1)
    process.kill()
    #res = process.communicate()
    res = (process.stdout.read(), process.stderr.read())

    if fail or 'exception' in res[1].decode().lower() or 'error' in res[1].decode().lower():
        print('.' * (50-len(name)) + 'FAILED')
        print(res[0].decode())
        print(res[1].decode())
    else:
        print('.' * (50-len(name)) + 'passed')
