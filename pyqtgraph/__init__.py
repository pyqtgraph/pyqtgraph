# -*- coding: utf-8 -*-
"""
PyQtGraph - Scientific Graphics and GUI Library for Python
www.pyqtgraph.org
"""

__version__ = '0.11.1.dev0'

### import all the goodies and add some helper functions for easy CLI use

## 'Qt' is a local module; it is intended mainly to cover up the differences
## between PyQt4 and PySide.
from .Qt import QtGui, mkQApp

## not really safe--If we accidentally create another QApplication, the process hangs (and it is very difficult to trace the cause)
#if QtGui.QApplication.instance() is None:
    #app = QtGui.QApplication([])

import numpy  ## pyqtgraph requires numpy
              ## (import here to avoid massive error dump later on if numpy is not available)

import os, sys

## check python version
## Allow anything >= 2.7
if sys.version_info[0] < 2 or (sys.version_info[0] == 2 and sys.version_info[1] < 6):
    raise Exception("Pyqtgraph requires Python version 2.6 or greater (this is %d.%d)" % (sys.version_info[0], sys.version_info[1]))

## helpers for 2/3 compatibility
from . import python2_3

## in general openGL is poorly supported with Qt+GraphicsView.
## we only enable it where the performance benefit is critical.
## Note this only applies to 2D graphics; 3D graphics always use OpenGL.
if 'linux' in sys.platform:  ## linux has numerous bugs in opengl implementation
    useOpenGL = False
elif 'darwin' in sys.platform: ## openGL can have a major impact on mac, but also has serious bugs
    useOpenGL = False
    if QtGui.QApplication.instance() is not None:
        print('Warning: QApplication was created before pyqtgraph was imported; there may be problems.')
else:
    useOpenGL = False  ## on windows there's a more even performance / bugginess tradeoff. 
                
CONFIG_OPTIONS = {
    'useOpenGL': useOpenGL, ## by default, this is platform-dependent (see widgets/GraphicsView). Set to True or False to explicitly enable/disable opengl.
    'leftButtonPan': True,  ## if false, left button drags a rubber band for zooming in viewbox
    # foreground/background take any arguments to the 'mkColor' in /pyqtgraph/functions.py
    'foreground': 'd',  ## default foreground color for axes, labels, etc.
    'background': 'k',        ## default background for GraphicsWidget
    'antialias': False,
    'editorCommand': None,  ## command used to invoke code editor from ConsoleWidgets
    'useWeave': False,       ## Use weave to speed up some operations, if it is available
    'weaveDebug': False,    ## Print full error message if weave compile fails
    'exitCleanup': True,    ## Attempt to work around some exit crash bugs in PyQt and PySide
    'enableExperimental': False, ## Enable experimental features (the curious can search for this key in the code)
    'crashWarning': False,  # If True, print warnings about situations that may result in a crash
    'mouseRateLimit': 100,  # For ignoring frequent mouse events, max number of mouse move events per second, if <= 0, then it is switched off
    'imageAxisOrder': 'col-major',  # For 'row-major', image data is expected in the standard (row, col) order.
                                 # For 'col-major', image data is expected in reversed (col, row) order.
                                 # The default is 'col-major' for backward compatibility, but this may
                                 # change in the future.
    'useCupy': False,  # When True, attempt to use cupy ( currently only with ImageItem and related functions )
} 


def setConfigOption(opt, value):
    if opt not in CONFIG_OPTIONS:
        raise KeyError('Unknown configuration option "%s"' % opt)
    if opt == 'imageAxisOrder' and value not in ('row-major', 'col-major'):
        raise ValueError('imageAxisOrder must be either "row-major" or "col-major"')
    CONFIG_OPTIONS[opt] = value

def setConfigOptions(**opts):
    """Set global configuration options. 
    
    Each keyword argument sets one global option. 
    """
    for k,v in opts.items():
        setConfigOption(k, v)

def getConfigOption(opt):
    """Return the value of a single global configuration option.
    """
    return CONFIG_OPTIONS[opt]


def systemInfo():
    print("sys.platform: %s" % sys.platform)
    print("sys.version: %s" % sys.version)
    from .Qt import VERSION_INFO
    print("qt bindings: %s" % VERSION_INFO)
    
    global __version__
    rev = None
    if __version__ is None:  ## this code was probably checked out from bzr; look up the last-revision file
        lastRevFile = os.path.join(os.path.dirname(__file__), '..', '.bzr', 'branch', 'last-revision')
        if os.path.exists(lastRevFile):
            with open(lastRevFile, 'r') as fd:
                rev = fd.read().strip()
    
    print("pyqtgraph: %s; %s" % (__version__, rev))
    print("config:")
    import pprint
    pprint.pprint(CONFIG_OPTIONS)

## Rename orphaned .pyc files. This is *probably* safe :)
## We only do this if __version__ is None, indicating the code was probably pulled
## from the repository. 
def renamePyc(startDir):
    ### Used to rename orphaned .pyc files
    ### When a python file changes its location in the repository, usually the .pyc file
    ### is left behind, possibly causing mysterious and difficult to track bugs. 

    ### Note that this is no longer necessary for python 3.2; from PEP 3147:
    ### "If the py source file is missing, the pyc file inside __pycache__ will be ignored. 
    ### This eliminates the problem of accidental stale pyc file imports."
    
    printed = False
    startDir = os.path.abspath(startDir)
    for path, dirs, files in os.walk(startDir):
        if '__pycache__' in path:
            continue
        for f in files:
            fileName = os.path.join(path, f)
            base, ext = os.path.splitext(fileName)
            py = base + ".py"
            if ext == '.pyc' and not os.path.isfile(py):
                if not printed:
                    print("NOTE: Renaming orphaned .pyc files:")
                    printed = True
                n = 1
                while True:
                    name2 = fileName + ".renamed%d" % n
                    if not os.path.exists(name2):
                        break
                    n += 1
                print("  " + fileName + "  ==>")
                print("  " + name2)
                os.rename(fileName, name2)
                
path = os.path.split(__file__)[0]
if __version__ is None and not hasattr(sys, 'frozen') and sys.version_info[0] == 2: ## If we are frozen, there's a good chance we don't have the original .py files anymore.
    renamePyc(path)


## Import almost everything to make it available from a single namespace
## don't import the more complex systems--canvas, parametertree, flowchart, dockarea
## these must be imported separately.
#from . import frozenSupport
#def importModules(path, globals, locals, excludes=()):
    #"""Import all modules residing within *path*, return a dict of name: module pairs.
    
    #Note that *path* MUST be relative to the module doing the import.    
    #"""
    #d = os.path.join(os.path.split(globals['__file__'])[0], path)
    #files = set()
    #for f in frozenSupport.listdir(d):
        #if frozenSupport.isdir(os.path.join(d, f)) and f not in ['__pycache__', 'tests']:
            #files.add(f)
        #elif f[-3:] == '.py' and f != '__init__.py':
            #files.add(f[:-3])
        #elif f[-4:] == '.pyc' and f != '__init__.pyc':
            #files.add(f[:-4])
        
    #mods = {}
    #path = path.replace(os.sep, '.')
    #for modName in files:
        #if modName in excludes:
            #continue
        #try:
            #if len(path) > 0:
                #modName = path + '.' + modName
            #print( "from .%s import * " % modName)
            #mod = __import__(modName, globals, locals, ['*'], 1)
            #mods[modName] = mod
        #except:
            #import traceback
            #traceback.print_stack()
            #sys.excepthook(*sys.exc_info())
            #print("[Error importing module: %s]" % modName)
            
    #return mods

#def importAll(path, globals, locals, excludes=()):
    #"""Given a list of modules, import all names from each module into the global namespace."""
    #mods = importModules(path, globals, locals, excludes)
    #for mod in mods.values():
        #if hasattr(mod, '__all__'):
            #names = mod.__all__
        #else:
            #names = [n for n in dir(mod) if n[0] != '_']
        #for k in names:
            #if hasattr(mod, k):
                #globals[k] = getattr(mod, k)

# Dynamic imports are disabled. This causes too many problems.
#importAll('graphicsItems', globals(), locals())
#importAll('widgets', globals(), locals(),
          #excludes=['MatplotlibWidget', 'RawImageWidget', 'RemoteGraphicsView'])

from .graphicsItems.VTickGroup import * 
from .graphicsItems.GraphicsWidget import * 
from .graphicsItems.ScaleBar import * 
from .graphicsItems.PlotDataItem import * 
from .graphicsItems.GraphItem import * 
from .graphicsItems.TextItem import * 
from .graphicsItems.GraphicsLayout import * 
from .graphicsItems.UIGraphicsItem import * 
from .graphicsItems.GraphicsObject import * 
from .graphicsItems.PlotItem import * 
from .graphicsItems.ROI import * 
from .graphicsItems.InfiniteLine import * 
from .graphicsItems.HistogramLUTItem import * 
from .graphicsItems.GridItem import * 
from .graphicsItems.GradientLegend import * 
from .graphicsItems.GraphicsItem import * 
from .graphicsItems.BarGraphItem import * 
from .graphicsItems.ViewBox import * 
from .graphicsItems.ArrowItem import * 
from .graphicsItems.ImageItem import * 
from .graphicsItems.PColorMeshItem import * 
from .graphicsItems.AxisItem import * 
from .graphicsItems.DateAxisItem import *
from .graphicsItems.LabelItem import * 
from .graphicsItems.CurvePoint import * 
from .graphicsItems.GraphicsWidgetAnchor import * 
from .graphicsItems.PlotCurveItem import * 
from .graphicsItems.ButtonItem import * 
from .graphicsItems.GradientEditorItem import * 
from .graphicsItems.MultiPlotItem import * 
from .graphicsItems.ErrorBarItem import * 
from .graphicsItems.IsocurveItem import * 
from .graphicsItems.LinearRegionItem import * 
from .graphicsItems.FillBetweenItem import * 
from .graphicsItems.LegendItem import * 
from .graphicsItems.ScatterPlotItem import * 
from .graphicsItems.ItemGroup import * 

from .widgets.MultiPlotWidget import * 
from .widgets.ScatterPlotWidget import * 
from .widgets.ColorMapWidget import * 
from .widgets.FileDialog import * 
from .widgets.ValueLabel import * 
from .widgets.HistogramLUTWidget import * 
from .widgets.CheckTable import * 
from .widgets.BusyCursor import * 
from .widgets.PlotWidget import * 
from .widgets.ComboBox import * 
from .widgets.GradientWidget import * 
from .widgets.DataFilterWidget import * 
from .widgets.SpinBox import * 
from .widgets.JoystickButton import * 
from .widgets.GraphicsLayoutWidget import * 
from .widgets.TreeWidget import * 
from .widgets.PathButton import * 
from .widgets.VerticalLabel import * 
from .widgets.FeedbackButton import * 
from .widgets.ColorButton import * 
from .widgets.DataTreeWidget import * 
from .widgets.DiffTreeWidget import * 
from .widgets.GraphicsView import * 
from .widgets.LayoutWidget import * 
from .widgets.TableWidget import * 
from .widgets.ProgressDialog import *
from .widgets.GroupBox import GroupBox
from .widgets.RemoteGraphicsView import RemoteGraphicsView

from .imageview import *
from .WidgetGroup import *
from .Point import Point
from .Vector import Vector
from .SRTTransform import SRTTransform
from .Transform3D import Transform3D
from .SRTTransform3D import SRTTransform3D
from .functions import *
from .graphicsWindows import *
from .SignalProxy import *
from .colormap import *
from .ptime import time
from .Qt import isQObjectAlive
from .ThreadsafeTimer import *


##############################################################
## PyQt and PySide both are prone to crashing on exit. 
## There are two general approaches to dealing with this:
##  1. Install atexit handlers that assist in tearing down to avoid crashes.
##     This helps, but is never perfect.
##  2. Terminate the process before python starts tearing down
##     This is potentially dangerous

## Attempts to work around exit crashes:
import atexit
_cleanupCalled = False
def cleanup():
    global _cleanupCalled
    if _cleanupCalled:
        return
    
    if not getConfigOption('exitCleanup'):
        return
    
    ViewBox.quit()  ## tell ViewBox that it doesn't need to deregister views anymore.
    
    ## Workaround for Qt exit crash:
    ## ALL QGraphicsItems must have a scene before they are deleted.
    ## This is potentially very expensive, but preferred over crashing.
    ## Note: this appears to be fixed in PySide as of 2012.12, but it should be left in for a while longer..
    app = QtGui.QApplication.instance()
    if app is None or not isinstance(app, QtGui.QApplication):
        # app was never constructed is already deleted or is an
        # QCoreApplication/QGuiApplication and not a full QApplication
        return
    import gc
    s = QtGui.QGraphicsScene()
    for o in gc.get_objects():
        try:
            if isinstance(o, QtGui.QGraphicsItem) and isQObjectAlive(o) and o.scene() is None:
                if getConfigOption('crashWarning'):
                    sys.stderr.write('Error: graphics item without scene. '
                        'Make sure ViewBox.close() and GraphicsView.close() '
                        'are properly called before app shutdown (%s)\n' % (o,))
                
                s.addItem(o)
        except (RuntimeError, ReferenceError):  ## occurs if a python wrapper no longer has its underlying C++ object
            continue
    _cleanupCalled = True

atexit.register(cleanup)

# Call cleanup when QApplication quits. This is necessary because sometimes
# the QApplication will quit before the atexit callbacks are invoked.
# Note: cannot connect this function until QApplication has been created, so
# instead we have GraphicsView.__init__ call this for us.
_cleanupConnected = False
def _connectCleanup():
    global _cleanupConnected
    if _cleanupConnected:
        return
    QtGui.QApplication.instance().aboutToQuit.connect(cleanup)
    _cleanupConnected = True


## Optional function for exiting immediately (with some manual teardown)
def exit():
    """
    Causes python to exit without garbage-collecting any objects, and thus avoids
    calling object destructor methods. This is a sledgehammer workaround for 
    a variety of bugs in PyQt and Pyside that cause crashes on exit.
    
    This function does the following in an attempt to 'safely' terminate
    the process:
    
    * Invoke atexit callbacks
    * Close all open file handles
    * os._exit()
    
    Note: there is some potential for causing damage with this function if you
    are using objects that _require_ their destructors to be called (for example,
    to properly terminate log files, disconnect from devices, etc). Situations
    like this are probably quite rare, but use at your own risk.
    """
    
    ## first disable our own cleanup function; won't be needing it.
    setConfigOptions(exitCleanup=False)
    
    ## invoke atexit callbacks
    atexit._run_exitfuncs()
    
    ## close file handles
    if sys.platform == 'darwin':
        for fd in range(3, 4096):
            if fd in [7]:  # trying to close 7 produces an illegal instruction on the Mac.
                continue
            try:
                os.close(fd)
            except OSError:
                pass
    else:
        os.closerange(3, 4096) ## just guessing on the maximum descriptor count..

    os._exit(0)
    

## Convenience functions for command-line use
plots = []
images = []
QAPP = None

def plot(*args, **kargs):
    """
    Create and return a :class:`PlotWidget <pyqtgraph.PlotWinPlotWidgetdow>` 
    Accepts a *title* argument to set the title of the window.
    All other arguments are used to plot data. (see :func:`PlotItem.plot() <pyqtgraph.PlotItem.plot>`)
    """
    mkQApp()
    pwArgList = ['title', 'labels', 'name', 'left', 'right', 'top', 'bottom', 'background']
    pwArgs = {}
    dataArgs = {}
    for k in kargs:
        if k in pwArgList:
            pwArgs[k] = kargs[k]
        else:
            dataArgs[k] = kargs[k]
    windowTitle = pwArgs.pop("title", "PlotWidget")
    w = PlotWidget(**pwArgs)
    w.setWindowTitle(windowTitle)
    if len(args) > 0 or len(dataArgs) > 0:
        w.plot(*args, **dataArgs)
    plots.append(w)
    w.show()
    return w

def image(*args, **kargs):
    """
    Create and return an :class:`ImageView <pyqtgraph.ImageView>` 
    Will show 2D or 3D image data.
    Accepts a *title* argument to set the title of the window.
    All other arguments are used to show data. (see :func:`ImageView.setImage() <pyqtgraph.ImageView.setImage>`)
    """
    mkQApp()
    w = ImageView()
    windowTitle = kargs.pop("title", "ImageView")
    w.setWindowTitle(windowTitle)
    w.setImage(*args, **kargs)
    images.append(w)
    w.show()
    return w
show = image  ## for backward compatibility


def dbg(*args, **kwds):
    """
    Create a console window and begin watching for exceptions.
    
    All arguments are passed to :func:`ConsoleWidget.__init__() <pyqtgraph.console.ConsoleWidget.__init__>`.
    """
    mkQApp()
    from . import console
    c = console.ConsoleWidget(*args, **kwds)
    c.catchAllExceptions()
    c.show()
    global consoles
    try:
        consoles.append(c)
    except NameError:
        consoles = [c]
    return c


def stack(*args, **kwds):
    """
    Create a console window and show the current stack trace.
    
    All arguments are passed to :func:`ConsoleWidget.__init__() <pyqtgraph.console.ConsoleWidget.__init__>`.
    """
    mkQApp()
    from . import console
    c = console.ConsoleWidget(*args, **kwds)
    c.setStack()
    c.show()
    global consoles
    try:
        consoles.append(c)
    except NameError:
        consoles = [c]
    return c
