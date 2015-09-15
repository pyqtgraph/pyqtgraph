"""
PyQt/PySide stress test:

Create lots of random widgets and graphics items, connect them together randomly,
the tear them down repeatedly.

The purpose of this is to attempt to generate segmentation faults.
"""
from pyqtgraph.Qt import QtTest
import pyqtgraph as pg
from random import seed, randint
import sys, gc, weakref

app = pg.mkQApp()

seed(12345)

widgetTypes = [
    pg.PlotWidget,
    pg.ImageView,
    pg.GraphicsView,
    pg.QtGui.QWidget,
    pg.QtGui.QTreeWidget,
    pg.QtGui.QPushButton,
    ]

itemTypes = [
    pg.PlotCurveItem,
    pg.ImageItem,
    pg.PlotDataItem,
    pg.ViewBox,
    pg.QtGui.QGraphicsRectItem
    ]

widgets = []
items = []
allWidgets = weakref.WeakSet()


def crashtest():
    global allWidgets
    try:
        gc.disable()
        actions = [
                createWidget,
                #setParent,
                forgetWidget,
                showWidget,
                processEvents,
                #raiseException,
                #addReference,
                ]

        thread = WorkThread()
        thread.start()

        while True:
            try:
                action = randItem(actions)
                action()
                print('[%d widgets alive, %d zombie]' % (len(allWidgets), len(allWidgets) - len(widgets)))
            except KeyboardInterrupt:
                print("Caught interrupt; send another to exit.")
                try:
                    for i in range(100):
                        QtTest.QTest.qWait(100)
                except KeyboardInterrupt:
                    thread.terminate()
                    break
            except:
                sys.excepthook(*sys.exc_info())
    finally:
        gc.enable()



class WorkThread(pg.QtCore.QThread):
    '''Intended to give the gc an opportunity to run from a non-gui thread.'''
    def run(self):
        i = 0
        while True:
            i += 1
            if (i % 1000000) == 0:
                print('--worker--')


def randItem(items):
    return items[randint(0, len(items)-1)]

def p(msg):
    print(msg)
    sys.stdout.flush()

def createWidget():
    p('create widget')
    global widgets, allWidgets
    if len(widgets) > 50:
        return
    widget = randItem(widgetTypes)()
    widget.setWindowTitle(widget.__class__.__name__)
    widgets.append(widget)
    allWidgets.add(widget)
    p("    %s" % widget)
    return widget

def setParent():
    p('set parent')
    global widgets
    if len(widgets) < 2:
        return
    child = parent = None
    while child is parent:
        child = randItem(widgets)
        parent = randItem(widgets)
    p("    %s parent of %s" % (parent, child))
    child.setParent(parent)

def forgetWidget():
    p('forget widget')
    global widgets
    if len(widgets) < 1:
        return
    widget = randItem(widgets)
    p('    %s' % widget)
    widgets.remove(widget)

def showWidget():
    p('show widget')
    global widgets
    if len(widgets) < 1:
        return
    widget = randItem(widgets)
    p('    %s' % widget)
    widget.show()

def processEvents():
    p('process events')
    QtTest.QTest.qWait(25)

class TstException(Exception):
    pass

def raiseException():
    p('raise exception')
    raise TstException("A test exception")

def addReference():
    p('add reference')
    global widgets
    if len(widgets) < 1:
        return
    obj1 = randItem(widgets)
    obj2 = randItem(widgets)
    p('    %s -> %s' % (obj1, obj2))
    obj1._testref = obj2



if __name__ == '__main__':
    test_stability()
