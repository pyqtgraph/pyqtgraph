import sys, traceback
from ..Qt import QtWidgets, QtGui


class StackWidget(QtWidgets.QTreeWidget):
    def __init__(self, parent=None):
        QtWidgets.QTreeWidget.__init__(self, parent)
        self.setAlternatingRowColors(True)
        self.setHeaderHidden(True)

    def selectedFrame(self):
        """Return the currently selected stack frame (or None if there is no selection)
        """
        sel = self.selectedItems()
        if len(sel) == 0:
            return None
        else:
            return sel[0].frame

    def clear(self):
        QtWidgets.QTreeWidget.clear(self)
        self.frames = []

    def setException(self, exc=None, lastFrame=None):
        """Display an exception chain with its tracebacks and call stack.
        """
        if exc is None:
            exc = sys.exc_info()[1]

        self.clear()

        exceptions = exceptionChain(exc)
        for ex, cause in exceptions:
            stackFrames, tbFrames = stacksFromTraceback(ex.__traceback__, lastFrame=lastFrame)
            catchMsg = textItem("Exception caught here")
            excStr = ''.join(traceback.format_exception_only(type(ex), ex)).strip()
            items = makeItemTree(stackFrames + [catchMsg] + tbFrames, excStr)
            self.addTopLevelItem(items[0])
            if cause is not None:
                if cause == 'cause':
                    causeItem = textItem("The above exception was the direct cause of the following exception:")
                elif cause == 'context':
                    causeItem = textItem("During handling of the above exception, another exception occurred:")
                self.addTopLevelItem(causeItem)

        items[0].setExpanded(True)

    def setStack(self, frame=None, expand=True, lastFrame=None):
        """Display a call stack and exception traceback.

        This allows the user to probe the contents of any frame in the given stack.

        *frame* may either be a Frame instance or None, in which case the current 
        frame is retrieved from ``sys._getframe()``. 

        If *tb* is provided then the frames in the traceback will be appended to 
        the end of the stack list. If *tb* is None, then sys.exc_info() will 
        be checked instead.
        """
        if frame is None:
            frame = sys._getframe().f_back

        self.clear()

        stack = stackFromFrame(frame, lastFrame=lastFrame)
        items = makeItemTree(stack, "Call stack")
        self.addTopLevelItem(items[0])
        if expand:
            items[0].setExpanded(True)

    
def stackFromFrame(frame, lastFrame=None):
    """Return (text, stack_frame) for the entire stack ending at *frame*

    If *lastFrame* is given and present in the stack, then the stack is truncated 
    at that frame.
    """
    lines = traceback.format_stack(frame)
    frames = []
    while frame is not None:
        frames.insert(0, frame)
        frame = frame.f_back
    if lastFrame is not None and lastFrame in frames:
        frames = frames[:frames.index(lastFrame)+1]
        
    return list(zip(lines[:len(frames)], frames))


def stacksFromTraceback(tb, lastFrame=None):
    """Return (text, stack_frame) for a traceback and the stack preceding it

    If *lastFrame* is given and present in the stack, then the stack is truncated 
    at that frame.
    """
    # get stack before tb
    stack = stackFromFrame(tb.tb_frame.f_back if tb is not None else lastFrame)
    if tb is None:
        return stack, []

    # walk to last frame of traceback        
    lines = traceback.format_tb(tb)
    frames = []
    while True:            
        frames.append(tb.tb_frame)
        if tb.tb_next is None or tb.tb_frame is lastFrame:
            break
        tb = tb.tb_next

    return stack, list(zip(lines[:len(frames)], frames))


def makeItemTree(stack, title):
    topItem = QtWidgets.QTreeWidgetItem([title])
    topItem.frame = None
    font = topItem.font(0)
    font.setWeight(font.Weight.Bold)
    topItem.setFont(0, font)
    items = [topItem]
    for entry in stack:
        if isinstance(entry, QtWidgets.QTreeWidgetItem):
            item = entry
        else:
            text, frame = entry
            item = QtWidgets.QTreeWidgetItem([text.rstrip()])
            item.frame = frame
        topItem.addChild(item)
        items.append(item)
    return items


def exceptionChain(exc):
    """Return a list of (exception, 'cause'|'context') pairs for exceptions
    leading up to *exc*
    """
    exceptions = [(exc, None)]
    while True:
        # walk through chained exceptions
        if exc.__cause__ is not None:
            exc = exc.__cause__
            exceptions.insert(0, (exc, 'cause'))
        elif exc.__context__ is not None and exc.__suppress_context__ is False:
            exc = exc.__context__
            exceptions.insert(0, (exc, 'context'))
        else:
            break
    return exceptions


def textItem(text):
    """Return a tree item with no associated stack frame and a darker background color
    """
    item = QtWidgets.QTreeWidgetItem([text])
    item.frame = None
    item.setBackground(0, QtGui.QBrush(QtGui.QColor(220, 220, 220)))
    item.setForeground(0, QtGui.QBrush(QtGui.QColor(0, 0, 0)))
    item.setChildIndicatorPolicy(item.ChildIndicatorPolicy.DontShowIndicator)
    return item
