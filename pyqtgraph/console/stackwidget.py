import sys, traceback
from ..Qt import QtWidgets, QtGui


class StackWidget(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        QtWidgets.QListWidget.__init__(self, parent)
        self.setAlternatingRowColors(True)
        self.frames = []

    def selectedFrame(self):
        """Return the currently selected stack frame (or None if there is no selection)
        """
        index = self.currentRow()
        if index >= 0 and index < len(self.frames):
            return self.frames[index]
        else:
            return None
        
    def clear(self):
        QtWidgets.QListWidget.clear(self)
        self.frames = []

    def setStack(self, frame=None, tb=None):
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

        if tb is None:
            tb = sys.exc_info()[2]

        self.clear()

        # Build stack up to this point
        for index, line in enumerate(traceback.extract_stack(frame)):
            # extract_stack return value changed in python 3.5
            if 'FrameSummary' in str(type(line)):
                line = (line.filename, line.lineno, line.name, line._line.strip())
            
            self.addItem('File "%s", line %s, in %s()\n  %s' % line)
        while frame is not None:
            self.frames.insert(0, frame)
            frame = frame.f_back

        if tb is None:
            return

        self.addItem('-- exception caught here: --')
        item = self.item(self.count()-1)
        item.setBackground(QtGui.QBrush(QtGui.QColor(200, 200, 200)))
        item.setForeground(QtGui.QBrush(QtGui.QColor(50, 50, 50)))
        self.frames.append(None)

        # And finish the rest of the stack up to the exception
        for index, line in enumerate(traceback.extract_tb(tb)):
            # extract_stack return value changed in python 3.5
            if 'FrameSummary' in str(type(line)):
                line = (line.filename, line.lineno, line.name, line._line.strip())
            
            self.addItem('File "%s", line %s, in %s()\n  %s' % line)
        while tb is not None:
            self.frames.append(tb.tb_frame)
            tb = tb.tb_next
