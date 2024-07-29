import numpy as np

from ... import functions as fn
from ...Qt import QtGui
from ..GLGraphicsItem import GLGraphicsItem
from .GLLinePlotItem import GLLinePlotItem

__all__ = ['GLBoxItem']

class GLBoxItem(GLGraphicsItem):
    """
    **Bases:** :class:`GLGraphicsItem <pyqtgraph.opengl.GLGraphicsItem>`
    
    Displays a wire-frame box.
    """
    def __init__(self, size=None, color=None, glOptions='translucent', parentItem=None):
        super().__init__()

        self.lineplot = None    # mark that we are still initializing

        if size is None:
            size = QtGui.QVector3D(1,1,1)
        self.setSize(size=size)
        if color is None:
            color = (255,255,255,80)
        self.setColor(color)

        self.lineplot = GLLinePlotItem(
            parentItem=self, glOptions=glOptions, mode='lines'
        )
        self.setParentItem(parentItem)
        self.updateLines()
    
    def setSize(self, x=None, y=None, z=None, size=None):
        """
        Set the size of the box (in its local coordinate system; this does not affect the transform)
        Arguments can be x,y,z or size=QVector3D().
        """
        if size is not None:
            x = size.x()
            y = size.y()
            z = size.z()
        self.__size = [x,y,z]
        self.updateLines()
        
    def size(self):
        return self.__size[:]
    
    def setColor(self, *args):
        """Set the color of the box. Arguments are the same as those accepted by functions.mkColor()"""
        self.__color = fn.mkColor(*args)
        self.updateLines()
        
    def color(self):
        return self.__color

    def updateLines(self):
        if self.lineplot is None:
            # still initializing
            return

        x,y,z = self.size()
        pos = np.array([
            [0, 0, 0],
            [0, 0, z],
            [x, 0, 0],
            [x, 0, z],
            [0, y, 0],
            [0, y, z],
            [x, y, 0],
            [x, y, z],

            [0, 0, 0],
            [0, y, 0],
            [x, 0, 0],
            [x, y, 0],
            [0, 0, z],
            [0, y, z],
            [x, 0, z],
            [x, y, z],
        
            [0, 0, 0],
            [x, 0, 0],
            [0, y, 0],
            [x, y, 0],
            [0, 0, z],
            [x, 0, z],
            [0, y, z],
            [x, y, z],
        ], dtype=np.float32)
        
        color = self.color().getRgbF()
        self.lineplot.setData(pos=pos, color=color)
        self.update()
