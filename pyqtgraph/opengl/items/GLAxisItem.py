import numpy as np

from ...Qt import QtGui
from ..GLGraphicsItem import GLGraphicsItem
from .GLLinePlotItem import GLLinePlotItem

__all__ = ['GLAxisItem']

class GLAxisItem(GLGraphicsItem):
    """
    **Bases:** :class:`GLGraphicsItem <pyqtgraph.opengl.GLGraphicsItem.GLGraphicsItem>`
    
    Displays three lines indicating origin and orientation of local coordinate system. 
    
    """
    
    def __init__(self, size=None, antialias=True, glOptions='translucent', parentItem=None):
        super().__init__()

        self.lineplot = None    # mark that we are still initializing

        if size is None:
            size = QtGui.QVector3D(1,1,1)
        self.setSize(size=size)

        self.lineplot = GLLinePlotItem(
            parentItem=self, glOptions=glOptions, mode='lines', antialias=antialias
        )
        self.setParentItem(parentItem)
        self.updateLines()

    def setSize(self, x=None, y=None, z=None, size=None):
        """
        Set the size of the axes (in its local coordinate system; this does not affect the transform)
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
    
    def updateLines(self):
        if self.lineplot is None:
            # still initializing
            return

        x,y,z = self.size()

        pos = np.array([
            [0, 0, 0, 0, 0, z],
            [0, 0, 0, 0, y, 0],
            [0, 0, 0, x, 0, 0],
        ], dtype=np.float32).reshape((-1, 3))

        color = np.array([
            [0, 1, 0, 0.6],     # z is green
            [1, 1, 0, 0.6],     # y is yellow
            [0, 0, 1, 0.6],     # x is blue
        ], dtype=np.float32)

        # color both vertices of each line segment
        color = np.hstack((color, color)).reshape((-1, 4))

        self.lineplot.setData(pos=pos, color=color)
        self.update()

