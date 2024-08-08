import numpy as np

from ...Qt import QtGui
from ... import functions as fn
from ..GLGraphicsItem import GLGraphicsItem
from .GLLinePlotItem import GLLinePlotItem

__all__ = ['GLGridItem']

class GLGridItem(GLGraphicsItem):
    """
    **Bases:** :class:`GLGraphicsItem <pyqtgraph.opengl.GLGraphicsItem.GLGraphicsItem>`
    
    Displays a wire-frame grid. 
    """
    
    def __init__(self, size=None, color=(255, 255, 255, 76.5), antialias=True, glOptions='translucent', parentItem=None):
        super().__init__()

        self.lineplot = None    # mark that we are still initializing

        if size is None:
            size = QtGui.QVector3D(20,20,1)
        self.setSize(size=size)
        self.setSpacing(1, 1, 1)
        self.setColor(color)

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

    def setSpacing(self, x=None, y=None, z=None, spacing=None):
        """
        Set the spacing between grid lines.
        Arguments can be x,y,z or spacing=QVector3D().
        """
        if spacing is not None:
            x = spacing.x()
            y = spacing.y()
            z = spacing.z()
        self.__spacing = [x,y,z]
        self.updateLines()
        
    def spacing(self):
        return self.__spacing[:]
        
    def setColor(self, color):
        """Set the color of the grid. Arguments are the same as those accepted by functions.mkColor()"""
        self.__color = fn.mkColor(color)
        self.updateLines()

    def color(self):
        return self.__color

    def updateLines(self):
        if self.lineplot is None:
            # still initializing
            return

        x,y,z = self.size()
        xs,ys,zs = self.spacing()
        xvals = np.arange(-x/2., x/2. + xs*0.001, xs) 
        yvals = np.arange(-y/2., y/2. + ys*0.001, ys)

        set1 = np.zeros((len(xvals), 6), dtype=np.float32)
        set1[:, 0] = xvals
        set1[:, 1] = yvals[0]
        set1[:, 3] = xvals
        set1[:, 4] = yvals[-1]

        set2 = np.zeros((len(yvals), 6), dtype=np.float32)
        set2[:, 0] = xvals[0]
        set2[:, 1] = yvals
        set2[:, 3] = xvals[-1]
        set2[:, 4] = yvals

        pos = np.vstack((set1, set2)).reshape((-1, 3))

        self.lineplot.setData(pos=pos, color=self.color())
        self.update()
