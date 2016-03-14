from ..Qt import QtGui, QtCore  
from ..GraphicsScene import GraphicsScene
from .GraphicsItem import GraphicsItem
import sip

from ..QtNativeUtils import GraphicsWidget

__all__ = ['GraphicsWidget']

'''
class GraphicsWidget(GraphicsItem, QGraphicsWidget2):
    
    _qtBaseClass = QGraphicsWidget2
    def __init__(self, *args, **kargs):
        """
        **Bases:** :class:`GraphicsItem <pyqtgraph.GraphicsItem>`, :class:`QtGui.QGraphicsWidget`
        
        Extends QGraphicsWidget with several helpful methods and workarounds for PyQt bugs. 
        Most of the extra functionality is inherited from :class:`GraphicsItem <pyqtgraph.GraphicsItem>`.
        """
        QGraphicsWidget2.__init__(self, *args, **kargs)
        GraphicsItem.__init__(self)
        
        ## done by GraphicsItem init
        #GraphicsScene.registerObject(self)  ## workaround for pyqt bug in graphicsscene.items()
'''
