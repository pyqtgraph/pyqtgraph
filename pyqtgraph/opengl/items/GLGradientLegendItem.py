from ... import functions as fn
from ...colormap import ColorMap
from ...Qt import QtCore, QtGui
from ..GLGraphicsItem import GLGraphicsItem

__all__ = ['GLGradientLegendItem']

class GLGradientLegendItem(GLGraphicsItem):
    """
    Displays legend colorbar on the screen.
    """

    def __init__(self, **kwds):
        """
        Arguments:
            pos: position of the colorbar on the screen, from the top left corner, in pixels
            size: size of the colorbar without the text, in pixels
            gradient: a pg.ColorMap used to color the colorbar
            labels: a dict of "text":value to display next to the colorbar.
                The value corresponds to a position in the gradient from 0 to 1.
            fontColor: sets the color of the texts. Accepts any single argument accepted by
                :func:`~pyqtgraph.mkColor`
            #Todo:
                size as percentage
                legend title
        """
        GLGraphicsItem.__init__(self)
        glopts = kwds.pop("glOptions", "additive")
        self.setGLOptions(glopts)
        self.pos = (10, 10)
        self.size = (10, 100)
        self.fontColor = QtGui.QColor(QtCore.Qt.GlobalColor.white)
        # setup a default trivial gradient
        stops = (0.0, 1.0)
        self.gradient = ColorMap(pos=stops, color=(0.0, 1.0))
        self._gradient = None
        self.labels = {str(x) : x for x in stops}
        self.setData(**kwds)

    def setData(self, **kwds):
        args = ["size", "pos", "gradient", "labels", "fontColor"]
        for k in kwds.keys():
            if k not in args:
                raise Exception(
                    "Invalid keyword argument: %s (allowed arguments are %s)"
                    % (k, str(args))
                )

        self.antialias = False

        for key in kwds:
            value = kwds[key]
            if key == 'fontColor':
                value = fn.mkColor(value)
            elif key == 'gradient':
                self._gradient = None
            setattr(self, key, value)

        ##todo: add title
        ##todo: take more gradient types
        self.update()

    def paint(self):
        self.setupGLState()

        if self._gradient is None:
            self._gradient = self.gradient.getGradient()

        barRect = QtCore.QRectF(self.pos[0], self.pos[1], self.size[0], self.size[1])
        self._gradient.setStart(barRect.bottomLeft())
        self._gradient.setFinalStop(barRect.topLeft())

        painter = QtGui.QPainter(self.view())
        painter.fillRect(barRect, self._gradient)
        painter.setPen(self.fontColor)
        for labelText, labelPosition in self.labels.items():
            ## todo: draw ticks, position text properly
            x = 1.1 * self.size[0] + self.pos[0]
            y = self.size[1] - labelPosition * self.size[1] + self.pos[1] + 8
            ##todo: fonts
            painter.drawText(QtCore.QPointF(x, y), labelText)
        painter.end()
