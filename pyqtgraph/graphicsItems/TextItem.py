import numpy as np
from ..Qt import QtCore, QtGui
from ..Point import Point
from .. import functions as fn
from .GraphicsObject import GraphicsObject


class TextItem(GraphicsObject):
    """
    GraphicsItem displaying unscaled text (the text will always appear normal even inside a scaled ViewBox). 
    """
    def __init__(self, text='', color=(200,200,200), html=None, anchor=(0,0),
                 border=None, fill=None, angle=0, rotateAxis=None):
        """
        ==============  =================================================================================
        **Arguments:**
        *text*          The text to display
        *color*         The color of the text (any format accepted by pg.mkColor)
        *html*          If specified, this overrides both *text* and *color*
        *anchor*        A QPointF or (x,y) sequence indicating what region of the text box will
                        be anchored to the item's position. A value of (0,0) sets the upper-left corner
                        of the text box to be at the position specified by setPos(), while a value of (1,1)
                        sets the lower-right corner.
        *border*        A pen to use when drawing the border
        *fill*          A brush to use when filling within the border
        *angle*         Angle in degrees to rotate text. Default is 0; text will be displayed upright.
        *rotateAxis*    If None, then a text angle of 0 always points along the +x axis of the scene.
                        If a QPointF or (x,y) sequence is given, then it represents a vector direction
                        in the parent's coordinate system that the 0-degree line will be aligned to. This
                        Allows text to follow both the position and orientation of its parent while still
                        discarding any scale and shear factors.
        ==============  =================================================================================


        The effects of the `rotateAxis` and `angle` arguments are added independently. So for example:

        * rotateAxis=None, angle=0 -> normal horizontal text
        * rotateAxis=None, angle=90 -> normal vertical text
        * rotateAxis=(1, 0), angle=0 -> text aligned with x axis of its parent
        * rotateAxis=(0, 1), angle=0 -> text aligned with y axis of its parent
        * rotateAxis=(1, 0), angle=90 -> text orthogonal to x axis of its parent        
        """
                     
        self.anchor = Point(anchor)
        self.rotateAxis = None if rotateAxis is None else Point(rotateAxis)
        #self.angle = 0
        GraphicsObject.__init__(self)
        self.textItem = QtGui.QGraphicsTextItem()
        self.textItem.setParentItem(self)
        self._lastTransform = None
        self._lastScene = None
        self._bounds = QtCore.QRectF()
        if html is None:
            self.setColor(color)
            self.setText(text)
        else:
            self.setHtml(html)
        self.fill = fn.mkBrush(fill)
        self.border = fn.mkPen(border)
        self.setAngle(angle)

    def setText(self, text, color=None):
        """
        Set the text of this item. 
        
        This method sets the plain text of the item; see also setHtml().
        """
        if color is not None:
            self.setColor(color)
        self.setPlainText(text)

    def setPlainText(self, text):
        """
        Set the plain text to be rendered by this item. 
        
        See QtGui.QGraphicsTextItem.setPlainText().
        """
        if text != self.toPlainText():
            self.textItem.setPlainText(text)
            self.updateTextPos()

    def toPlainText(self):
        return self.textItem.toPlainText()
        
    def setHtml(self, html):
        """
        Set the HTML code to be rendered by this item. 
        
        See QtGui.QGraphicsTextItem.setHtml().
        """
        if self.toHtml() != html:
            self.textItem.setHtml(html)
            self.updateTextPos()
        
    def toHtml(self):
        return self.textItem.toHtml()
        
    def setTextWidth(self, *args):
        """
        Set the width of the text.
        
        If the text requires more space than the width limit, then it will be
        wrapped into multiple lines.
        
        See QtGui.QGraphicsTextItem.setTextWidth().
        """
        self.textItem.setTextWidth(*args)
        self.updateTextPos()
        
    def setFont(self, *args):
        """
        Set the font for this text. 
        
        See QtGui.QGraphicsTextItem.setFont().
        """
        self.textItem.setFont(*args)
        self.updateTextPos()
        
    def setAngle(self, angle):
        """
        Set the angle of the text in degrees.

        This sets the rotation angle of the text as a whole, measured
        counter-clockwise from the x axis of the parent. Note that this rotation
        angle does not depend on horizontal/vertical scaling of the parent.
        """
        self.angle = angle
        self.updateTransform(force=True)

    def setAnchor(self, anchor):
        self.anchor = Point(anchor)
        self.updateTextPos()

    def setColor(self, color):
        """
        Set the color for this text.
        
        See QtGui.QGraphicsItem.setDefaultTextColor().
        """
        self.color = fn.mkColor(color)
        self.textItem.setDefaultTextColor(self.color)
        
    def updateTextPos(self):
        # update text position to obey anchor
        r = self.textItem.boundingRect()
        tl = self.textItem.mapToParent(r.topLeft())
        br = self.textItem.mapToParent(r.bottomRight())
        offset = (br - tl) * self.anchor
        self.textItem.setPos(-offset)
        
        ### Needed to maintain font size when rendering to image with increased resolution
        #self.textItem.resetTransform()
        ##self.textItem.rotate(self.angle)
        #if self._exportOpts is not False and 'resolutionScale' in self._exportOpts:
            #s = self._exportOpts['resolutionScale']
            #self.textItem.scale(s, s)
        
    def boundingRect(self):
        return self.textItem.mapToParent(self.textItem.boundingRect()).boundingRect()

    def viewTransformChanged(self):
        # called whenever view transform has changed.
        # Do this here to avoid double-updates when view changes.
        self.updateTransform()
        
    def paint(self, p, *args):
        # this is not ideal because it requires the transform to be updated at every draw.
        # ideally, we would have a sceneTransformChanged event to react to..
        s = self.scene()
        ls = self._lastScene
        if s is not ls:
            if ls is not None:
                ls.sigPrepareForPaint.disconnect(self.updateTransform)
            self._lastScene = s
            if s is not None:
                s.sigPrepareForPaint.connect(self.updateTransform)
            self.updateTransform()
            p.setTransform(self.sceneTransform())
        
        if self.border.style() != QtCore.Qt.NoPen or self.fill.style() != QtCore.Qt.NoBrush:
            p.setPen(self.border)
            p.setBrush(self.fill)
            p.setRenderHint(p.Antialiasing, True)
            p.drawPolygon(self.textItem.mapToParent(self.textItem.boundingRect()))
        
    def setVisible(self, v):
        GraphicsObject.setVisible(self, v)
        if v:
            self.updateTransform()
    
    def updateTransform(self, force=False):
        if not self.isVisible():
            return

        # update transform such that this item has the correct orientation
        # and scaling relative to the scene, but inherits its position from its
        # parent.
        # This is similar to setting ItemIgnoresTransformations = True, but 
        # does not break mouse interaction and collision detection.
        p = self.parentItem()
        if p is None:
            pt = QtGui.QTransform()
        else:
            pt = p.sceneTransform()
        
        if not force and pt == self._lastTransform:
            return

        t = pt.inverted()[0]
        # reset translation
        t.setMatrix(t.m11(), t.m12(), t.m13(), t.m21(), t.m22(), t.m23(), 0, 0, t.m33())
        
        # apply rotation
        angle = -self.angle
        if self.rotateAxis is not None:
            d = pt.map(self.rotateAxis) - pt.map(Point(0, 0))
            a = np.arctan2(d.y(), d.x()) * 180 / np.pi
            angle += a
        t.rotate(angle)
        
        self.setTransform(t)
        
        self._lastTransform = pt
        
        self.updateTextPos()
