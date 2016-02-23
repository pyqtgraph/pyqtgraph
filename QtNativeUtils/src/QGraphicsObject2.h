#ifndef BASEGRAPHICSITEM2_H
#define BASEGRAPHICSITEM2_H

#include <QGraphicsView>
#include <QGraphicsObject>

#include "Point.h"
#include "Interfaces.h"

class QGraphicsObject2: public QGraphicsObject
{
    Q_OBJECT
public:
    QGraphicsObject2(QGraphicsItem* parent=nullptr) : QGraphicsObject(parent)
    {}
    virtual ~QGraphicsObject2() {}

    enum { Type = QGraphicsItem::UserType + 1 };

#define ENABLE_EXTENDEDTEM_CODE     1
#define BASE_GRAPHICSITEM_CLASS     QGraphicsObject
#include "ExtendedItem.h"
#undef ENABLE_EXTENDEDTEM_CODE
#undef BASE_GRAPHICSITEM_CLASS

};

#endif // BASEGRAPHICSITEM2_H



/*
    def mapToView(self, obj):
        vt = self.viewTransform()
        if vt is None:
            return None
        return vt.map(obj)

    def mapRectToView(self, obj):
        vt = self.viewTransform()
        if vt is None:
            return None
        return vt.mapRect(obj)

    def mapFromView(self, obj):
        vt = self.viewTransform()
        if vt is None:
            return None
        vt = fn.invertQTransform(vt)
        return vt.map(obj)

    def mapRectFromView(self, obj):
        vt = self.viewTransform()
        if vt is None:
            return None
        vt = fn.invertQTransform(vt)
        return vt.mapRect(obj)

    def pos(self):
        return Point(self._qtBaseClass.pos(self))

    def viewPos(self):
        return self.mapToView(self.mapFromParent(self.pos()))

    def parentItem(self):
        ## PyQt bug -- some items are returned incorrectly.
        return GraphicsScene.translateGraphicsItem(self._qtBaseClass.parentItem(self))

    def setParentItem(self, parent):
        ## Workaround for Qt bug: https://bugreports.qt-project.org/browse/QTBUG-18616
        if parent is not None:
            pscene = parent.scene()
            if pscene is not None and self.scene() is not pscene:
                pscene.addItem(self)
        return self._qtBaseClass.setParentItem(self, parent)

    def childItems(self):
        ## PyQt bug -- some child items are returned incorrectly.
        return list(map(GraphicsScene.translateGraphicsItem, self._qtBaseClass.childItems(self)))


    def transformAngle(self, relativeItem=None):
        """Return the rotation produced by this item's transform (this assumes there is no shear in the transform)
        If relativeItem is given, then the angle is determined relative to that item.
        """
        if relativeItem is None:
            relativeItem = self.parentItem()


        tr = self.itemTransform(relativeItem)
        if isinstance(tr, tuple):  ## difference between pyside and pyqt
            tr = tr[0]
        #vec = tr.map(Point(1,0)) - tr.map(Point(0,0))
        vec = tr.map(QtCore.QLineF(0,0,1,0))
        #return Point(vec).angle(Point(1,0))
        return vec.angleTo(QtCore.QLineF(vec.p1(), vec.p1()+QtCore.QPointF(1,0)))
*/


