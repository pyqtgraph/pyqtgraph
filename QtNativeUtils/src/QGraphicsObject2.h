#ifndef BASEGRAPHICSITEM2_H
#define BASEGRAPHICSITEM2_H

#include <QGraphicsView>
#include <QGraphicsObject>
#include <QDebug>

#include "Point.h"
#include "Interfaces.h"
#include "ItemDefines.h"

class ViewBoxBase;
class GraphicsViewBase;

class QGraphicsObject2: public QGraphicsObject
{
    Q_OBJECT
public:
    QGraphicsObject2(QGraphicsItem* parent=nullptr) : QGraphicsObject(parent)
    {}
    virtual ~QGraphicsObject2() {}

    enum { Type = CustomItemTypes::TypeGraphicsObject };

#define ENABLE_EXTENDEDTEM_CODE     1
#define BASE_GRAPHICSITEM_CLASS     QGraphicsObject
#include "ExtendedItem.h"
#undef ENABLE_EXTENDEDTEM_CODE
#undef BASE_GRAPHICSITEM_CLASS

};

#endif // BASEGRAPHICSITEM2_H



/*


    def pos(self):
        return Point(self._qtBaseClass.pos(self))

    def viewPos(self):
        return self.mapToView(self.mapFromParent(self.pos()))

    def parentItem(self):
        ## PyQt bug -- some items are returned incorrectly.
        return GraphicsScene.translateGraphicsItem(self._qtBaseClass.parentItem(self))

    def childItems(self):
        ## PyQt bug -- some child items are returned incorrectly.
        return list(map(GraphicsScene.translateGraphicsItem, self._qtBaseClass.childItems(self)))

*/


