#include "QGraphicsObject2.h"

QGraphicsObject2::QGraphicsObject2(QGraphicsItem* parent): QGraphicsObject(parent)
{
}

QList<QGraphicsItem *> QGraphicsObject2::getBoundingParents() const
{
    // Return a list of parents to this item that have child clipping enabled.
    QGraphicsItem* p = parentItem();
    QList<QGraphicsItem*> parents;

    while(p!=nullptr)
    {
        p = p->parentItem();
        if(p==nullptr)
            break;
        if(p->flags() & ItemClipsChildrenToShape)
            parents.append(p);
    }

    return parents;
}

QVector<Point> QGraphicsObject2::pixelVectors(const QPointF &direction) const
{
    // Return vectors in local coordinates representing the width and height of a view pixel.
    // If direction is specified, then return vectors parallel and orthogonal to it.

    // Return (None, None) if pixel size is not yet defined (usually because the item has not yet been displayed)
    // or if pixel size is below floating-point precision limit.

    QVector<Point> result(2, Point(0.0, 0.0));

    QTransform devTr = deviceTransform();
    QTransform dt(devTr.m11(), devTr.m12(), devTr.m21(), devTr.m22(), 0.0, 0.0);

    if(direction.manhattanLength()==0.0)
        return result;

    QLineF dirLine; // p1 and p2 are (0, 0)
    dirLine.setP2(direction);
    dirLine = dt.map(dirLine);
    if(dirLine.length()==0.0)
        return result; // pixel size cannot be represented on this scale

    QLineF normView(dirLine.unitVector());
    QLineF normOrtho(normView.normalVector());

    QTransform dti = dt.inverted();
    result[0] = Point(dti.map(normView).p2());
    result[1] = Point(dti.map(normOrtho).p2());

    return result;
}
