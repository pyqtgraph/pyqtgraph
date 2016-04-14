#include "UIGraphicsItem.h"

#include <QPainterPathStroker>

UIGraphicsItem::UIGraphicsItem(QGraphicsItem *parent) :
    GraphicsObject(parent),
    mBoundingRect(QRectF())
{
    setFlag(ItemSendsScenePositionChanges, true);

    parentIsChanged();
}

QRectF UIGraphicsItem::boundingRect() const
{
    if(mBoundingRect.isNull())
        mBoundingRect = viewRect();
    return QRectF(mBoundingRect);
}

void UIGraphicsItem::setNewBounds()
{
    // Update the item's bounding rect to match the viewport
    mBoundingRect = QRectF();  // invalidate bounding rect, regenerate later if needed.
    prepareGeometryChange();
}

void UIGraphicsItem::setPos(const QPointF& pos)
{
    GraphicsObject::setPos(pos);
    setNewBounds();
}

void UIGraphicsItem::setPos(double x, double y)
{
    GraphicsObject::setPos(x, y);
    setNewBounds();
}

Range UIGraphicsItem::dataBounds(const ViewBoxBase::Axis ax, const double frac, const Range orthoRange) const
{
    return Range();  // Empty Range
}

QPainterPath UIGraphicsItem::mouseShape() const
{
    // Return the shape of this item after expanding by 2 pixels
    QPainterPath ds = mapToDevice(shape());
    QPainterPathStroker stroker;
    stroker.setWidth(2.0);
    QPainterPath ds2 = stroker.createStroke(ds).united(ds);
    return mapFromDevice(ds2);
}

void UIGraphicsItem::viewRangeChanged(const Range& xRange, const Range& yRange)
{
    setNewBounds();
    update();
}

QVariant UIGraphicsItem::itemChange(QGraphicsItem::GraphicsItemChange change, const QVariant& value)
{
    if(change==ItemScenePositionHasChanged)
        setNewBounds();

    return GraphicsObject::itemChange(change, value);
}
