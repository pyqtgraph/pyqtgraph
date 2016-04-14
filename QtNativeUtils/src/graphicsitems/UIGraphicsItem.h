#ifndef UIGRAPHICSITEMBASE_H
#define UIGRAPHICSITEMBASE_H

#include "GraphicsObject.h"
#include "Range.h"
#include "ViewBoxBase.h"

class UIGraphicsItem : public GraphicsObject
{
    Q_OBJECT
public:
    explicit UIGraphicsItem(QGraphicsItem *parent=nullptr);
    virtual ~UIGraphicsItem() {}

    virtual QRectF boundingRect() const;

    void setNewBounds();

    void setPos(const QPointF & pos);
    void setPos(double x, double y);

    Range dataBounds(const ViewBoxBase::Axis ax, const double frac=1.0, const Range orthoRange=Range()) const;

    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget=nullptr) {}

    virtual QPainterPath mouseShape() const;

public slots:

    virtual void viewRangeChanged(const Range& xRange, const Range& yRange);

protected:

    virtual QVariant itemChange(GraphicsItemChange change, const QVariant& value);

protected:

    mutable QRectF mBoundingRect;
};

#endif // UIGRAPHICSITEMBASE_H
