#ifndef QGRAPHICSWIDGET2_H
#define QGRAPHICSWIDGET2_H

#include <QGraphicsWidget>
#include <QGraphicsView>
#include <QDebug>
#include <QList>

#include "Point.h"
#include "Interfaces.h"
#include "ItemDefines.h"
#include "ExtendedItem.h"

class ViewBoxBase;
class GraphicsViewBase;

class QGraphicsWidget2: public QGraphicsWidget, public ExtendedItem
{
    Q_OBJECT
public:
    QGraphicsWidget2(QGraphicsItem* parent=nullptr, Qt::WindowFlags wFlags=0);
    virtual ~QGraphicsWidget2() {}

    enum { Type = CustomItemTypes::TypeGraphicsWidget };

    virtual int type() const
    {
        // Enable the use of qgraphicsitem_cast with this item.
        return Type;
    }

    QTransform deviceTransform() const;

    QTransform deviceTransform(const QTransform& viewportTransform) const
    {
        return QGraphicsObject::deviceTransform(viewportTransform);
    }

    void setParentItem(QGraphicsItem* newParent);

    virtual QTransform sceneTransform() const;

    void setFixedHeight(const double h)
    {
        setMaximumHeight(h);
        setMinimumHeight(h);
    }

    void setFixedWidth(const double w)
    {
        setMaximumWidth(w);
        setMinimumWidth(w);
    }

    double height() const { return geometry().height(); }

    double width() const { return geometry().width(); }

    virtual QRectF boundingRect() const
    {
        return mapRectFromParent(geometry()).normalized();
    }

    virtual QPainterPath shape() const
    {
        QPainterPath p;
        p.addRect(boundingRect());
        return p;
    }

public slots:

    virtual void viewRangeChanged(const QList<Point>& range);
    virtual void viewTransformChanged();

};

#endif // QGRAPHICSWIDGET2_H
