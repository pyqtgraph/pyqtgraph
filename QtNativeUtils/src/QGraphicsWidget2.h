#ifndef QGRAPHICSWIDGET2_H
#define QGRAPHICSWIDGET2_H

#include <QGraphicsWidget>
#include <QGraphicsView>
#include <QDebug>
#include <QList>

#include "Point.h"
#include "Interfaces.h"
#include "ItemDefines.h"

class ViewBoxBase;
class GraphicsViewBase;

class QGraphicsWidget2: public QGraphicsWidget
{
    Q_OBJECT
public:
    QGraphicsWidget2(QGraphicsItem* parent=nullptr, Qt::WindowFlags wFlags=0);
    virtual ~QGraphicsWidget2() {}

    enum { Type = CustomItemTypes::TypeGraphicsWidget };

#define ENABLE_EXTENDEDTEM_CODE     1
#define BASE_GRAPHICSITEM_CLASS     QGraphicsWidget
#include "ExtendedItem.h"
#undef ENABLE_EXTENDEDTEM_CODE
#undef BASE_GRAPHICSITEM_CLASS

public:

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

};

#endif // QGRAPHICSWIDGET2_H
