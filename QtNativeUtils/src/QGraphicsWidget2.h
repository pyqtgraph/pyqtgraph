#ifndef QGRAPHICSWIDGET2_H
#define QGRAPHICSWIDGET2_H

#include <QGraphicsWidget>
#include <QGraphicsView>

#include "Point.h"


class QGraphicsWidget2: public QGraphicsWidget
{
public:
    QGraphicsWidget2(QGraphicsItem* parent=nullptr, Qt::WindowFlags wFlags=0);
    ~QGraphicsWidget2() {}

    QGraphicsView* getViewWidget() const
    {
        QGraphicsScene* s = scene();
        if(s==nullptr)
            return nullptr;
        QList<QGraphicsView*> views = s->views();
        if(views.size()>0)
            return views[0];
        return nullptr;
    }

    void forgetViewWidget()
    {}

    QTransform deviceTransform() const
    {
        QGraphicsView* view = getViewWidget();
        if(view==nullptr)
            return QTransform();
        return QGraphicsObject::deviceTransform(view->viewportTransform());
    }

    QTransform deviceTransform(const QTransform& viewportTransform) const
    {
        return QGraphicsObject::deviceTransform(viewportTransform);
    }

    QList<QGraphicsItem*> getBoundingParents() const;

    QVector<Point> pixelVectors() const
    {
        return pixelVectors(QPointF(1.0, 0.0));
    }

    QVector<Point> pixelVectors(const QPointF& direction) const;

    double pixelLength(const QPointF& direction, const bool ortho=false) const
    {
        QVector<Point> p = pixelVectors(direction);
        if(ortho)
            return p[1].length();
        return p[0].length();
    }


    QPointF	mapFromDevice(const QPointF& point) const { return deviceTransform().inverted().map(point); }
    QPointF	mapFromDevice(const QPoint& point) const { return deviceTransform().inverted().map(QPointF(point)); }
    QPolygonF mapFromDevice(const QRectF& rect) const { return deviceTransform().inverted().map(rect); }
    QPolygonF mapFromDevice(const QPolygonF& polygon) const { return deviceTransform().inverted().map(polygon); }
    QPainterPath mapFromDevice(const QPainterPath& path) const { return deviceTransform().inverted().map(path); }
    QPointF	mapFromDevice(qreal x, qreal y) const { return mapFromDevice(QPointF(x, y)); }

    QPointF	mapToDevice(const QPointF& point) const { return deviceTransform().map(point); }
    QPointF	mapToDevice(const QPoint& point) const { return deviceTransform().map(QPointF(point)); }
    QPolygonF mapToDevice(const QRectF& rect) const { return deviceTransform().map(rect); }
    QPolygonF mapToDevice(const QPolygonF& polygon) const { return deviceTransform().map(polygon); }
    QPainterPath mapToDevice(const QPainterPath& path) const { return deviceTransform().map(path); }
    QPointF	mapToDevice(qreal x, qreal y) const { return mapToDevice(QPointF(x, y)); }

    QRectF mapRectToDevice(const QRectF& rect) const { return deviceTransform().mapRect(rect); }
    QRect mapRectToDevice(const QRect& rect) const { return deviceTransform().mapRect(rect); }

    QRectF mapRectFromDevice(const QRectF& rect) const { return deviceTransform().inverted().mapRect(rect); }
    QRect mapRectFromDevice(const QRect& rect) const { return deviceTransform().inverted().mapRect(rect); }

    double transformAngle(QGraphicsItem* relativeItem=nullptr) const;
};

#endif // QGRAPHICSWIDGET2_H
