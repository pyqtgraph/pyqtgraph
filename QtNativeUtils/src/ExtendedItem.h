#ifndef EXTENDED_ITEM_H
#define EXTENDED_ITEM_H

#include <QGraphicsObject>
#include <QHash>
#include <QVariant>

#include "GraphicsViewBase.h"
#include "Point.h"
#include "mouseevents/MouseClickEvent.h"
#include "mouseevents/HoverEvent.h"
#include "mouseevents/MouseDragEvent.h"
#include "Range.h"


class GraphicsViewBase;
class ViewBoxBase;


class ExtendedItem
{

public:
    ExtendedItem(QGraphicsObject* impl) : mItemImpl(impl) {}
    virtual ~ExtendedItem() {}

    virtual GraphicsViewBase* getViewWidget() const;

    virtual void forgetViewWidget()
    {
        mView = nullptr;
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

    virtual QTransform deviceTransform() const = 0;

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

    QPointF	mapToView(const QPointF& point) const { return viewTransform().map(point); }
    QPointF	mapToView(const QPoint& point) const { return viewTransform().map(QPointF(point)); }
    QPolygonF mapToView(const QRectF& rect) const { return viewTransform().map(rect); }
    QPolygonF mapToView(const QPolygonF& polygon) const { return viewTransform().map(polygon); }
    QPainterPath mapToView(const QPainterPath& path) const { return viewTransform().map(path); }
    QPointF	mapToView(qreal x, qreal y) const { return mapToView(QPointF(x, y)); }

    QRectF mapRectToView(const QRectF& rect) const { return viewTransform().mapRect(rect); }
    QRect mapRectToView(const QRect& rect) const { return viewTransform().mapRect(rect); }

    QPointF	mapFromView(const QPointF& point) const { return viewTransform().inverted().map(point); }
    QPointF	mapFromView(const QPoint& point) const { return viewTransform().inverted().map(QPointF(point)); }
    QPolygonF mapFromView(const QRectF& rect) const { return viewTransform().inverted().map(rect); }
    QPolygonF mapFromView(const QPolygonF& polygon) const { return viewTransform().inverted().map(polygon); }
    QPainterPath mapFromView(const QPainterPath& path) const { return viewTransform().inverted().map(path); }
    QPointF	mapFromView(qreal x, qreal y) const { return mapFromView(QPointF(x, y)); }

    QRectF mapRectFromView(const QRectF& rect) const { return viewTransform().inverted().mapRect(rect); }
    QRect mapRectFromView(const QRect& rect) const { return viewTransform().inverted().mapRect(rect); }

    double transformAngle(QGraphicsItem* relativeItem=nullptr) const;

    virtual void mouseClickEvent(MouseClickEvent* event) { event->ignore(); }
    virtual void hoverEvent(HoverEvent* event) { event->ignore(); }
    virtual void mouseDragEvent(MouseDragEvent* event) { event->ignore(); }

    virtual ViewBoxBase* getNativeViewBox() const;

    QObject* getViewBox() const
    {
        getNativeViewBox();
        if(mViewBox!=nullptr)
            return (QObject*)mViewBox;
        else if(mView!=nullptr)
            return (QObject*)mView;

        return nullptr;
    }

    virtual void forgetViewBox()
    {
        mViewBox = nullptr;
        mViewBoxIsViewWidget = false;
    }

    virtual QTransform viewTransform() const;

    virtual QRectF viewRect() const;

    QList<QGraphicsItem*> allChildItems(QGraphicsItem* root=nullptr) const;

    QPainterPath childrenShape() const;

    QPointF pixelSize() const;
    double pixelWidth() const;
    double pixelHeight() const;

    QPointF viewPos() const;

    virtual void informViewBoundsChanged();

    virtual void _updateView();

    void _replaceView(QGraphicsItem* item=nullptr);

    bool isViewBox(const ViewBoxBase* vb) const;
    bool isViewBox(const GraphicsViewBase* vb) const;

    void parentIsChanged();

    virtual void viewRangeChanged(const Range& xRange, const Range& yRange) = 0;
    virtual void viewTransformChanged() = 0;

    void setExportMode(const bool isExporting, const QVariantHash& opt=QVariantHash());

    const QVariantHash& getExportMode() const;

protected:

    virtual void viewChanged();

    virtual void disconnectView(ViewBoxBase* view) = 0;
    virtual void disconnectView(GraphicsViewBase* view) = 0;

    virtual void connectView(ViewBoxBase* view) = 0;
    virtual void connectView(GraphicsViewBase* view) = 0;

protected:

    mutable GraphicsViewBase* mView = nullptr;
    mutable ViewBoxBase* mViewBox = nullptr;
    mutable bool mViewBoxIsViewWidget = false;

    QVariantHash mExportOptions;

private:

    QGraphicsObject* mItemImpl;


};

#endif // ENABLE_EXTENDEDTEM_CODE

