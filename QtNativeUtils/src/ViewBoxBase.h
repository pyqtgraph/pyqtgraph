#ifndef VIEWBOXBASE_H
#define VIEWBOXBASE_H

#include <QList>
#include <QGraphicsRectItem>

#include "GraphicsWidget.h"
#include "Point.h"
#include "ItemDefines.h"
#include "GraphicsObject.h"
#include "Interfaces.h"
#include "graphicsitems/ChildGroup.h"

class ViewBoxBase: public GraphicsWidget, public ItemChangedListener
{
    Q_OBJECT
public:

    enum MouseMode
    {
        PanMode = 3,
        RectMode = 1
    };

    enum Axis
    {
        XAxis = 0,
        YAxis = 1,
        XYAxes = 2
    };

    ViewBoxBase(QGraphicsItem* parent=nullptr, Qt::WindowFlags wFlags=0,
                const bool invertX=false, const bool invertY=false);
    virtual ~ViewBoxBase() {}

    enum { Type = CustomItemTypes::TypeViewBox };

    virtual int type() const
    {
        // Enable the use of qgraphicsitem_cast with this item.
        return Type;
    }

    virtual void updateViewRange(const bool forceX=false, const bool forceY=false) {}
    virtual void updateAutoRange() {}
    virtual void updateMatrix();

    virtual void itemBoundsChanged(QGraphicsItem* item);


    bool matrixNeedsUpdate() const { return mMatrixNeedsUpdate; }
    void setMatrixNeedsUpdate(const bool on) { mMatrixNeedsUpdate = on; }

    bool autoRangeNeedsUpdate() const { return mAutoRangeNeedsUpdate; }
    void setAutoRangeNeedsUpdate(const bool on) { mAutoRangeNeedsUpdate = on; }

    void invertY(const bool b=true);
    bool yInverted() const { return mYInverted; }

    void invertX(const bool b=true);
    bool xInverted() const { return mXInverted; }

    void setBackgroundColor(const QColor& color);
    QColor backgroundColor() const;
    void updateBackground();

    const QVector<Point>& viewRange() const { return mViewRange; }
    const QVector<Point>& targetRange() const { return mTargetRange; }
    const QVector<bool>& autoRangeEnabled() const { return mAutoRangeEnabled; }

    void setAutoPan(const bool x=false, const bool y=false);
    const QVector<bool>& autoPan() const { return mAutoPan; }

    void setAutoVisible(const bool x=false, const bool y=false);
    const QVector<bool>& autoVisible() const { return mAutoVisibleOnly; }

    double aspectLocked() const { return mAspectLocked; }
    void setAspectLocked(const bool lock=true, const double ratio=1.0);

    virtual QRectF viewRect() const;
    QRectF targetRect() const;

    GraphicsObject* innerSceneItem() const;

    // called when items are added/removed from self.childGroup
    virtual void itemsChanged();

    ChildGroup* getChildGroup() const;

    QTransform childTransform() const;

    const QList<QGraphicsItem*>& addedItems() const;

    void addItem(QGraphicsItem* item, const bool ignoreBounds=false);

    void removeItem(QGraphicsItem* item);

    void clear();

    double suggestPadding(const Axis ax) const;

    void enableAutoRange(const Axis axis=XYAxes, const bool enable=true);
    void enableAutoRange(const QString& axis="xy", const bool enable=true);

    // Maps from the local coordinates of the ViewBox to the coordinate system displayed inside the ViewBox
    QPointF	mapToView(const QPointF& point) const { return childTransform().inverted().map(point); }
    QPointF	mapToView(const QPoint& point) const { return childTransform().inverted().map(QPointF(point)); }
    QPolygonF mapToView(const QRectF& rect) const { return childTransform().inverted().map(rect); }
    QPolygonF mapToView(const QPolygonF& polygon) const { return childTransform().inverted().map(polygon); }
    QPainterPath mapToView(const QPainterPath& path) const { return childTransform().inverted().map(path); }
    QPointF	mapToView(qreal x, qreal y) const { return mapToView(QPointF(x, y)); }

    // Maps from the coordinate system displayed inside the ViewBox to the local coordinates of the ViewBox
    QPointF	mapFromView(const QPointF& point) const { return childTransform().map(point); }
    QPointF	mapFromView(const QPoint& point) const { return childTransform().map(QPointF(point)); }
    QPolygonF mapFromView(const QRectF& rect) const { return childTransform().map(rect); }
    QPolygonF mapFromView(const QPolygonF& polygon) const { return childTransform().map(polygon); }
    QPainterPath mapFromView(const QPainterPath& path) const { return childTransform().map(path); }
    QPointF	mapFromView(qreal x, qreal y) const { return mapFromView(QPointF(x, y)); }

    // Maps from scene coordinates to the coordinate system displayed inside the ViewBox
    QPointF	mapSceneToView(const QPointF& point) const { return mapToView(mapFromScene(point)); }
    QPointF	mapSceneToView(const QPoint& point) const { return mapToView(mapFromScene(QPointF(point))); }
    QPolygonF mapSceneToView(const QRectF& rect) const { return mapToView(mapFromScene(rect)); }
    QPolygonF mapSceneToView(const QPolygonF& polygon) const { return mapToView(mapFromScene(polygon)); }
    QPainterPath mapSceneToView(const QPainterPath& path) const { return mapToView(mapFromScene(path)); }
    QPointF	mapSceneToView(qreal x, qreal y) const { return mapSceneToView(QPointF(x, y)); }

    // Maps from the coordinate system displayed inside the ViewBox to scene coordinates
    QPointF	mapViewToScene(const QPointF& point) const { return mapToScene(mapFromView(point)); }
    QPointF	mapViewToScene(const QPoint& point) const { return mapToScene(mapFromView(QPointF(point))); }
    QPolygonF mapViewToScene(const QRectF& rect) const { return mapToScene(mapFromView(rect)); }
    QPolygonF mapViewToScene(const QPolygonF& polygon) const { return mapToScene(mapFromView(polygon)); }
    QPainterPath mapViewToScene(const QPainterPath& path) const { return mapToScene(mapFromView(path)); }
    QPointF	mapViewToScene(qreal x, qreal y) const { return mapViewToScene(QPointF(x, y)); }

    // Maps *obj* from the local coordinate system of *item* to the view coordinates
    QPointF	mapFromItemToView(const QGraphicsItem* item, const QPointF& point) const { return mChildGroup->mapFromItem(item, point); }
    QPointF	mapFromItemToView(const QGraphicsItem* item, const QPoint& point) const { return mChildGroup->mapFromItem(item, QPointF(point)); }
    QPolygonF mapFromItemToView(const QGraphicsItem* item, const QRectF& rect) const { return mChildGroup->mapFromItem(item, rect); }
    QPolygonF mapFromItemToView(const QGraphicsItem* item, const QPolygonF& polygon) const { return mChildGroup->mapFromItem(item, polygon); }
    QPainterPath mapFromItemToView(const QGraphicsItem* item, const QPainterPath& path) const { return mChildGroup->mapFromItem(item, path); }
    QPolygonF mapFromItemToView(const QGraphicsItem* item, qreal x, qreal y, qreal w, qreal h ) const { return mChildGroup->mapFromItem(item, x, y, w, h); }
    QPointF	mapFromItemToView(const QGraphicsItem* item, qreal x, qreal y) const { return mapFromItemToView(item, QPointF(x, y)); }

    // Maps *obj* from view coordinates to the local coordinate system of *item*.
    QPointF	mapFromViewToItem(const QGraphicsItem* item, const QPointF& point) const { return mChildGroup->mapToItem(item, point); }
    QPointF	mapFromViewToItem(const QGraphicsItem* item, const QPoint& point) const { return mChildGroup->mapToItem(item, QPointF(point)); }
    QPolygonF mapFromViewToItem(const QGraphicsItem* item, const QRectF& rect) const { return mChildGroup->mapToItem(item, rect); }
    QPolygonF mapFromViewToItem(const QGraphicsItem* item, const QPolygonF& polygon) const { return mChildGroup->mapToItem(item, polygon); }
    QPainterPath mapFromViewToItem(const QGraphicsItem* item, const QPainterPath& path) const { return mChildGroup->mapToItem(item, path); }
    QPolygonF mapFromViewToItem(const QGraphicsItem* item, qreal x, qreal y, qreal w, qreal h ) const { return mChildGroup->mapToItem(item, x, y, w, h); }
    QPointF	mapFromViewToItem(const QGraphicsItem* item, qreal x, qreal y) const { return mapFromViewToItem(item, QPointF(x, y)); }

    QPointF	mapViewToDevice(const QPointF& point) const { return mapToDevice(mapFromView(point)); }
    QPointF	mapViewToDevice(const QPoint& point) const { return mapToDevice(mapFromView(QPointF(point))); }
    QPolygonF mapViewToDevice(const QRectF& rect) const { return mapToDevice(mapFromView(rect)); }
    QPolygonF mapViewToDevice(const QPolygonF& polygon) const { return mapToDevice(mapFromView(polygon)); }
    QPainterPath mapViewToDevice(const QPainterPath& path) const { return mapToDevice(mapFromView(path)); }
    QPointF	mapViewToDevice(qreal x, qreal y) const { return mapViewToDevice(QPointF(x, y)); }

    QPointF	mapDeviceToView(const QPointF& point) const { return mapToView(mapFromDevice(point)); }
    QPointF	mapDeviceToView(const QPoint& point) const { return mapToView(mapFromDevice(QPointF(point))); }
    QPolygonF mapDeviceToView(const QRectF& rect) const { return mapToView(mapFromDevice(rect)); }
    QPolygonF mapDeviceToView(const QPolygonF& polygon) const { return mapToView(mapFromDevice(polygon)); }
    QPainterPath mapDeviceToView(const QPainterPath& path) const { return mapToView(mapFromDevice(path)); }
    QPointF	mapDeviceToView(qreal x, qreal y) const { return mapDeviceToView(QPointF(x, y)); }

    QRectF itemBoundingRect(const QGraphicsItem* item) const;


public slots:

    void prepareForPaint();

protected:

    void setViewRange(const Point& x, const Point& y);
    void setTargetRange(const Point& x, const Point& y);
    void setAutoRangeEnabled(const bool enableX, const bool enableY);

    void _resetTarget();

    void setInnerSceneItem(GraphicsObject* innerItem);

    virtual QVariant itemChange(GraphicsItemChange change, const QVariant& value);

signals:

    void sigYRangeChanged(const Point& range);
    void sigXRangeChanged(const Point& range);
    void sigRangeChangedManually(const bool mouseLeft, const bool mouseRight);
    void sigRangeChanged(const QList<Point>& range);
    void sigStateChanged(ViewBoxBase* viewBox);
    void sigTransformChanged();
    void sigResized();

protected:

    bool mMatrixNeedsUpdate;
    bool mAutoRangeNeedsUpdate;

    bool mXInverted;
    bool mYInverted;

    QVector<Point> mViewRange;    // actual range viewed
    QVector<Point> mTargetRange;  // child coord. range visible [[xmin, xmax], [ymin, ymax]]
    double mAspectLocked;   // 0.0: aspect unlocked, double for the aspect ratio
    QVector<bool> mAutoRangeEnabled;
    QVector<bool> mAutoPan;  // whether to only pan (do not change scaling) when auto-range is enabled
    QVector<bool> mAutoVisibleOnly;  // whether to auto-range only to the visible portion of a plot

    QGraphicsRectItem* mBackground = nullptr;

    GraphicsObject* mInnerSceneItem = nullptr;

    ChildGroup* mChildGroup;

    QList<QGraphicsItem*> mAddedItems;
};

#endif // VIEWBOXBASE_H
