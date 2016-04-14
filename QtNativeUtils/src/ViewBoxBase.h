#ifndef VIEWBOXBASE_H
#define VIEWBOXBASE_H

#include <QList>
#include <QGraphicsRectItem>
#include <QWidget>

#include "GraphicsWidget.h"
#include "Point.h"
#include "ItemDefines.h"
#include "GraphicsObject.h"
#include "Interfaces.h"
#include "graphicsitems/ChildGroup.h"
#include "Range.h"

//#include <sip.h>

class Limits
{
public:
    Limits()
    {}

    Range xLimits() const { return mXLimits; }
    Range yLimits() const { return mYLimits; }
    Range xRange() const { return mXRange; }
    Range yRange() const { return mYRange; }

    double xMin() const { return mXLimits.min(); }
    double xMax() const { return mXLimits.max(); }
    double yMin() const { return mYLimits.min(); }
    double yMax() const { return mYLimits.max(); }
    double minXRange() const { return mXRange.min(); }
    double maxXRange() const { return mXRange.max(); }
    double minYRange() const { return mYRange.min(); }
    double maxYRange() const { return mYRange.max(); }

    void setXLimits(const double xMin, const double xMax) { mXLimits.setRange(xMin, xMax); }
    void setYLimits(const double yMin, const double yMax) { mYLimits.setRange(yMin, yMax); }

    void setXRange(const double xMin, const double xMax) { mXRange.setRange(xMin, xMax); }
    void setYRange(const double yMin, const double yMax) { mYRange.setRange(yMin, yMax); }

    void setXMin(const double val) { mXLimits.setMin(val); }
    void setXMax(const double val) { mXLimits.setMax(val); }
    void setYMin(const double val) { mYLimits.setMin(val); }
    void setYMax(const double val) { mYLimits.setMax(val); }
    void setMinXRange(const double val) { mXRange.setMin(val); }
    void setMaxXRange(const double val) { mXRange.setMax(val); }
    void setMinYRange(const double val) { mYRange.setMin(val); }
    void setMaxYRange(const double val) { mYRange.setMax(val); }

protected:

    Range mXLimits; // Maximum and minimum visible X values
    Range mYLimits; // Maximum and minimum visible Y values
    Range mXRange;  // Maximum and minimum X range
    Range mYRange;  // Maximum and minimum Y range

};



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

    ViewBoxBase(QGraphicsItem* parent=nullptr, Qt::WindowFlags wFlags=0, const QPen& border=QPen(Qt::NoPen),
                const bool invertX=false, const bool invertY=false, const bool enableMouse=true);
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

    const QList<Range>& viewRange() const { return mViewRange; }
    const QList<Range>& targetRange() const { return mTargetRange; }
    const QList<bool>& autoRangeEnabled() const { return mAutoRangeEnabled; }

    void setAutoPan(const bool x=false, const bool y=false);
    const QList<bool>& autoPan() const { return mAutoPan; }

    void setAutoVisible(const bool x=false, const bool y=false);
    const QList<bool>& autoVisible() const { return mAutoVisibleOnly; }

    double aspectLocked() const { return mAspectLocked; }
    void setAspectLocked(const bool lock=true, const double ratio=1.0);

    const QVector<bool>& mouseEnabled() const { return mMouseEnabled; }
    void setMouseEnabled(const bool enabledOnX=true, const bool enabledOnY=true);

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

    void setRange(const Range& xRange=Range(), const Range& yRange=Range(), const double padding=AutoPadding, const bool disableAutoRange=true);
    void setRange(const QRectF& rect, const double padding=AutoPadding, const bool disableAutoRange=true);

    void setXRange(const double minR, const double maxR, const double padding=AutoPadding);
    void setYRange(const double minR, const double maxR, const double padding=AutoPadding);

    void scaleBy(const QPointF& s, const QPointF& center);
    void scaleBy(const double x, const double y, const QPointF& center) { scaleBy(QPointF(x, y), center); }

    void scaleBy(const QPointF& s);
    void scaleBy(const double x, const double y) { scaleBy(QPointF(x, y)); }

    void translateBy(const QPointF& t);
    void translateBy(const double x, const double y) { translateBy(QPointF(x, y)); }

    void setXLimits(const double xMin, const double xMax) { mLimits.setXLimits(xMin, xMax); }
    void setXLimits(const Range& rng) { mLimits.setXLimits(rng.min(), rng.max()); }

    void setYLimits(const double yMin, const double yMax) { mLimits.setYLimits(yMin, yMax); }
    void setYLimits(const Range& rng) { mLimits.setYLimits(rng.min(), rng.max()); }

    void setXRangeLimits(const double xMin, const double xMax) { mLimits.setXRange(xMin, xMax); }
    void setXRangeLimits(const Range& rng) { mLimits.setXRange(rng.min(), rng.max()); }

    void setYRangeLimits(const double yMin, const double yMax) { mLimits.setYRange(yMin, yMax); }
    void setYRangeLimits(const Range& rng) { mLimits.setYRange(rng.min(), rng.max()); }

    Range xLimits() const { return mLimits.xLimits(); }
    Range yLimits() const { return mLimits.yLimits(); }

    Range xRangeLimits() const { return mLimits.xRange(); }
    Range yRangeLimits() const { return mLimits.yRange(); }

    MouseMode mouseMode() const { return mMouseMode; }
    void setMouseMode(const MouseMode mode);

    void setWheelScaleFactor(const double factor) { mWheelScaleFactor = factor; }
    double wheelScaleFactor() const { return mWheelScaleFactor; }

    void linkedWheelEvent(QGraphicsSceneWheelEvent* event, const Axis axis=XYAxes);

    virtual void mouseDragEvent(MouseDragEvent* event);
    virtual void mouseDragEvent(MouseDragEvent* event, const Axis axis);

    QList<QGraphicsItem*> allChildren(QGraphicsItem* item=nullptr) const;

    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget=nullptr);

public slots:

    void prepareForPaint();

    void showAxRect(const QRectF& axRect);

    void scaleHistory(const int d);

protected:

    void setViewRange(const Range& x, const Range& y);
    void setTargetRange(const Range& x, const Range& y);
    void setAutoRangeEnabled(const bool enableX, const bool enableY);

    void _resetTarget();

    void setInnerSceneItem(GraphicsObject* innerItem);

    virtual QVariant itemChange(GraphicsItemChange change, const QVariant& value);

    virtual void wheelEvent(QGraphicsSceneWheelEvent* event);
    virtual void keyPressEvent(QKeyEvent *event);

    void updateScaleBox(const QPointF& p1, const QPointF& p2);
    void hideScaleBox();

    void addToHistory(const QRectF& r);

signals:

    void sigYRangeChanged(const Range& range);
    void sigXRangeChanged(const Range& range);
    void sigRangeChangedManually(const bool mouseLeft, const bool mouseRight);
    void sigRangeChanged(const Range& xRange, const Range& yRange);
    void sigStateChanged(ViewBoxBase* viewBox);
    void sigTransformChanged();
    void sigResized();

public:

    static constexpr double AutoPadding = std::numeric_limits<double>::quiet_NaN();

protected:

    bool mMatrixNeedsUpdate;
    bool mAutoRangeNeedsUpdate;

    bool mXInverted;
    bool mYInverted;

    QList<Range> mViewRange;    // actual range viewed
    QList<Range> mTargetRange;  // child coord. range visible [[xmin, xmax], [ymin, ymax]]
    double mAspectLocked;   // 0.0: aspect unlocked, double for the aspect ratio
    QList<bool> mAutoRangeEnabled;
    QList<bool> mAutoPan;  // whether to only pan (do not change scaling) when auto-range is enabled
    QList<bool> mAutoVisibleOnly;  // whether to auto-range only to the visible portion of a plot
    QVector<bool> mMouseEnabled;    //

    MouseMode mMouseMode;

    QGraphicsRectItem* mBackground = nullptr;

    GraphicsObject* mInnerSceneItem = nullptr;

    QGraphicsRectItem* mRbScaleBox = nullptr;

    ChildGroup* mChildGroup;

    QList<QGraphicsItem*> mAddedItems;

    QPen mBorder;

    Limits mLimits;

    double mWheelScaleFactor;

    QVector<QRectF> mAxHistory;
    int mAxHistoryPointer;
};

#endif // VIEWBOXBASE_H
