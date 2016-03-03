#ifndef VIEWBOXBASE_H
#define VIEWBOXBASE_H

#include <QList>
#include <QGraphicsRectItem>

#include "QGraphicsWidget2.h"
#include "Point.h"
#include "ItemDefines.h"
#include "QGraphicsObject2.h"

class ViewBoxBase: public QGraphicsWidget2
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

    QGraphicsObject2* innerSceneItem() const;


    void setViewRange(const Point& x, const Point& y);
    void setTargetRange(const Point& x, const Point& y);
    void setAutoRangeEnabled(const bool enableX, const bool enableY);

    void _resetTarget();

    void setInnerSceneItem(QGraphicsObject2* innerItem);
protected:

signals:

    void sigYRangeChanged(ViewBoxBase* viewBox, const Point& range);
    void sigXRangeChanged(ViewBoxBase* viewBox, const Point& range);
    void sigRangeChangedManually(const bool mouseLeft, const bool mouseRight);
    void sigRangeChanged(ViewBoxBase* viewBox, const QList<Point>& range);
    void sigStateChanged(ViewBoxBase* viewBox);
    void sigTransformChanged(ViewBoxBase* viewBox);
    void sigResized(ViewBoxBase* viewBox);

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

    QGraphicsObject2* mInnerSceneItem = nullptr;
};

#endif // VIEWBOXBASE_H
