#ifndef VIEWBOXBASE_H
#define VIEWBOXBASE_H

#include <QList>

#include "QGraphicsWidget2.h"
#include "Point.h"
#include "ItemDefines.h"

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


    bool matrixNeedsUpdate() const { return mMatrixNeedsUpdate; }
    void setMatrixNeedsUpdate(const bool on) { mMatrixNeedsUpdate = on; }

    bool autoRangeNeedsUpdate() const { return mAutoRangeNeedsUpdate; }
    void setAutoRangeNeedsUpdate(const bool on) { mAutoRangeNeedsUpdate = on; }

    void invertY(const bool b=true) { mYInverted = b; }
    bool yInverted() const { return mYInverted; }

    void invertX(const bool b=true) { mXInverted = b; }
    bool xInverted() const { return mXInverted; }

protected:

    void setViewRange(const Point& x, const Point& y);
    const QList<Point>& viewRange() const { return mViewRange; }


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

    QList<Point> mViewRange;
};

#endif // VIEWBOXBASE_H
