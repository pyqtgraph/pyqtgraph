#include "ViewBoxBase.h"

#include <QSizePolicy>

ViewBoxBase::ViewBoxBase(QGraphicsItem *parent, Qt::WindowFlags wFlags, const bool invertX, const bool invertY) :
    QGraphicsWidget2(parent, wFlags),
    mMatrixNeedsUpdate(true),
    mAutoRangeNeedsUpdate(true),
    mXInverted(invertX),
    mYInverted(invertY)
{
    setFlag(ItemClipsChildrenToShape, true);
    setFlag(ItemIsFocusable, true);

    setZValue(-100);
    setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding));

    mViewRange.clear();
    mViewRange << Point(0.0, 1.0) << Point(0.0, 1.0);

    mTargetRange.clear();
    mTargetRange << Point(0.0, 1.0) << Point(0.0, 1.0);

    mAutoRangeEnabled.clear();
    mAutoRangeEnabled << true << true;

    mAutoPan.clear();
    mAutoPan << false << false;

    mBackground = new QGraphicsRectItem(rect());
    mBackground->setParentItem(this);
    mBackground->setZValue(-1e6);
    mBackground->setPen(QPen(Qt::NoPen));
    mBackground->setBrush(QBrush(QColor(Qt::transparent)));
    updateBackground();
}

void ViewBoxBase::invertY(const bool b)
{
    if(mYInverted==b)
        return;

    mYInverted = b;
    mMatrixNeedsUpdate = true; // updateViewRange won't detect this for us
    updateViewRange();

    emit sigStateChanged(this);
    emit sigYRangeChanged(this, mViewRange[1]);
}

void ViewBoxBase::invertX(const bool b)
{
    if(mXInverted==b)
        return;

    mXInverted = b;
    mMatrixNeedsUpdate = true; // updateViewRange won't detect this for us
    updateViewRange();

    emit sigStateChanged(this);
    emit sigYRangeChanged(this, mViewRange[0]);
}

void ViewBoxBase::setBackgroundColor(const QColor &color)
{
    mBackground->setBrush(QBrush(color));
    updateBackground();
}

QColor ViewBoxBase::backgroundColor() const
{
    return mBackground->brush().color();
}

void ViewBoxBase::updateBackground()
{
    if(mBackground->brush().color().alpha()==0)
        mBackground->hide();
    else
    {
        mBackground->show();
        mBackground->setRect(rect());
    }
}

void ViewBoxBase::setAutoPan(const bool x, const bool y)
{
    mAutoPan[0] = x;
    mAutoPan[1] = y;

    updateAutoRange();
}

void ViewBoxBase::setAspectLocked(const bool lock, const double ratio)
{
    mAspectLocked = lock ? ratio: 0.0;
}

void ViewBoxBase::setViewRange(const Point& x, const Point& y)
{
    mViewRange[0] = x;
    mViewRange[1] = y;
}

void ViewBoxBase::setTargetRange(const Point &x, const Point &y)
{
    mTargetRange[0] = x;
    mTargetRange[1] = y;
}

void ViewBoxBase::setAutoRangeEnabled(const bool enableX, const bool enableY)
{
    mAutoRangeEnabled[0] = enableX;
    mAutoRangeEnabled[1] = enableY;
}
