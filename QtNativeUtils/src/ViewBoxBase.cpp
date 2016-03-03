#include "ViewBoxBase.h"

#include <QSizePolicy>
#include <QDebug>

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

void ViewBoxBase::itemBoundsChanged(QGraphicsItem *item)
{
    if(mAutoRangeEnabled[0] || mAutoRangeEnabled[1])
    {
        mAutoRangeNeedsUpdate = true;
        update();
    }
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

void ViewBoxBase::setAutoVisible(const bool x, const bool y)
{
    mAutoVisibleOnly[0] = x;
    if(x)
        mAutoVisibleOnly[1] = false;
    mAutoVisibleOnly[1] = y;
    if(y)
        mAutoVisibleOnly[0] = false;

    updateAutoRange();
}

void ViewBoxBase::setAspectLocked(const bool lock, const double ratio)
{
    mAspectLocked = lock ? ratio: 0.0;
}

QRectF ViewBoxBase::viewRect() const
{
    const Point& p1 = mViewRange[0];
    const Point& p2 = mViewRange[1];
    QRect r(p1.x(), p2.x(), p1.y()-p1.x(), p2.y()-p2.x());
    qDebug()<<"Rect "<<r;
    return r;
}

QRectF ViewBoxBase::targetRect() const
{
    const Point& p1 = mTargetRange[0];
    const Point& p2 = mTargetRange[1];
    return QRectF(p1.x(), p2.x(), p1.y()-p1.x(), p2.y()-p2.x());
}

QGraphicsObject2* ViewBoxBase::innerSceneItem() const
{
    //qDebug()<<"Children"<<childItems();
    qDebug()<<"innerSceneItem "<<mInnerSceneItem<< (mInnerSceneItem!=nullptr);
    return mInnerSceneItem;
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

void ViewBoxBase::_resetTarget()
{
    // Reset target range to exactly match current view range.
    // This is used during mouse interaction to prevent unpredictable
    // behavior (because the user is unaware of targetRange).
    if(mAspectLocked == 0.0)    // interferes with aspect locking
    {
        mTargetRange[0] = mViewRange[0];
        mTargetRange[1] = mViewRange[1];
    }
}

void ViewBoxBase::setInnerSceneItem(QGraphicsObject2* innerItem)
{
    qDebug()<<"setInnerSceneItem "<<(mInnerSceneItem==nullptr);
    mInnerSceneItem = innerItem;
}




/*
    def viewRect(self):
        """Return a QRectF bounding the region visible within the ViewBox"""
        try:
            viewRange = self.viewRange()
            vr0 = viewRange[0]
            vr1 = viewRange[1]
            return QtCore.QRectF(vr0[0], vr1[0], vr0[1]-vr0[0], vr1[1] - vr1[0])
        except:
            print("make qrectf failed:", self.viewRange())
            raise

    def targetRect(self):
        """
        Return the region which has been requested to be visible.
        (this is not necessarily the same as the region that is *actually* visible--
        resizing and aspect ratio constraints can cause targetRect() and viewRect() to differ)
        """
        try:
            tr = self.targetRange()
            tr0 = tr[0]
            tr1 = tr[1]
            return QtCore.QRectF(tr0[0], tr1[0], tr0[1]-tr0[0], tr1[1] - tr1[0])
        except:
            print("make qrectf failed:", tr)
            raise

    def _resetTarget(self):
        # Reset target range to exactly match current view range.
        # This is used during mouse interaction to prevent unpredictable
        # behavior (because the user is unaware of targetRange).
        if self.aspectLocked() == 0.0:  # (interferes with aspect locking)
            viewRange = self.viewRange()
            self.setTargetRange(viewRange[0], viewRange[1])
*/
