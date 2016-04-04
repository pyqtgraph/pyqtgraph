#include "ViewBoxBase.h"

#include <QSizePolicy>
#include <QDebug>

#include "QGraphicsScene2.h"
#include "graphicsitems/ChildGroup.h"

ViewBoxBase::ViewBoxBase(QGraphicsItem *parent, Qt::WindowFlags wFlags, const bool invertX, const bool invertY) :
    GraphicsWidget(parent, wFlags),
    mMatrixNeedsUpdate(true),
    mAutoRangeNeedsUpdate(true),
    mXInverted(invertX),
    mYInverted(invertY)
{
    Range::registerMetatype();

    setFlag(ItemClipsChildrenToShape);
    setFlag(ItemIsFocusable, true);

    setZValue(-100);
    setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding));

    mViewRange.clear();
    mViewRange << Range(0.0, 1.0) << Range(0.0, 1.0);

    mTargetRange.clear();
    mTargetRange << Range(0.0, 1.0) << Range(0.0, 1.0);

    mAutoRangeEnabled.clear();
    mAutoRangeEnabled << true << true;

    mAutoPan.clear();
    mAutoPan << false << false;

    mChildGroup = new ChildGroup(this);
    mChildGroup->addListener(this);
    setInnerSceneItem(mChildGroup);

    mBackground = new QGraphicsRectItem(rect());
    mBackground->setParentItem(this);
    mBackground->setZValue(-1e6);
    mBackground->setPen(QPen(Qt::NoPen));
    mBackground->setBrush(QBrush(QColor(Qt::transparent)));
    updateBackground();
}

void ViewBoxBase::updateMatrix()
{
    QRectF vr = viewRect();
    if(vr.height()==0.0 || vr.width()==0.0)
        return;

    QRectF bounds = rect();

    QPointF scale(bounds.width()/vr.width(), bounds.height()/vr.height());
    if(!mYInverted)
        scale.ry() *= -1.0;
    if(mXInverted)
        scale.rx() *= -1.0;

    QTransform m;

    // First center the viewport at 0
    QPointF center = bounds.center();
    m.translate(center.x(), center.y());

    // Now scale and translate properly
    m.scale(scale.x(), scale.y());
    center = vr.center();
    m.translate(-center.x(), -center.y());

    mChildGroup->setTransform(m);

    emit sigTransformChanged();
    mMatrixNeedsUpdate = false;
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
    emit sigYRangeChanged(mViewRange[1]);
}

void ViewBoxBase::invertX(const bool b)
{
    if(mXInverted==b)
        return;

    mXInverted = b;
    mMatrixNeedsUpdate = true; // updateViewRange won't detect this for us
    updateViewRange();

    emit sigStateChanged(this);
    emit sigYRangeChanged(mViewRange[0]);
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
    const Range& p1 = mViewRange[0];
    const Range& p2 = mViewRange[1];
    QRectF r(p1.min(), p2.min(), p1.max()-p1.min(), p2.max()-p2.min());
    return r;
}

QRectF ViewBoxBase::targetRect() const
{
    const Range& p1 = mTargetRange[0];
    const Range& p2 = mTargetRange[1];
    return QRectF(p1.min(), p2.min(), p1.max()-p1.min(), p2.max()-p2.min());
}

GraphicsObject* ViewBoxBase::innerSceneItem() const
{
    return mInnerSceneItem;
}

void ViewBoxBase::itemsChanged()
{
    updateAutoRange();
}

ChildGroup *ViewBoxBase::getChildGroup() const
{
    return mChildGroup;
}

QTransform ViewBoxBase::childTransform() const
{
    // Return the transform that maps from child(item in the childGroup) coordinates to local coordinates.
    // (This maps from inside the viewbox to outside)

    if(mMatrixNeedsUpdate)
        const_cast<ViewBoxBase*>(this)->updateMatrix();

    return mChildGroup->transform();
}

const QList<QGraphicsItem *> &ViewBoxBase::addedItems() const
{
    return mAddedItems;
}

void ViewBoxBase::addItem(QGraphicsItem *item, const bool ignoreBounds)
{
    // Add a QGraphicsItem to this view. The view will include this item when determining how to set its range
    // automatically unless *ignoreBounds* is True.

    if(item->zValue() < zValue())
        item->setZValue(zValue()+1.0);

    QGraphicsScene* gScene = scene();
    if(gScene!=nullptr && gScene!=item->scene())
        gScene->addItem(item);
    item->setParentItem(mChildGroup);
    if(!ignoreBounds)
        mAddedItems.append(item);
    updateAutoRange();
}

void ViewBoxBase::removeItem(QGraphicsItem* item)
{
    // Remove an item from this view.
    mAddedItems.removeOne(item);
    scene()->removeItem(item);
    updateAutoRange();
}

void ViewBoxBase::clear()
{
    int count = mAddedItems.size();
    for(int i=0; i<count; ++i)
        scene()->removeItem(mAddedItems[i]);

    QList<QGraphicsItem*> cItems = mChildGroup->childItems();
    count = cItems.size();
    for(int i=0; i<count; ++i)
        cItems[i]->setParentItem(nullptr);

    mAddedItems.clear();
}

double ViewBoxBase::suggestPadding(const ViewBoxBase::Axis ax) const
{
    const double l = ax==Axis::XAxis ? width() : height();
    if(l>0.0)
        return std::min(std::max(1.0/std::sqrt(l), 0.02), 0.1);
    return 0.02;
}

void ViewBoxBase::enableAutoRange(const ViewBoxBase::Axis axis, const bool enable)
{
    // Enable (or disable) auto-range for *axis*, which may be ViewBox.XAxis, ViewBox.YAxis, or ViewBox.XYAxes for both
    // (if *axis* is omitted, both axes will be changed).
    // When enabled, the axis will automatically rescale when items are added/removed or change their shape.
    // The argument *enable* may optionally be a float (0.0-1.0) which indicates the fraction of the data that should
    // be visible (this only works with items implementing a dataRange method, such as PlotDataItem).


    if(axis==XYAxes || axis==XAxis)
    {
        if(mAutoRangeEnabled[0] != enable)
        {
            // If we are disabling, do one last auto-range to make sure that
            // previously scheduled auto-range changes are enacted
            if(!enable && mAutoRangeNeedsUpdate)
                updateAutoRange();

           mAutoRangeEnabled[0] = enable;
           mAutoRangeNeedsUpdate |= enable;
           update();
        }
    }

    if(axis==XYAxes || axis==YAxis)
    {
        if(mAutoRangeEnabled[1] != enable)
        {
            // If we are disabling, do one last auto-range to make sure that
            // previously scheduled auto-range changes are enacted
            if(!enable && mAutoRangeNeedsUpdate)
                updateAutoRange();

           mAutoRangeEnabled[1] = enable;
           mAutoRangeNeedsUpdate |= enable;
           update();
        }
    }

    if(mAutoRangeNeedsUpdate)
        updateAutoRange();

    emit sigStateChanged(this);

    /*
    if axis == ViewBox.XYAxes or axis == 'xy':
        axes = [0, 1]
    elif axis == ViewBox.XAxis or axis == 'x':
        axes = [0]
    elif axis == ViewBox.YAxis or axis == 'y':
        axes = [1]
    else:
        raise Exception('axis argument must be ViewBox.XAxis, ViewBox.YAxis, or ViewBox.XYAxes.')

    for ax in axes:
        are = self.autoRangeEnabled()
        if are[ax] != enable:
            # If we are disabling, do one last auto-range to make sure that
            # previously scheduled auto-range changes are enacted
            if enable is False and self.autoRangeNeedsUpdate():
                self.updateAutoRange()

            are[ax] = enable
            self.setAutoRangeEnabled(are[0], are[1])
            self.setAutoRangeNeedsUpdate(self.autoRangeNeedsUpdate() or (enable is not False))
            self.update()


    if self.autoRangeNeedsUpdate():
        self.updateAutoRange()

    self.sigStateChanged.emit(self)
    */
}

void ViewBoxBase::enableAutoRange(const QString& axis, const bool enable)
{
    if(axis=="xy")
        enableAutoRange(XYAxes, enable);
    else if(axis=="x")
        enableAutoRange(XAxis, enable);
    else if(axis=="y")
        enableAutoRange(YAxis, enable);
}

QRectF ViewBoxBase::itemBoundingRect(const QGraphicsItem *item) const
{
    // Return the bounding rect of the item in view coordinates
    return mapSceneToView(item->sceneBoundingRect()).boundingRect();
}

void ViewBoxBase::setRange(const Range& xRange, const Range& yRange, const double padding, const bool disableAutoRange)
{
    const Range range[2] {xRange, yRange};
    const bool changes[2] {xRange.isValid(), yRange.isValid()};

    // Update axes one at a time
    bool changed[2] {false, false};
    for(int i=0; i<2; ++i)
    {
        if(range[i].isValid())
            continue;

        double mn = range[i].min();
        double mx = range[i].max();
        double xpad = padding;

        // If we requested 0 range, try to preserve previous scale.
        // Otherwise just pick an arbitrary scale.
        if(mn==mx)
        {
            double dy = mViewRange[i].max() - mViewRange[i].min();
            dy = (dy==0.0) ? 1.0 : dy;
            mn -= dy*0.5;
            mx += dy*0.5;
            xpad = 0.0;
        }

        if(!std::isfinite(xpad))
            xpad = suggestPadding(i);

        double p = (mx-mn) * xpad;
        mn -= p;
        mx += p;

        // Set target range
        Range tr(mn, mx);
        if(!tr.finiteEqual(mTargetRange[i]))
        {
            mTargetRange[i] = tr;
            changed[i] = true;
        }
    }

    // Update viewRange to match targetRange as closely as possible while
    // accounting for aspect ratio constraint
    if(changes[0] && changes[1])
        updateViewRange(false, false);
    else
        updateViewRange(changes[0], changes[1]);

    if(disableAutoRange)
    {
        if(changed[0])
            enableAutoRange(XAxis, false);
        if(changed[1])
            enableAutoRange(YAxis, false);
    }

    if(changed[0] || changed[1])
        emit sigStateChanged(this);

    if(changed[0] && mAutoVisibleOnly[0] && mAutoRangeEnabled[0])
        setAutoRangeNeedsUpdate(true);
    else if(changed[1] && mAutoVisibleOnly[1] && mAutoRangeEnabled[1])
        setAutoRangeNeedsUpdate(true);
}

void ViewBoxBase::setRange(const QRectF &rect, const double padding, const bool disableAutoRange)
{
    Range xRange;
    if(rect.width()>0)
        xRange.setRange(rect.left(), rect.right());
    Range yRange;
    if(rect.height()>0)
        xRange.setRange(rect.bottom(), rect.top());

    setRange(xRange, yRange, padding, disableAutoRange);
}

void ViewBoxBase::prepareForPaint()
{
    // don't check whether auto range is enabled here--only check when setting dirty flag.
    if(mAutoRangeNeedsUpdate) // and autoRangeEnabled
        updateAutoRange();
    if(mMatrixNeedsUpdate)
        updateMatrix();
}

void ViewBoxBase::setViewRange(const Range& x, const Range& y)
{
    mViewRange[0] = Range(x);
    mViewRange[1] = Range(y);
}

void ViewBoxBase::setTargetRange(const Range &x, const Range &y)
{
    mTargetRange[0] = Range(x);
    mTargetRange[1] = Range(y);
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
        mTargetRange[0] = Range(mViewRange[0]);
        mTargetRange[1] = Range(mViewRange[1]);
    }
}

void ViewBoxBase::setInnerSceneItem(GraphicsObject* innerItem)
{
    mInnerSceneItem = innerItem;
}

QVariant ViewBoxBase::itemChange(QGraphicsItem::GraphicsItemChange change, const QVariant &value)
{
    if(change == QGraphicsItem::ItemSceneChange)
    {
        // Disconnect from old scene
        QGraphicsScene2* oldScene = qobject_cast<QGraphicsScene2*>(scene());
        if(oldScene)
            QObject::disconnect(oldScene, SIGNAL(sigPrepareForPaint()), this, SLOT(prepareForPaint()));
    } else if (change == QGraphicsItem::ItemSceneHasChanged)
    {
        // Connect to a new scene
        QGraphicsScene2* newScene = qobject_cast<QGraphicsScene2*>(scene());
        if(newScene)
            QObject::connect(newScene, SIGNAL(sigPrepareForPaint()), this, SLOT(prepareForPaint()));
    }

    return GraphicsWidget::itemChange(change, value);
}
