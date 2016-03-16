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
    setFlag(ItemClipsChildrenToShape);
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
    const Point& p1 = mViewRange[0];
    const Point& p2 = mViewRange[1];
    QRectF r(p1.x(), p2.x(), p1.y()-p1.x(), p2.y()-p2.x());
    return r;
}

QRectF ViewBoxBase::targetRect() const
{
    const Point& p1 = mTargetRange[0];
    const Point& p2 = mTargetRange[1];
    return QRectF(p1.x(), p2.x(), p1.y()-p1.x(), p2.y()-p2.x());
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

QTransform ViewBoxBase::childTransform()
{
    // Return the transform that maps from child(item in the childGroup) coordinates to local coordinates.
    // (This maps from inside the viewbox to outside)

    if(mMatrixNeedsUpdate)
        updateMatrix();

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

/*
"""
Add a QGraphicsItem to this view. The view will include this item when determining how to set its range
automatically unless *ignoreBounds* is True.
"""
if item.zValue() < self.zValue():
    item.setZValue(self.zValue()+1)
scene = self.scene()
if scene is not None and scene is not item.scene():
    scene.addItem(item)  ## Necessary due to Qt bug: https://bugreports.qt-project.org/browse/QTBUG-18616
item.setParentItem(self.getChildGroup())
if not ignoreBounds:
    self.addedItems.append(item)
self.updateAutoRange()
*/

void ViewBoxBase::removeItem(QGraphicsItem* item)
{
    // Remove an item from this view.
    mAddedItems.removeOne(item);
    scene()->removeItem(item);
    updateAutoRange();
}

/*
"""Remove an item from this view."""
try:
    self.addedItems.remove(item)
except:
    pass
self.scene().removeItem(item)
self.updateAutoRange()

*/

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

/*
for i in self.addedItems[:]:
    self.removeItem(i)
for ch in self.getChildGroup().childItems():
    ch.setParentItem(None)
*/


void ViewBoxBase::prepareForPaint()
{
    // don't check whether auto range is enabled here--only check when setting dirty flag.
    if(mAutoRangeNeedsUpdate) // and autoRangeEnabled
        updateAutoRange();
    if(mMatrixNeedsUpdate)
        updateMatrix();
}

void ViewBoxBase::setViewRange(const Point& x, const Point& y)
{
    mViewRange[0] = Point(x);
    mViewRange[1] = Point(y);
}

void ViewBoxBase::setTargetRange(const Point &x, const Point &y)
{
    mTargetRange[0] = Point(x);
    mTargetRange[1] = Point(y);
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
        mTargetRange[0] = Point(mViewRange[0]);
        mTargetRange[1] = Point(mViewRange[1]);
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
