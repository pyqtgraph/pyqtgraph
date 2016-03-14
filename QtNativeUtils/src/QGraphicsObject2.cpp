#include "QGraphicsObject2.h"

#include "QGraphicsWidget2.h"
#include "ViewBoxBase.h"
#include "GraphicsViewBase.h"


QGraphicsObject2::QGraphicsObject2(QGraphicsItem *parent) :
    QGraphicsObject(parent),
    ExtendedItem(this)
{
    setFlag(ItemSendsGeometryChanges, true);
}

QGraphicsObject2::~QGraphicsObject2()
{

}

QTransform QGraphicsObject2::sceneTransform() const
{
    if(scene()==nullptr)
        return transform();
    return QGraphicsObject::sceneTransform();
}

void QGraphicsObject2::viewRangeChanged(const QList<Point> &range)
{
    // Called whenever the view coordinates of the ViewBox containing this item have changed.
}

void QGraphicsObject2::viewTransformChanged()
{
    // Called whenever the transformation matrix of the view has changed.
    // (eg, the view range has changed or the view was resized)
}

QVariant QGraphicsObject2::itemChange(QGraphicsItem::GraphicsItemChange change, const QVariant &value)
{
    QVariant ret = QGraphicsObject::itemChange(change, value);

    if(change==ItemParentHasChanged || change==ItemSceneHasChanged)
        parentIsChanged();
    else if(change==ItemPositionHasChanged || change==ItemTransformHasChanged)
        informViewBoundsChanged();

    return ret;
}


/*
ret = QGraphicsObject2.itemChange(self, change, value)
if change in [self.ItemParentHasChanged, self.ItemSceneHasChanged]:
    self.parentIsChanged()
try:
    inform_view_on_change = self.__inform_view_on_changes
except AttributeError:
    # It's possible that the attribute was already collected when the itemChange happened
    # (if it was triggered during the gc of the object).
    pass
else:
    if inform_view_on_change and change in [self.ItemPositionHasChanged, self.ItemTransformHasChanged]:
        self.informViewBoundsChanged()

## workaround for pyqt bug:
## http://www.riverbankcomputing.com/pipermail/pyqt/2012-August/031818.html
if not USE_PYSIDE and change == self.ItemParentChange and isinstance(ret, QtGui.QGraphicsItem):
    ret = sip.cast(ret, QtGui.QGraphicsItem)

return ret
*/

QTransform QGraphicsObject2::deviceTransform() const
{
    GraphicsViewBase* view = getViewWidget();
    if(view==nullptr)
        return QTransform();
    return QGraphicsObject::deviceTransform(view->viewportTransform());
}


void QGraphicsObject2::setParentItem(QGraphicsItem* newParent)
{
    // Workaround for Qt bug: https://bugreports.qt-project.org/browse/QTBUG-18616
    if(newParent!=nullptr)
    {
        QGraphicsScene* pscene = newParent->scene();
        if(pscene!=nullptr && pscene!=scene())
            pscene->addItem(this);
    }
    QGraphicsObject::setParentItem(newParent);
}

void QGraphicsObject2::disconnectView(ViewBoxBase* view)
{
    QObject::disconnect(view, SIGNAL(sigRangeChanged(QList<Point>)), this, SLOT(viewRangeChanged(QList<Point>)));
    QObject::disconnect(view, SIGNAL(sigTransformChanged()), this, SLOT(viewTransformChanged()));
}

void QGraphicsObject2::disconnectView(GraphicsViewBase* view)
{
    QObject::disconnect(view, SIGNAL(sigDeviceRangeChanged(QList<Point>)), this, SLOT(viewRangeChanged(QList<Point>)));
    QObject::disconnect(view, SIGNAL(sigDeviceTransformChanged()), this, SLOT(viewTransformChanged()));
}

void QGraphicsObject2::connectView(ViewBoxBase* view)
{
    QObject::connect(view, SIGNAL(sigRangeChanged(QList<Point>)), this, SLOT(viewRangeChanged(QList<Point>)));
    QObject::connect(view, SIGNAL(sigTransformChanged()), this, SLOT(viewTransformChanged()));
}

void QGraphicsObject2::connectView(GraphicsViewBase* view)
{
    QObject::connect(view, SIGNAL(sigDeviceRangeChanged(QList<Point>)), this, SLOT(viewRangeChanged(QList<Point>)));
    QObject::connect(view, SIGNAL(sigDeviceTransformChanged()), this, SLOT(viewTransformChanged()));
}

