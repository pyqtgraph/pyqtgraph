#include "QGraphicsWidget2.h"

#include "QGraphicsObject2.h"
#include "ViewBoxBase.h"
#include "GraphicsViewBase.h"

QGraphicsWidget2::QGraphicsWidget2(QGraphicsItem* parent, Qt::WindowFlags wFlags) :
    QGraphicsWidget(parent, wFlags),
    ExtendedItem(this)
{
}

QTransform QGraphicsWidget2::sceneTransform() const
{
    if(scene()==nullptr)
        return transform();
    return QGraphicsWidget::sceneTransform();
}


QTransform QGraphicsWidget2::deviceTransform() const
{
    GraphicsViewBase* view = getViewWidget();
    if(view==nullptr)
        return QTransform();
    return QGraphicsWidget::deviceTransform(view->viewportTransform());
}

void QGraphicsWidget2::setParentItem(QGraphicsItem* newParent)
{
    // Workaround for Qt bug: https://bugreports.qt-project.org/browse/QTBUG-18616
    if(newParent!=nullptr)
    {
        QGraphicsScene* pscene = newParent->scene();
        if(pscene!=nullptr && pscene!=scene())
            pscene->addItem(this);
    }
    QGraphicsWidget::setParentItem(newParent);
}


void QGraphicsWidget2::viewRangeChanged(const QList<Point> &range)
{
    // Called whenever the view coordinates of the ViewBox containing this item have changed.
}

void QGraphicsWidget2::viewTransformChanged()
{
    // Called whenever the transformation matrix of the view has changed.
    // (eg, the view range has changed or the view was resized)
}
