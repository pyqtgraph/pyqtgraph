#include "GraphicsWidget.h"

#include "GraphicsObject.h"
#include "ViewBoxBase.h"
#include "GraphicsViewBase.h"

GraphicsWidget::GraphicsWidget(QGraphicsItem* parent, Qt::WindowFlags wFlags) :
    QGraphicsWidget(parent, wFlags),
    ExtendedItem(this)
{
}

QTransform GraphicsWidget::sceneTransform() const
{
    if(scene()==nullptr)
        return transform();
    return QGraphicsWidget::sceneTransform();
}


QTransform GraphicsWidget::deviceTransform() const
{
    GraphicsViewBase* view = getViewWidget();
    if(view==nullptr)
        return QTransform();
    return QGraphicsWidget::deviceTransform(view->viewportTransform());
}

void GraphicsWidget::setParentItem(QGraphicsItem* newParent)
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


void GraphicsWidget::viewRangeChanged(const QList<Point> &range)
{
    // Called whenever the view coordinates of the ViewBox containing this item have changed.
}

void GraphicsWidget::viewTransformChanged()
{
    // Called whenever the transformation matrix of the view has changed.
    // (eg, the view range has changed or the view was resized)
}


void GraphicsWidget::disconnectView(ViewBoxBase* view)
{
    QObject::disconnect(view, SIGNAL(sigRangeChanged(QList<Point>)), this, SLOT(viewRangeChanged(QList<Point>)));
    QObject::disconnect(view, SIGNAL(sigTransformChanged()), this, SLOT(viewTransformChanged()));
}

void GraphicsWidget::disconnectView(GraphicsViewBase* view)
{
    QObject::disconnect(view, SIGNAL(sigDeviceRangeChanged(QList<Point>)), this, SLOT(viewRangeChanged(QList<Point>)));
    QObject::disconnect(view, SIGNAL(sigDeviceTransformChanged()), this, SLOT(viewTransformChanged()));
}

void GraphicsWidget::connectView(ViewBoxBase* view)
{
    QObject::connect(view, SIGNAL(sigRangeChanged(QList<Point>)), this, SLOT(viewRangeChanged(QList<Point>)));
    QObject::connect(view, SIGNAL(sigTransformChanged()), this, SLOT(viewTransformChanged()));
}

void GraphicsWidget::connectView(GraphicsViewBase* view)
{
    QObject::connect(view, SIGNAL(sigDeviceRangeChanged(QList<Point>)), this, SLOT(viewRangeChanged(QList<Point>)));
    QObject::connect(view, SIGNAL(sigDeviceTransformChanged()), this, SLOT(viewTransformChanged()));
}
