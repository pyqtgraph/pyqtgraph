#include "QGraphicsObject2.h"

#include "QGraphicsWidget2.h"
#include "ViewBoxBase.h"
#include "GraphicsViewBase.h"


QTransform QGraphicsObject2::sceneTransform() const
{
    if(scene()==nullptr)
        return transform();
    return QGraphicsObject::sceneTransform();
}


QGraphicsObject2::QGraphicsObject2(QGraphicsItem *parent) :
    QGraphicsObject(parent),
    ExtendedItem(this)
{

}

QGraphicsObject2::~QGraphicsObject2()
{

}

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


