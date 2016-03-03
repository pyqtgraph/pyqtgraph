

#ifdef ENABLE_EXTENDEDTEM_CODE

#ifndef BASE_GRAPHICSITEM_CLASS
    #error "No QGraphicsItem base class defined with BASE_GRAPHICSITEM_CLASS"
#endif

#ifndef GRAPHICSITEM_CLASS
    #error "No QGraphicsItem base class defined with GRAPHICSITEM_CLASS"
#endif


GraphicsViewBase* GRAPHICSITEM_CLASS::getViewWidget() const
{
    if(mView==nullptr)
    {
        QGraphicsScene* s = scene();
        if(s==nullptr)
            return nullptr;
        QList<QGraphicsView*> views = s->views();
        if(views.size()==0)
            return nullptr;

        mView = qobject_cast<GraphicsViewBase*>(views[0]);
    }

    return mView;
}


ViewBoxBase* GRAPHICSITEM_CLASS::getViewBox() const
{
    if(mViewBox==nullptr && !mViewBoxIsViewWidget)
    {
        QGraphicsItem* p = (QGraphicsItem*)this;
        while(p!=nullptr)
        {
            p = p->parentItem();
            if(p==nullptr)
            {
                GraphicsViewBase* view = getViewWidget();
                if(view==nullptr)
                    return nullptr;
                else
                {
                    mViewBoxIsViewWidget = true;
                    return nullptr;
                }
            }
            else if(p->type()==CustomItemTypes::TypeViewBox)
            {
                mViewBox = qgraphicsitem_cast<ViewBoxBase*>(p);
                return mViewBox;
            }
        }
    }
    return mViewBox;
}



QTransform GRAPHICSITEM_CLASS::deviceTransform() const
{
    GraphicsViewBase* view = getViewWidget();
    if(view==nullptr)
        return QTransform();
    return BASE_GRAPHICSITEM_CLASS::deviceTransform(view->viewportTransform());
}


QRectF GRAPHICSITEM_CLASS::mapRectFromView(const QRectF& r) const
{
    return viewTransform().inverted().mapRect(r);
}


QTransform GRAPHICSITEM_CLASS::viewTransform() const
{
    // Return the transform that maps from local coordinates to the item's ViewBox coordinates
    // If there is no ViewBox, return the scene transform.
    // Returns None if the item does not have a view.

    ViewBoxBase* viewBox = getViewBox();
    qDebug()<<this<<mViewBoxIsViewWidget;
    if(mViewBoxIsViewWidget || !viewBox)
        return sceneTransform();

    return itemTransform(viewBox->innerSceneItem());
}


QRectF GRAPHICSITEM_CLASS::viewRect() const
{
    // Return the bounds (in item coordinates) of this item's ViewBox or GraphicsWidget
    ViewBoxBase* viewBox = getViewBox();
    QRectF bounds;
    if(viewBox)
        bounds = viewBox->viewRect();
    else
        bounds = mView->viewRect();

    //return bounds.normalized();
    return bounds;
}


/*
view = self.getViewBox()
if view is None:
    return None
bounds = self.mapRectFromView(view.viewRect())
if bounds is None:
    return None

bounds = bounds.normalized()

return bounds
*/















#endif // ENABLE_EXTENDEDTEM_CODE
