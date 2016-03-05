

#ifdef ENABLE_EXTENDEDTEM_CODE

#ifndef BASE_GRAPHICSITEM_CLASS
    #error "No QGraphicsItem base class defined with BASE_GRAPHICSITEM_CLASS"
#endif

#ifndef GRAPHICSITEM_CLASS
    #error "No QGraphicsItem base class defined with GRAPHICSITEM_CLASS"
#endif


QVector<Point> GRAPHICSITEM_CLASS::pixelVectors(const QPointF& direction) const
{
    // Return vectors in local coordinates representing the width and height of a view pixel.
    // If direction is specified, then return vectors parallel and orthogonal to it.

    // Return (None, None) if pixel size is not yet defined (usually because the item has not yet been displayed)
    // or if pixel size is below floating-point precision limit.

    QVector<Point> result(2, Point(0.0, 0.0));

    QTransform devTr = deviceTransform();
    QTransform dt(devTr.m11(), devTr.m12(), devTr.m21(), devTr.m22(), 0.0, 0.0);

    if(direction.manhattanLength()==0.0)
        return result;

    QLineF dirLine; // p1 and p2 are (0, 0)
    dirLine.setP2(direction);
    dirLine = dt.map(dirLine);
    if(dirLine.length()==0.0)
        return result; // pixel size cannot be represented on this scale

    QLineF normView(dirLine.unitVector());
    QLineF normOrtho(normView.normalVector());

    QTransform dti = dt.inverted();
    result[0] = Point(dti.map(normView).p2());
    result[1] = Point(dti.map(normOrtho).p2());

    return result;
}

QList<QGraphicsItem*> GRAPHICSITEM_CLASS::getBoundingParents() const
{
    // Return a list of parents to this item that have child clipping enabled.
    QGraphicsItem* p = parentItem();
    QList<QGraphicsItem*> parents;

    while(p!=nullptr)
    {
        p = p->parentItem();
        if(p==nullptr)
            break;
        if(p->flags() & ItemClipsChildrenToShape)
            parents.append(p);
    }

    return parents;
}

double GRAPHICSITEM_CLASS::transformAngle(QGraphicsItem* relativeItem) const
{
    if(relativeItem==nullptr)
        relativeItem = parentItem();

    QTransform tr = itemTransform(relativeItem);
    QLineF vec = tr.map(QLineF(0.0, 0.0, 1.0, 0.0));
    return vec.angleTo(QLineF(vec.p1(), vec.p1()+QPointF(1.0, 0.0)));
}

void GRAPHICSITEM_CLASS::setParentItem(QGraphicsItem* newParent)
{
    // Workaround for Qt bug: https://bugreports.qt-project.org/browse/QTBUG-18616
    if(newParent!=nullptr)
    {
        QGraphicsScene* pscene = newParent->scene();
        if(pscene!=nullptr && pscene!=scene())
            pscene->addItem(this);
    }
    BASE_GRAPHICSITEM_CLASS::setParentItem(newParent);
}

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


QTransform GRAPHICSITEM_CLASS::sceneTransform() const
{
    if(scene()==nullptr)
        return transform();
    return BASE_GRAPHICSITEM_CLASS::sceneTransform();
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
    return viewTransform().inverted().mapRect(QRectF(r));
}


QTransform GRAPHICSITEM_CLASS::viewTransform() const
{
    // Return the transform that maps from local coordinates to the item's ViewBox coordinates
    // If there is no ViewBox, return the scene transform.
    // Returns None if the item does not have a view.

    ViewBoxBase* viewBox = getViewBox();
    //qDebug()<<"viewTransform"<<this<<mViewBoxIsViewWidget;
    if(mViewBoxIsViewWidget || viewBox==nullptr)
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
    bounds = mapRectFromView(bounds);

    return bounds.normalized();
}


#endif // ENABLE_EXTENDEDTEM_CODE
