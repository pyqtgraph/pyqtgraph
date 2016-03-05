

#ifdef ENABLE_EXTENDEDTEM_CODE

#ifndef BASE_GRAPHICSITEM_CLASS
    #error "No QGraphicsItem base class defined with BASE_GRAPHICSITEM_CLASS"
#endif


virtual int type() const
{
    // Enable the use of qgraphicsitem_cast with this item.
    return Type;
}


virtual GraphicsViewBase* getViewWidget() const;

virtual void forgetViewWidget()
{
    mView = nullptr;
}

QTransform deviceTransform() const;

QTransform deviceTransform(const QTransform& viewportTransform) const
{
    return BASE_GRAPHICSITEM_CLASS::deviceTransform(viewportTransform);
}

QList<QGraphicsItem*> getBoundingParents() const
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

QVector<Point> pixelVectors() const
{
    return pixelVectors(QPointF(1.0, 0.0));
}

QVector<Point> pixelVectors(const QPointF& direction) const
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

double pixelLength(const QPointF& direction, const bool ortho=false) const
{
    QVector<Point> p = pixelVectors(direction);
    if(ortho)
        return p[1].length();
    return p[0].length();
}


QPointF	mapFromDevice(const QPointF& point) const { return deviceTransform().inverted().map(point); }
QPointF	mapFromDevice(const QPoint& point) const { return deviceTransform().inverted().map(QPointF(point)); }
QPolygonF mapFromDevice(const QRectF& rect) const { return deviceTransform().inverted().map(rect); }
QPolygonF mapFromDevice(const QPolygonF& polygon) const { return deviceTransform().inverted().map(polygon); }
QPainterPath mapFromDevice(const QPainterPath& path) const { return deviceTransform().inverted().map(path); }
QPointF	mapFromDevice(qreal x, qreal y) const { return mapFromDevice(QPointF(x, y)); }

QPointF	mapToDevice(const QPointF& point) const { return deviceTransform().map(point); }
QPointF	mapToDevice(const QPoint& point) const { return deviceTransform().map(QPointF(point)); }
QPolygonF mapToDevice(const QRectF& rect) const { return deviceTransform().map(rect); }
QPolygonF mapToDevice(const QPolygonF& polygon) const { return deviceTransform().map(polygon); }
QPainterPath mapToDevice(const QPainterPath& path) const { return deviceTransform().map(path); }
QPointF	mapToDevice(qreal x, qreal y) const { return mapToDevice(QPointF(x, y)); }

QRectF mapRectToDevice(const QRectF& rect) const { return deviceTransform().mapRect(rect); }
QRect mapRectToDevice(const QRect& rect) const { return deviceTransform().mapRect(rect); }

QRectF mapRectFromDevice(const QRectF& rect) const { return deviceTransform().inverted().mapRect(rect); }
QRect mapRectFromDevice(const QRect& rect) const { return deviceTransform().inverted().mapRect(rect); }

double transformAngle(QGraphicsItem* relativeItem=nullptr) const
{
    if(relativeItem==nullptr)
        relativeItem = parentItem();

    QTransform tr = itemTransform(relativeItem);
    QLineF vec = tr.map(QLineF(0.0, 0.0, 1.0, 0.0));
    return vec.angleTo(QLineF(vec.p1(), vec.p1()+QPointF(1.0, 0.0)));
}

virtual void mouseClickEvent(MouseClickEvent* event) { event->ignore(); }
virtual void hoverEvent(HoverEvent* event) { event->ignore(); }
virtual void mouseDragEvent(MouseDragEvent* event) { event->ignore(); }


virtual ViewBoxBase* getViewBox() const;

virtual void forgetViewBox()
{
    mViewBox = nullptr;
    mViewBoxIsViewWidget = false;
}


void setParentItem(QGraphicsItem* newParent)
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

virtual QTransform sceneTransform() const;

virtual QTransform viewTransform() const;

virtual QRectF mapRectFromView(const QRectF& r) const;

virtual QRectF viewRect() const;

protected:
    mutable GraphicsViewBase* mView = nullptr;
    mutable ViewBoxBase* mViewBox = nullptr;
    mutable bool mViewBoxIsViewWidget = false;



#endif // ENABLE_EXTENDEDTEM_CODE

