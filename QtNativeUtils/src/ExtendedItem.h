

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

QList<QGraphicsItem*> getBoundingParents() const;

QVector<Point> pixelVectors() const
{
    return pixelVectors(QPointF(1.0, 0.0));
}

QVector<Point> pixelVectors(const QPointF& direction) const;

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

double transformAngle(QGraphicsItem* relativeItem=nullptr) const;

virtual void mouseClickEvent(MouseClickEvent* event) { event->ignore(); }
virtual void hoverEvent(HoverEvent* event) { event->ignore(); }
virtual void mouseDragEvent(MouseDragEvent* event) { event->ignore(); }


virtual ViewBoxBase* getViewBox() const;

virtual void forgetViewBox()
{
    mViewBox = nullptr;
    mViewBoxIsViewWidget = false;
}


void setParentItem(QGraphicsItem* newParent);

virtual QTransform sceneTransform() const;

virtual QTransform viewTransform() const;

virtual QRectF mapRectFromView(const QRectF& r) const;

virtual QRectF viewRect() const;

protected:
    mutable GraphicsViewBase* mView = nullptr;
    mutable ViewBoxBase* mViewBox = nullptr;
    mutable bool mViewBoxIsViewWidget = false;



#endif // ENABLE_EXTENDEDTEM_CODE

