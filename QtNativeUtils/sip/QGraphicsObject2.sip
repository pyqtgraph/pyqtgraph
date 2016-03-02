
%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip




class QGraphicsObject2: public QGraphicsObject
{

%TypeHeaderCode
#include "QGraphicsObject2.h"
%End

public:
    QGraphicsObject2(QGraphicsItem* parent /TransferThis/ =nullptr) /HoldGIL/;
    virtual ~QGraphicsObject2() /HoldGIL/;

    virtual QGraphicsView* getViewWidget() const /HoldGIL/;
    virtual void forgetViewWidget() /HoldGIL/;

/*
    virtual QObject* getViewBox();
%MethodCode
    if(sipSelfWasArg)
        sipRes = sipCpp->QGraphicsObject2::getViewBox();
    else
        sipRes = sipCpp->getViewBox();

    if(sipRes->isWidgetType())
        return sipConvertFromType(sipRes, sipType_QGraphicsView, nullptr);
    else
        return sipConvertFromType(sipRes, sipType_QGraphicsWidget, nullptr);
%End

    virtual void forgetViewBox();
*/

    virtual int type() const /HoldGIL/;

    QTransform deviceTransform() const /HoldGIL/;
    QTransform deviceTransform(const QTransform& viewportTransform) const /HoldGIL/;

    QList<QGraphicsItem*> getBoundingParents() const /HoldGIL/;

    QVector<Point> pixelVectors() const /HoldGIL/;

    QVector<Point> pixelVectors(const QPointF& direction) const /HoldGIL/;

    double pixelLength(const QPointF& direction, const bool ortho=false) const /HoldGIL/;

    // transform mapping
    QPointF	mapFromDevice(const QPointF& point) const /HoldGIL/;
    QPointF	mapFromDevice(const QPoint& point) const /HoldGIL/;
    QPolygonF mapFromDevice(const QRectF& rect) const /HoldGIL/;
    QPolygonF mapFromDevice(const QPolygonF& polygon) const /HoldGIL/;
    QPainterPath mapFromDevice(const QPainterPath& path) const /HoldGIL/;
    QPointF	mapFromDevice(qreal x, qreal y) const /HoldGIL/;

    QPointF	mapToDevice(const QPointF& point) const /HoldGIL/;
    QPointF	mapToDevice(const QPoint& point) const /HoldGIL/;
    QPolygonF mapToDevice(const QRectF& rect) const /HoldGIL/;
    QPolygonF mapToDevice(const QPolygonF& polygon) const /HoldGIL/;
    QPainterPath mapToDevice(const QPainterPath& path) const /HoldGIL/;
    QPointF	mapToDevice(qreal x, qreal y) const /HoldGIL/;

    QRectF mapRectToDevice(const QRectF& rect) const /HoldGIL/;
    QRect mapRectToDevice(const QRect& rect) const /HoldGIL/;

    QRectF mapRectFromDevice(const QRectF& rect) const /HoldGIL/;
    QRect mapRectFromDevice(const QRect& rect) const /HoldGIL/;

    double transformAngle(QGraphicsItem* relativeItem=nullptr) const /HoldGIL/;

    virtual void mouseClickEvent(MouseClickEvent* event) /HoldGIL/;
    virtual void hoverEvent(HoverEvent* event) /HoldGIL/;
    virtual void mouseDragEvent(MouseDragEvent* event) /HoldGIL/;

    void setParentItem(QGraphicsItem* newParent);

    virtual QTransform viewTransform() const;

    virtual QRectF mapRectFromView(const QRectF& r) const;
};
