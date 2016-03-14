%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip




class QGraphicsWidget2: public QGraphicsWidget, public ExtendedItem
{

%TypeHeaderCode
#include "QGraphicsWidget2.h"
%End

public:
    QGraphicsWidget2(QGraphicsItem* parent /TransferThis/ =nullptr, Qt::WindowFlags wFlags=0) /HoldGIL/;
    virtual ~QGraphicsWidget2() /HoldGIL/;

    virtual GraphicsViewBase* getViewWidget() const /HoldGIL/;
    virtual void forgetViewWidget() /HoldGIL/;

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

    QPointF	mapToView(const QPointF& point) const /HoldGIL/;
    QPointF	mapToView(const QPoint& point) const /HoldGIL/;
    QPolygonF mapToView(const QRectF& rect) const /HoldGIL/;
    QPolygonF mapToView(const QPolygonF& polygon) const /HoldGIL/;
    QPainterPath mapToView(const QPainterPath& path) const /HoldGIL/;
    QPointF	mapToView(qreal x, qreal y) const /HoldGIL/;

    QRectF mapRectToView(const QRectF& rect) const /HoldGIL/;
    QRect mapRectToView(const QRect& rect) const /HoldGIL/;

    QPointF	mapFromView(const QPointF& point) const /HoldGIL/;
    QPointF	mapFromView(const QPoint& point) const /HoldGIL/;
    QPolygonF mapFromView(const QRectF& rect) const /HoldGIL/;
    QPolygonF mapFromView(const QPolygonF& polygon) const /HoldGIL/;
    QPainterPath mapFromView(const QPainterPath& path) const /HoldGIL/;
    QPointF	mapFromView(qreal x, qreal y) const /HoldGIL/;

    QRectF mapRectFromView(const QRectF& rect) const /HoldGIL/;
    QRect mapRectFromView(const QRect& rect) const /HoldGIL/;

    double transformAngle(QGraphicsItem* relativeItem=nullptr) const /HoldGIL/;

    virtual void mouseClickEvent(MouseClickEvent* event) /HoldGIL/;
    virtual void hoverEvent(HoverEvent* event) /HoldGIL/;
    virtual void mouseDragEvent(MouseDragEvent* event) /HoldGIL/;

    void setFixedHeight(const double h) /HoldGIL/;
    void setFixedWidth(const double w) /HoldGIL/;

    double height() const /HoldGIL/;
    double width() const /HoldGIL/;

    virtual QRectF boundingRect() const /HoldGIL/;
    virtual QPainterPath shape() const /HoldGIL/;

    void setParentItem(QGraphicsItem* newParent) /HoldGIL/;

    virtual QTransform viewTransform() const /HoldGIL/;

    virtual QRectF viewRect() const /HoldGIL/;

    virtual QTransform sceneTransform() const /HoldGIL/;

    QList<QGraphicsItem*> allChildItems(QGraphicsItem* root=nullptr) const /HoldGIL/;

    QPainterPath childrenShape() const /HoldGIL/;

    QPointF pixelSize() const /HoldGIL/;
    double pixelWidth() const /HoldGIL/;
    double pixelHeight() const /HoldGIL/;

    QPointF viewPos() const /HoldGIL/;

    virtual void informViewBoundsChanged() /HoldGIL/;

    //virtual void _updateView();
    //void _replaceView(GraphicsViewBase* oldView, QGraphicsItem* item=nullptr);
    //void _replaceView(ViewBoxBase* oldView, QGraphicsItem* item=nullptr);

    void parentIsChanged();

public slots:

    virtual void viewRangeChanged(const QList<Point>& range);
    virtual void viewTransformChanged();

protected:

    virtual void viewChanged() /HoldGIL/;
};
