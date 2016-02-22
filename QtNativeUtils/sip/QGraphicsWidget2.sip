%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip




class QGraphicsWidget2: public QGraphicsWidget
{

%TypeHeaderCode
#include "QGraphicsWidget2.h"
%End

public:
    QGraphicsWidget2(QGraphicsItem* parent /TransferThis/ =nullptr, Qt::WindowFlags wFlags=0);
    ~QGraphicsWidget2();

    virtual QGraphicsView* getViewWidget() const;

    virtual void forgetViewWidget();

    virtual int type() const;

    QTransform deviceTransform() const;
    QTransform deviceTransform(const QTransform& viewportTransform) const;

    QList<QGraphicsItem*> getBoundingParents() const;

    QVector<Point> pixelVectors() const;

    QVector<Point> pixelVectors(const QPointF& direction) const;

    double pixelLength(const QPointF& direction, const bool ortho=false) const;

    // transform mapping
    QPointF	mapFromDevice(const QPointF& point) const;
    QPointF	mapFromDevice(const QPoint& point) const;
    QPolygonF mapFromDevice(const QRectF& rect) const;
    QPolygonF mapFromDevice(const QPolygonF& polygon) const;
    QPainterPath mapFromDevice(const QPainterPath& path) const;
    QPointF	mapFromDevice(qreal x, qreal y) const;

    QPointF	mapToDevice(const QPointF& point) const;
    QPointF	mapToDevice(const QPoint& point) const;
    QPolygonF mapToDevice(const QRectF& rect) const;
    QPolygonF mapToDevice(const QPolygonF& polygon) const;
    QPainterPath mapToDevice(const QPainterPath& path) const;
    QPointF	mapToDevice(qreal x, qreal y) const;

    QRectF mapRectToDevice(const QRectF& rect) const;
    QRect mapRectToDevice(const QRect& rect) const;

    QRectF mapRectFromDevice(const QRectF& rect) const;
    QRect mapRectFromDevice(const QRect& rect) const;

    double transformAngle(QGraphicsItem* relativeItem=nullptr) const;

    virtual void mouseClickEvent(MouseClickEvent* event);
    virtual void hoverEvent(HoverEvent* event);
    virtual void mouseDragEvent(MouseDragEvent* event);

    void setFixedHeight(const double h);
    void setFixedWidth(const double w);

    double height() const;
    double width() const;

    virtual QRectF boundingRect() const;
    virtual QPainterPath shape() const;
};
