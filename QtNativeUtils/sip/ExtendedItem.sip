
%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip

class ExtendedItem /Abstract/
{

%TypeHeaderCode
#include "ExtendedItem.h"
%End

public:
    ExtendedItem(QGraphicsObject* impl);
    virtual ~ExtendedItem();

    virtual GraphicsViewBase* getViewWidget() const;

    virtual void forgetViewWidget();

    QList<QGraphicsItem*> getBoundingParents() const;

    QVector<Point> pixelVectors() const;

    QVector<Point> pixelVectors(const QPointF& direction) const;

    double pixelLength(const QPointF& direction, const bool ortho=false) const;

    virtual QTransform deviceTransform() const = 0;

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

    QPointF	mapToView(const QPointF& point) const;
    QPointF	mapToView(const QPoint& point) const;
    QPolygonF mapToView(const QRectF& rect) const;
    QPolygonF mapToView(const QPolygonF& polygon) const;
    QPainterPath mapToView(const QPainterPath& path) const;
    QPointF	mapToView(qreal x, qreal y) const;

    QRectF mapRectToView(const QRectF& rect) const;
    QRect mapRectToView(const QRect& rect) const;

    QPointF	mapFromView(const QPointF& point) const;
    QPointF	mapFromView(const QPoint& point) const;
    QPolygonF mapFromView(const QRectF& rect) const;
    QPolygonF mapFromView(const QPolygonF& polygon) const;
    QPainterPath mapFromView(const QPainterPath& path) const;
    QPointF	mapFromView(qreal x, qreal y) const;

    QRectF mapRectFromView(const QRectF& rect) const;
    QRect mapRectFromView(const QRect& rect) const;

    double transformAngle(QGraphicsItem* relativeItem=nullptr) const;

    virtual void mouseClickEvent(MouseClickEvent* event);
    virtual void hoverEvent(HoverEvent* event);
    virtual void mouseDragEvent(MouseDragEvent* event);

    SIP_PYOBJECT getViewBox() const [QObject*()];
    %MethodCode
        QObject* obj = sipCpp->getViewBox();

        if(obj==nullptr)
            sipRes = sipBuildResult(&sipIsErr, "S", Py_None);
        else
        {
            ViewBoxBase* vb = qobject_cast<ViewBoxBase*>(obj);
            if(vb!=nullptr)
                //sipRes = sipBuildResult(&sipIsErr, "N", vb, sipType_ViewBoxBase, sipSelf);
                sipRes = sipConvertFromType(vb, sipType_ViewBoxBase, sipSelf);
            else
            {
                GraphicsViewBase* gv = qobject_cast<GraphicsViewBase*>(obj);
                if(gv!=nullptr)
                    //sipRes = sipBuildResult(&sipIsErr, "N", gv, sipType_GraphicsViewBase, sipSelf);
                    sipRes = sipConvertFromType(gv, sipType_GraphicsViewBase, sipSelf);
                else
                    sipRes = sipBuildResult(&sipIsErr, "S", Py_None);
            }
        }
    %End

    virtual void forgetViewBox();

    virtual QTransform viewTransform() const;

    virtual QRectF viewRect() const;

    QList<QGraphicsItem*> allChildItems(QGraphicsItem* root=nullptr) const;

    QPainterPath childrenShape() const;

    QPointF pixelSize() const;
    double pixelWidth() const;
    double pixelHeight() const;

    QPointF viewPos() const;

    virtual void informViewBoundsChanged();

    bool isViewBox(const ViewBoxBase* vb) const;
    bool isViewBox(const GraphicsViewBase* vb) const;

    void parentIsChanged();

    void setExportMode(const bool isExporting, const QVariantHash& opt=QVariantHash());

protected:

    virtual void viewChanged();

};
