%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip




class GraphicsWidget: public QGraphicsWidget, public ExtendedItem
{

%TypeHeaderCode
#include "GraphicsWidget.h"
%End

public:
    GraphicsWidget(QGraphicsItem* parent /TransferThis/ =nullptr, Qt::WindowFlags wFlags=0) /HoldGIL/;
    virtual ~GraphicsWidget() /HoldGIL/;

    virtual int type() const /HoldGIL/;

    QTransform deviceTransform() const /HoldGIL/;
    QTransform deviceTransform(const QTransform& viewportTransform) const /HoldGIL/;

    void setFixedHeight(const double h) /HoldGIL/;
    void setFixedWidth(const double w) /HoldGIL/;

    double height() const /HoldGIL/;
    double width() const /HoldGIL/;

    virtual QRectF boundingRect() const /HoldGIL/;
    virtual QPainterPath shape() const /HoldGIL/;

    void setParentItem(QGraphicsItem* newParent) /HoldGIL/;

    virtual QTransform sceneTransform() const /HoldGIL/;

public slots:

    virtual void viewRangeChanged(const QList<Point>& range);
    virtual void viewTransformChanged();

protected:

};
