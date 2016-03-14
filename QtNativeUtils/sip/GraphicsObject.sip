
%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip




class GraphicsObject: public QGraphicsObject, public ExtendedItem
{

%TypeHeaderCode
#include "GraphicsObject.h"
%End

public:
    GraphicsObject(QGraphicsItem* parent /TransferThis/ =nullptr) /HoldGIL/;
    virtual ~GraphicsObject() /HoldGIL/;

    virtual int type() const /HoldGIL/;

    QTransform deviceTransform() const /HoldGIL/;
    QTransform deviceTransform(const QTransform& viewportTransform) const /HoldGIL/;

    void setParentItem(QGraphicsItem* newParent) /HoldGIL/;

    virtual QTransform sceneTransform() const /HoldGIL/;

public slots:

    virtual void viewRangeChanged(const QList<Point>& range);
    virtual void viewTransformChanged();

protected:

    virtual QVariant itemChange(GraphicsItemChange change, const QVariant& value);
};
