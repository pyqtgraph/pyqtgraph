
%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip




class QGraphicsObject2: public QGraphicsObject, public ExtendedItem
{

%TypeHeaderCode
#include "QGraphicsObject2.h"
%End

public:
    QGraphicsObject2(QGraphicsItem* parent /TransferThis/ =nullptr) /HoldGIL/;
    virtual ~QGraphicsObject2() /HoldGIL/;

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
