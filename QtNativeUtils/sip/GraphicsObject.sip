
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

    virtual Range dataBounds(const Axis ax, const double frac=1.0, const Range orthoRange=Range()) const;

public slots:

    virtual void viewRangeChanged(const Range& xRange, const Range& yRange);
    virtual void viewTransformChanged();

protected:

    virtual QVariant itemChange(GraphicsItemChange change, const QVariant& value);
};
