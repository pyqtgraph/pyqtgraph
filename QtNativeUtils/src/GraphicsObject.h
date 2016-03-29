#ifndef BASEGRAPHICSITEM2_H
#define BASEGRAPHICSITEM2_H

#include <QGraphicsView>
#include <QGraphicsObject>
#include <QDebug>
#include <QList>

#include "Point.h"
#include "Interfaces.h"
#include "ItemDefines.h"
#include "ExtendedItem.h"
#include "Range.h"

class ViewBoxBase;
class GraphicsViewBase;

class GraphicsObject: public QGraphicsObject, public ExtendedItem
{
    Q_OBJECT
public:
    GraphicsObject(QGraphicsItem* parent=nullptr);
    virtual ~GraphicsObject();

    enum { Type = CustomItemTypes::TypeGraphicsObject };

    virtual int type() const
    {
        // Enable the use of qgraphicsitem_cast with this item.
        return Type;
    }

    QTransform deviceTransform() const;

    QTransform deviceTransform(const QTransform& viewportTransform) const
    {
        return QGraphicsObject::deviceTransform(viewportTransform);
    }

    void setParentItem(QGraphicsItem* newParent);

    virtual QTransform sceneTransform() const;

public slots:

    virtual void viewRangeChanged(const Range& xRange, const Range& yRange);
    virtual void viewTransformChanged();


protected:

    virtual QVariant itemChange(GraphicsItemChange change, const QVariant& value);

    virtual void disconnectView(ViewBoxBase* view);
    virtual void disconnectView(GraphicsViewBase* view);

    virtual void connectView(ViewBoxBase* view);
    virtual void connectView(GraphicsViewBase* view);

};

#endif // BASEGRAPHICSITEM2_H






