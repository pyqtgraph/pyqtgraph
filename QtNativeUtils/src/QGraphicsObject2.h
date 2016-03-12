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

class ViewBoxBase;
class GraphicsViewBase;

class QGraphicsObject2: public QGraphicsObject, public ExtendedItem
{
    Q_OBJECT
public:
    QGraphicsObject2(QGraphicsItem* parent=nullptr);
    virtual ~QGraphicsObject2();

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

    virtual void viewRangeChanged(const QList<Point>& range);
    virtual void viewTransformChanged();

};

#endif // BASEGRAPHICSITEM2_H






