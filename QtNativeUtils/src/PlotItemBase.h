#ifndef PLOTITEM_H
#define PLOTITEM_H

#include <QList>

#include "Point.h"
#include "QGraphicsWidget2.h"
#include "ViewBoxBase.h"

class PlotItemBase: public QGraphicsWidget2
{
    Q_OBJECT
public:
    PlotItemBase(QGraphicsItem* parent=nullptr, Qt::WindowFlags wFlags=0);
    virtual ~PlotItemBase() {}

    enum { Type = QGraphicsItem::UserType + 4 };

    virtual int type() const
    {
        // Enable the use of qgraphicsitem_cast with this item.
        return Type;
    }

signals:

    void sigRangeChanged(ViewBoxBase* viewBox, const QList<Point>& range);
};

#endif // PLOTITEM_H
