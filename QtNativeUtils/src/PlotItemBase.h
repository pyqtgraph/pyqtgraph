#ifndef PLOTITEM_H
#define PLOTITEM_H

#include <QList>

#include "Point.h"
#include "GraphicsWidget.h"
#include "ViewBoxBase.h"
#include "ItemDefines.h"

class PlotItemBase: public GraphicsWidget
{
    Q_OBJECT
public:
    PlotItemBase(QGraphicsItem* parent=nullptr, Qt::WindowFlags wFlags=0);
    virtual ~PlotItemBase() {}

    enum { Type = CustomItemTypes::TypePlotItem };

    virtual int type() const
    {
        // Enable the use of qgraphicsitem_cast with this item.
        return Type;
    }

signals:

    void sigYRangeChanged(const Range& range);
    void sigXRangeChanged(const Range& range);
    void sigRangeChanged(const Range& xRange, const Range& yRange);
};

#endif // PLOTITEM_H
