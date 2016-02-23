

class PlotItemBase: public QGraphicsWidget2
{

%TypeHeaderCode
#include "PlotItemBase.h"
%End

public:
    PlotItemBase(QGraphicsItem* parent=nullptr, Qt::WindowFlags wFlags=0);
    virtual ~PlotItemBase();


    virtual int type() const;

signals:

    void sigRangeChanged(ViewBoxBase* viewBox, const QList<Point>& range);
};
