

class PlotItemBase: public QGraphicsWidget2
{

%TypeHeaderCode
#include "PlotItemBase.h"
%End

public:
    PlotItemBase(QGraphicsItem* parent=nullptr, Qt::WindowFlags wFlags=0) /HoldGIL/;
    virtual ~PlotItemBase();


    virtual int type() const /HoldGIL/;

signals:

    void sigYRangeChanged(const Point& range);
    void sigXRangeChanged(const Point& range);
    void sigRangeChanged(const QList<Point>& range);
};
