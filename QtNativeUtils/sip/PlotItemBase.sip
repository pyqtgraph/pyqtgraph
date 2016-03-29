

class PlotItemBase: public GraphicsWidget
{

%TypeHeaderCode
#include "PlotItemBase.h"
%End

public:
    PlotItemBase(QGraphicsItem* parent=nullptr, Qt::WindowFlags wFlags=0) /HoldGIL/;
    virtual ~PlotItemBase();


    virtual int type() const /HoldGIL/;

signals:

    void sigYRangeChanged(const Range& range);
    void sigXRangeChanged(const Range& range);
    void sigRangeChanged(const Range& xRange, const Range& yRange);
};
