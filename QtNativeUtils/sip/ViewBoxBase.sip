

class ViewBoxBase: public QGraphicsWidget2
{

%TypeHeaderCode
#include "ViewBoxBase.h"
%End

public:

    enum MouseMode
    {
        PanMode = 3,
        RectMode = 1
    };

    enum Axis
    {
        XAxis = 0,
        YAxis = 1,
        XYAxes = 2
    };

    ViewBoxBase(QGraphicsItem* parent=nullptr, Qt::WindowFlags wFlags=0);
    virtual ~ViewBoxBase();

    virtual int type() const;


signals:
    void sigStateChanged(ViewBoxBase* viewBox);
    void sigTransformChanged(ViewBoxBase* viewBox);
    void sigResized(ViewBoxBase* viewBox);
    void sigRangeChanged(ViewBoxBase* viewBox, const QList<Point>& range);
    void sigRangeChangedManually(const bool mouseLeft, const bool mouseRight);
};
