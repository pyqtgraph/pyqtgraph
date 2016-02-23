#ifndef VIEWBOXBASE_H
#define VIEWBOXBASE_H

#include <QGraphicsWidget2.h>

class ViewBoxBase: public QGraphicsWidget2
{
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
    virtual ~ViewBoxBase() {}

    enum { Type = QGraphicsItem::UserType + 3 };

    virtual int type() const
    {
        // Enable the use of qgraphicsitem_cast with this item.
        return Type;
    }


signals:

};

#endif // VIEWBOXBASE_H
