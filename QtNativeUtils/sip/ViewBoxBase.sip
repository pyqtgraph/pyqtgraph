

class ViewBoxBase: public QGraphicsWidget2
{

%TypeHeaderCode
#include "ViewBoxBase.h"
%End

public:
    ViewBoxBase(QGraphicsItem* parent=nullptr, Qt::WindowFlags wFlags=0);
    virtual ~ViewBoxBase();

    virtual int type() const;


signals:

};
