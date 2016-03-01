
%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip


class MouseClickEvent: public MouseEvent
{

%TypeHeaderCode
#include "MouseClickEvent.h"
%End

public:
    explicit MouseClickEvent(const QGraphicsSceneMouseEvent* event, const bool doubleClick=false) /HoldGIL/;

    virtual ~MouseClickEvent() /HoldGIL/;

    void accept() /HoldGIL/;

    bool isDoubleClick() const /HoldGIL/;

    double time() const /HoldGIL/;

    QGraphicsItem* acceptedItem() /HoldGIL/;
};
