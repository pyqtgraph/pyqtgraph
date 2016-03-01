
%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip

class HoverEvent: public MouseEvent
{

%TypeHeaderCode
#include "HoverEvent.h"
%End

public:
    explicit HoverEvent(const QGraphicsSceneMouseEvent* moveEvent, const bool acceptable) /HoldGIL/;

    virtual ~HoverEvent() /HoldGIL/;

    bool isEnter() const /HoldGIL/;

    bool isExit() const /HoldGIL/;

    bool acceptClicks(Qt::MouseButton button) /HoldGIL/;

    bool acceptDrags(Qt::MouseButton button) /HoldGIL/;

    void setEnter(const bool enter) /HoldGIL/;

    void setExit(const bool isExit) /HoldGIL/;

    QGraphicsItem* getDragItem(const Qt::MouseButton key, QGraphicsItem* defaultValue) /HoldGIL/;

    QGraphicsItem* getClickItems(const Qt::MouseButton key, QGraphicsItem* defaultValue) /HoldGIL/;
};
