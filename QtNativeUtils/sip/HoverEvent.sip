
%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip

class HoverEvent: public MouseEvent
{

%TypeHeaderCode
#include "HoverEvent.h"
%End

public:
    explicit HoverEvent(const QGraphicsSceneMouseEvent* moveEvent, const bool acceptable);

    virtual ~HoverEvent();

    bool isEnter() const;

    bool isExit() const;

    bool acceptClicks(Qt::MouseButton button);

    bool acceptDrags(Qt::MouseButton button);

    void setEnter(const bool enter);

    void setExit(const bool isExit);

    QGraphicsItem* getDragItem(const Qt::MouseButton key, QGraphicsItem* defaultValue);

    QGraphicsItem* getClickItems(const Qt::MouseButton key, QGraphicsItem* defaultValue);
};
