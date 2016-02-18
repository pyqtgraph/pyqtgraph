
%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip


class MouseClickEvent: public MouseEvent
{

%TypeHeaderCode
#include "MouseClickEvent.h"
%End

public:
    explicit MouseClickEvent(const QGraphicsSceneMouseEvent* event, const bool doubleClick=false);

    virtual ~MouseClickEvent();

    void accept();

    bool isDoubleClick() const;

    qint64 time() const;
};
