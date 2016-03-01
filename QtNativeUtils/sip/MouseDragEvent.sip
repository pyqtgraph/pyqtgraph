%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip


class MouseDragEvent: public MouseEvent
{

%TypeHeaderCode
#include "MouseDragEvent.h"
%End

public:
    explicit MouseDragEvent(const QGraphicsSceneMouseEvent* moveEvent,
                            const MouseClickEvent *pressEvent, const MouseDragEvent *lastEvent,
                            const bool start=false, const bool finish=false) /HoldGIL/;

    virtual ~MouseDragEvent() /HoldGIL/;

    void accept() /HoldGIL/;

    QPointF	buttonDownPos(Qt::MouseButton button) const /HoldGIL/;

    QPointF	buttonDownPos() const /HoldGIL/;

    QPointF buttonDownScenePos(Qt::MouseButton button) const /HoldGIL/;

    virtual QPointF	buttonDownScenePos() const /HoldGIL/;

    QPoint buttonDownScreenPos(Qt::MouseButton button) const /HoldGIL/;

    QPoint	buttonDownScreenPos() const /HoldGIL/;

    bool isStart() const /HoldGIL/;

    bool isFinish() const /HoldGIL/;

};
