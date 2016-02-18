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
                            const bool start=false, const bool finish=false);

    virtual ~MouseDragEvent();

    void accept();

    QPointF	buttonDownPos(Qt::MouseButton button) const;

    QPointF	buttonDownPos() const;

    QPointF buttonDownScenePos(Qt::MouseButton button) const;

    virtual QPointF	buttonDownScenePos() const;

    QPoint buttonDownScreenPos(Qt::MouseButton button) const;

    QPoint	buttonDownScreenPos() const;

    bool isStart() const;

    bool isFinish() const;

};
