
%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip

class MouseEvent
{

%TypeHeaderCode
#include "MouseEvent.h"
%End

public:
    MouseEvent() /HoldGIL/;

    virtual ~MouseEvent() /HoldGIL/;

    void setCurrentItem(QGraphicsItem* item) /HoldGIL/;

    void accept() /HoldGIL/;
    void ignore() /HoldGIL/;

    bool isAccepted() const /HoldGIL/;
    QPointF scenePos() const /HoldGIL/;
    QPoint screenPos() const /HoldGIL/;
    Qt::MouseButtons buttons() const /HoldGIL/;

    Qt::MouseButton button() const /HoldGIL/;

    QPointF pos() const /HoldGIL/;
    QPointF lastPos() const /HoldGIL/;
    Qt::KeyboardModifiers modifiers() const /HoldGIL/;

    virtual QPointF buttonDownScenePos() const /HoldGIL/;

    QPoint lastScreenPos() const /HoldGIL/;

};
