
%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip

class MouseEvent
{

%TypeHeaderCode
#include "MouseEvent.h"
%End

public:
    MouseEvent();

    virtual ~MouseEvent();

    void setCurrentItem(QGraphicsItem* item);

    void accept();
    void ignore();

    bool isAccepted() const;
    QPointF scenePos() const;
    QPoint screenPos() const;
    Qt::MouseButtons buttons() const;

    Qt::MouseButton button() const;

    QPointF pos() const;
    QPointF lastPos() const;
    Qt::KeyboardModifiers modifiers() const;

    virtual QPointF buttonDownScenePos() const;

};
