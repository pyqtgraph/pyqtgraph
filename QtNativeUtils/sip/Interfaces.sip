
%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip




class ItemChangedListener /Abstract/
{

%TypeHeaderCode
#include "Interfaces.h"
%End

public:
    virtual ~ItemChangedListener();

    virtual void itemsChanged() = 0;
};





class MouseEventsInterface
{

%TypeHeaderCode
#include "Interfaces.h"
%End

public:
    virtual ~MouseEventsInterface();

    void setAcceptCustomMouseClickEvent(const bool accept);
    void setAcceptCustomMouseHoverEvent(const bool accept);
    void setAcceptCustomMouseDragEvent(const bool accept);

    bool acceptCustomMouseClickEvent() const;
    bool acceptCustomMouseHoverEvent() const;
    bool acceptCustomMouseDragEvent() const;

    virtual void mouseClickEvent(MouseClickEvent* event);
    virtual void hoverEvent(HoverEvent* event);
    virtual void mouseDragEvent(MouseDragEvent* event);

};
