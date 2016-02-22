
%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip



class ViewBoxInterface
{

%TypeHeaderCode
#include "Interfaces.h"
%End

public:

};




class ViewBoxGetterInterface
{

%TypeHeaderCode
#include "Interfaces.h"
%End

public:
    virtual ~ViewBoxGetterInterface();
    virtual ViewBoxInterface* getViewBox() const;
    virtual void forgetViewBox();

};





class ViewWidgetGetterInterface
{

%TypeHeaderCode
#include "Interfaces.h"
%End

public:
    virtual ~ViewWidgetGetterInterface();
    virtual QGraphicsView* getViewWidget() const;
    virtual void forgetViewWidget();
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
