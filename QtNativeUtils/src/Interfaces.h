#ifndef INTERFACES_H
#define INTERFACES_H


#include <QGraphicsView>

#include "mouseevents/MouseClickEvent.h"
#include "mouseevents/MouseDragEvent.h"
#include "mouseevents/HoverEvent.h"

class ViewBoxInterface
{
public:

};




class ViewBoxGetterInterface
{
public:

    virtual ~ViewBoxGetterInterface() {}

    virtual ViewBoxInterface* getViewBox() const
    {
        return nullptr;
    }

    virtual void forgetViewBox()
    {
        mViewBox = nullptr;
    }

protected:

    ViewBoxInterface* mViewBox;

};





class ViewWidgetGetterInterface
{

public:
    virtual ~ViewWidgetGetterInterface() {}

    virtual QGraphicsView* getViewWidget() const
    {
        return nullptr;
    }

    virtual void forgetViewWidget() {}
};



class MouseEventsInterface
{

public:
    virtual ~MouseEventsInterface() {}

    virtual void mouseClickEvent(MouseClickEvent* event) { event->ignore(); }
    virtual void hoverEvent(HoverEvent* event) { event->ignore(); }
    virtual void mouseDragEvent(MouseDragEvent* event) { event->ignore(); }

};


#endif // INTERFACES_H
