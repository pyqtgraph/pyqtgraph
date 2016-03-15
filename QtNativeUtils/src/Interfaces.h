#ifndef INTERFACES_H
#define INTERFACES_H

#include <iostream>
#include <QGraphicsView>

#include "mouseevents/MouseClickEvent.h"
#include "mouseevents/MouseDragEvent.h"
#include "mouseevents/HoverEvent.h"

class ViewBoxInterface
{
public:


signals:

};



class ItemChangedListener
{
public:

    virtual ~ItemChangedListener() {}

    virtual void itemsChanged() = 0;
};






class MouseEventsInterface
{

public:
    virtual ~MouseEventsInterface() {}

    void setAcceptCustomMouseClickEvent(const bool accept) { mAcceptCustomMouseClickEvent = accept; }
    void setAcceptCustomMouseHoverEvent(const bool accept) { mAcceptCustomMouseHoverEvent = accept; }
    void setAcceptCustomMouseDragEvent(const bool accept) { mAcceptCustomMouseDragEvent = accept; }

    bool acceptCustomMouseClickEvent() const { return mAcceptCustomMouseClickEvent; }
    bool acceptCustomMouseHoverEvent() const { return mAcceptCustomMouseHoverEvent; }
    bool acceptCustomMouseDragEvent() const { return mAcceptCustomMouseDragEvent; }

    virtual void mouseClickEvent(MouseClickEvent* event) { event->ignore(); }
    virtual void hoverEvent(HoverEvent* event) { event->ignore(); }
    virtual void mouseDragEvent(MouseDragEvent* event) { event->ignore(); }

protected:
    bool mAcceptCustomMouseClickEvent = false;
    bool mAcceptCustomMouseHoverEvent = false;
    bool mAcceptCustomMouseDragEvent = false;
};


#endif // INTERFACES_H
