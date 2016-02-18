#ifndef HOVEREVENT_H
#define HOVEREVENT_H

#include <QGraphicsSceneMouseEvent>
#include <QGraphicsItem>
#include <QMap>

#include "MouseEvent.h"

class HoverEvent: public MouseEvent
{
public:
    explicit HoverEvent(const QGraphicsSceneMouseEvent* moveEvent, const bool acceptable);

    virtual ~HoverEvent() {}

    bool isEnter() const
    {
        return mEnter;
    }

    bool isExit() const
    {
        return mExit;
    }

    bool acceptClicks(Qt::MouseButton button)
    {
        // Inform the scene that the item (that the event was delivered to)
        // would accept a mouse click event if the user were to click before
        // moving the mouse again.

        // Returns True if the request is successful, otherwise returns False (indicating
        // that some other item would receive an incoming click).
        if(mAcceptable==false)
            return false;

        if(mClickItems.contains(button))
            return false;

        mClickItems.insert(button, mCurrentItem);
        return true;
    }

    bool acceptDrags(Qt::MouseButton button)
    {
        // Inform the scene that the item (that the event was delivered to)
        // would accept a mouse drag event if the user were to drag before
        // the next hover event.
        //
        // Returns True if the request is successful, otherwise returns False (indicating
        // that some other item would receive an incoming drag event).

        if(mAcceptable==false)
            return false;

        if(mDragItems.contains(button))
            return false;

        mDragItems.insert(button, mCurrentItem);
        return true;
    }

    void setEnter(const bool enter)
    {
        mEnter = enter;
        mExit = !enter;
    }

    void setExit(const bool isExit)
    {
        mEnter = !isExit;
        mExit = isExit;
    }

    QGraphicsItem* getDragItem(const Qt::MouseButton key, QGraphicsItem* defaultValue)
    {
        return mDragItems.value(key, defaultValue);
    }

    QGraphicsItem* getClickItems(const Qt::MouseButton key, QGraphicsItem* defaultValue)
    {
        return mClickItems.value(key, defaultValue);
    }

protected:

    bool mEnter;
    bool mAcceptable;
    bool mExit;
    QMap<Qt::MouseButton, QGraphicsItem*> mClickItems;
    QMap<Qt::MouseButton, QGraphicsItem*> mDragItems;
};

#endif // HOVEREVENT_H
