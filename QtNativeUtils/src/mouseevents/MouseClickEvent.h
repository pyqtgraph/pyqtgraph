#ifndef MOUSECLICKEVENT_H
#define MOUSECLICKEVENT_H

#include <QGraphicsSceneMouseEvent>
#include <QDateTime>
#include <QGraphicsItem>

#include "MouseEvent.h"


class MouseClickEvent: public MouseEvent
{
public:
    explicit MouseClickEvent(const QGraphicsSceneMouseEvent* event, const bool doubleClick=false);

    virtual ~MouseClickEvent() {}

    void accept()
    {
        // An item should call this method if it can handle the event.
        // This will prevent the event being delivered to any other items.
        MouseEvent::accept();
        mAcceptedItem = mCurrentItem;
    }

    bool isDoubleClick() const
    {
        // Return True if this is a double-click.
        return mDoubleClick;
    }

    double time() const
    {
        return mTime;
    }

    QGraphicsItem* acceptedItem()
    {
        return mAcceptedItem;
    }


protected:

    bool mDoubleClick;
    double mTime;
    QGraphicsItem* mAcceptedItem;
};

#endif // MOUSECLICKEVENT_H
