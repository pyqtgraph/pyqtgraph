#ifndef MOUSEDRAGEVENT_H
#define MOUSEDRAGEVENT_H

#include <QGraphicsSceneMouseEvent>

#include "MouseEvent.h"
#include "MouseClickEvent.h"

class MouseDragEvent: public MouseEvent
{
public:
    explicit MouseDragEvent(const QGraphicsSceneMouseEvent* moveEvent,
                            const MouseClickEvent *pressEvent, const MouseDragEvent *lastEvent,
                            const bool start=false, const bool finish=false);

    virtual ~MouseDragEvent() {}

    void accept()
    {
        // An item should call this method if it can handle the event.
        // This will prevent the event being delivered to any other items.
        MouseEvent::accept();
        mAcceptedItem = mCurrentItem;
    }

    QPointF	buttonDownPos(Qt::MouseButton button) const
    {
        if(mCurrentItem!=nullptr)
        {
            switch(button)
            {
            case Qt::LeftButton:
                return mCurrentItem->mapFromScene(mButtonDownScenePos[0]);
            case Qt::MidButton:
                return mCurrentItem->mapFromScene(mButtonDownScenePos[1]);
            case Qt::RightButton:
                return mCurrentItem->mapFromScene(mButtonDownScenePos[2]);
            default:
                return QPointF();
            }
        }
        else
            return QPointF();
    }

    QPointF	buttonDownPos() const
    {
        return buttonDownPos(mButton);
    }

    QPointF buttonDownScenePos(Qt::MouseButton button) const
    {
        switch(button)
        {
        case Qt::LeftButton:
            return mButtonDownScenePos[0];
        case Qt::MidButton:
            return mButtonDownScenePos[1];
        case Qt::RightButton:
            return mButtonDownScenePos[2];
        default:
            return QPointF();
        }
    }

    virtual QPointF	buttonDownScenePos() const
    {
        return buttonDownScenePos(mButton);
    }

    QPoint buttonDownScreenPos(Qt::MouseButton button) const
    {
        switch(button)
        {
        case Qt::LeftButton:
            return mButtonDownScreenPos[0];
        case Qt::MidButton:
            return mButtonDownScreenPos[1];
        case Qt::RightButton:
            return mButtonDownScreenPos[2];
        default:
            return QPoint();
        }
    }

    QPoint	buttonDownScreenPos() const
    {
        return buttonDownScreenPos(mButton);
    }

    bool isStart() const
    {
        return mStart;
    }

    bool isFinish() const
    {
        return mFinish;
    }

protected:

    bool mStart;
    bool mFinish;
    bool mAccepted;
    //QPointF mButtonDownPos[3];
    QPointF mButtonDownScenePos[3];
    QPoint mButtonDownScreenPos[3];
    QGraphicsItem* mAcceptedItem;

};

#endif // MOUSEDRAGEVENT_H
