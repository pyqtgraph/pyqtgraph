#ifndef MOUSEEVENT_H
#define MOUSEEVENT_H


#include <QGraphicsItem>


class MouseEvent
{
public:
    MouseEvent();

    virtual ~MouseEvent()
    {
    }

    void setCurrentItem(QGraphicsItem* item)
    {
        mCurrentItem = item;
    }

    void accept()
    {
        // An item should call this method if it can handle the event.
        // This will prevent the event being delivered to any other items.
        mAccepted = true;
    }

    void ignore()
    {
        // An item should call this method if it cannot handle the event.
        // This will allow the event to be delivered to other items.
        mAccepted = false;
    }

    bool isAccepted() const { return mAccepted; }

    QPointF scenePos() const
    {
        // Return the current scene position of the mouse.
        return mScenePos;
    }

    QPoint screenPos() const
    {
        // Return the current screen position (pixels relative to widget) of the mouse.
        return mScreenPos;
    }

    Qt::MouseButtons buttons() const
    {
        // Return the buttons currently pressed on the mouse.
        return mButtons;
    }

    Qt::MouseButton button() const
    {
        // Return the mouse button that generated the click event.
        return mButton;
    }

    QPointF pos() const
    {
        if(mCurrentItem!=nullptr)
            return mCurrentItem->mapFromScene(mScenePos);
        return QPointF();
    }

    QPointF lastPos() const
    {
        // Return the previous position of the mouse in the coordinate system of the item
        // that the event was delivered to.
        if(mCurrentItem!=nullptr)
            return mCurrentItem->mapFromScene(mLastScenePos);
        return QPointF();
    }

    Qt::KeyboardModifiers modifiers() const
    {
        // Return any keyboard modifiers currently pressed.
        return mModifiers;
    }

    QPoint lastScreenPos() const
    {
        return mLastScreenPos;
    }

    virtual QPointF buttonDownScenePos() const
    {
        return mScenePos;
    }

protected:

    bool mAccepted;
    QGraphicsItem* mCurrentItem;
    QPointF mScenePos;
    QPoint mScreenPos;
    QPointF mLastScenePos;
    QPoint mLastScreenPos;
    Qt::MouseButton mButton;
    Qt::MouseButtons mButtons;
    Qt::KeyboardModifiers mModifiers;
};

#endif // MOUSEEVENT_H
