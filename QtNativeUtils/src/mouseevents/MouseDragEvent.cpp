#include "MouseDragEvent.h"


MouseDragEvent::MouseDragEvent(const QGraphicsSceneMouseEvent *moveEvent,
                               const MouseClickEvent *pressEvent, const MouseDragEvent *lastEvent,
                               const bool start, const bool finish)
{
    mStart = start;
    mFinish = finish;
    mAccepted = false;
    mCurrentItem = NULL;
    mButtonDownPos[0] = moveEvent->buttonDownPos(Qt::LeftButton);
    mButtonDownPos[1] = moveEvent->buttonDownPos(Qt::MidButton);
    mButtonDownPos[2] = moveEvent->buttonDownPos(Qt::RightButton);
    mButtonDownScenePos[0] = moveEvent->buttonDownScenePos(Qt::LeftButton);
    mButtonDownScenePos[1] = moveEvent->buttonDownScenePos(Qt::MidButton);
    mButtonDownScenePos[2] = moveEvent->buttonDownScenePos(Qt::RightButton);
    mButtonDownScreenPos[0] = moveEvent->buttonDownScreenPos(Qt::LeftButton);
    mButtonDownScreenPos[1] = moveEvent->buttonDownScreenPos(Qt::MidButton);
    mButtonDownScreenPos[2] = moveEvent->buttonDownScreenPos(Qt::RightButton);
    mScenePos = moveEvent->scenePos();
    mScreenPos = moveEvent->screenPos();
    if(lastEvent==NULL)
    {
        mLastScenePos = pressEvent->scenePos();
        mLastScreenPos = pressEvent->screenPos();
    }
    else
    {
        mLastScenePos = lastEvent->scenePos();
        mLastScreenPos = lastEvent->screenPos();
    }
    mButtons = moveEvent->buttons();
    mButton = pressEvent->button();
    mModifiers = moveEvent->modifiers();
    mAcceptedItem = NULL;
}


