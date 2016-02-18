#include "HoverEvent.h"

HoverEvent::HoverEvent(const QGraphicsSceneMouseEvent* moveEvent, const bool acceptable)
{
    mEnter = false;
    mAcceptable = acceptable;
    mExit = false;
    mClickItems.clear();
    mDragItems.clear();
    mCurrentItem = NULL;
    if(moveEvent!=NULL)
    {
        mScenePos = moveEvent->scenePos();
        mScreenPos = moveEvent->screenPos();
        mLastScenePos = moveEvent->lastScenePos();
        mLastScreenPos = moveEvent->lastScreenPos();
        mButtons = moveEvent->buttons();
        mModifiers = moveEvent->modifiers();
    }
    else
        mExit = true;
}

