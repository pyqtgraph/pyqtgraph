#include "MouseClickEvent.h"

#include <QDebug>

MouseClickEvent::MouseClickEvent(const QGraphicsSceneMouseEvent* event, const bool doubleClick)
{
    mAccepted = false;
    mCurrentItem = NULL;
    mDoubleClick = doubleClick;
    mScenePos = event->scenePos();
    mScreenPos = event->screenPos();
    mLastScenePos = event->lastScenePos();
    mButton = event->button();
    mButtons = event->buttons();
    mModifiers = event->modifiers();
    mTime = ((double)QDateTime::currentMSecsSinceEpoch())/1000.0; // Covert from ms to s because of python compatibility
    mAcceptedItem = NULL;
}

