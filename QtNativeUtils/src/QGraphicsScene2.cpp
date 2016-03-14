#include "QGraphicsScene2.h"

#include <algorithm>

#include <QGraphicsView>

#include "Point.h"
#include "Interfaces.h"

static double absZValue(QGraphicsItem* item)
{
    if(item==nullptr)
        return 0;
    return item->zValue() + absZValue(item->parentItem());
}


static bool absZvalueCompareDescending(QGraphicsItem* aItem, QGraphicsItem* bItem)
{
    return absZValue(bItem) < absZValue(aItem);
}


QGraphicsScene2::QGraphicsScene2(const double clickRadius, const double moveDistance, QObject* parent) :
    QGraphicsScene(parent)
{
    setClickRadius(clickRadius);
    setMoveDistance(moveDistance);

    //mDragItem = nullptr;
    //mLastDrag = nullptr;
    //mLastHoverEvent = nullptr;
}

QGraphicsScene2::~QGraphicsScene2()
{
    /*
    for(int i=0; i<mClickEvents.size(); ++i)
        delete mClickEvents[i];
    if(mLastDrag!=nullptr)
        delete mLastDrag;
    if(mLastHoverEvent!=nullptr)
        delete mLastHoverEvent;
    */
}

/*
void QGraphicsScene2::mousePressEvent(QGraphicsSceneMouseEvent *ev)
{
    QGraphicsScene::mousePressEvent(ev);
    if(mouseGrabberItem()==nullptr)
    {
        // nobody claimed press; we are free to generate drag/click events
        if(mLastHoverEvent!=nullptr)
        {
            // If the mouse has moved since the last hover event, send a new one.
            // This can happen if a context menu is open while the mouse is moving.
            if(ev->scenePos()!=mLastHoverEvent->scenePos())
                sendHoverEvents(ev);
        }

        mClickEvents.append(new MouseClickEvent(ev));

        // set focus on the topmost focusable item under this click
        QList<QGraphicsItem*> focusableItems = items(ev->scenePos());
        int count = focusableItems.size();
        for(int i=0; i<count; ++i)
        {
            QGraphicsItem* fitem = focusableItems[i];
            if(fitem->isEnabled() && fitem->isVisible() &&
               (fitem->flags() & QGraphicsItem::ItemIsFocusable))
            {
                fitem->setFocus(Qt::MouseFocusReason);
                break;
            }
        }
    }
}

void QGraphicsScene2::mouseMoveEvent(QGraphicsSceneMouseEvent *ev)
{
    emit sigMouseMoved(ev->scenePos());

    // First allow QGraphicsScene to deliver hoverEnter/Move/ExitEvents
    QGraphicsScene::mouseMoveEvent(ev);

    // Next deliver our own HoverEvents
    sendHoverEvents(ev);

    if(ev->buttons() != Qt::NoButton)
    {
        // button is pressed; send mouseMoveEvents and mouseDragEvents
        QGraphicsScene::mouseMoveEvent(ev);
        if(mouseGrabberItem()==nullptr)
        {
            qint64 now = QDateTime::currentMSecsSinceEpoch();
            bool init = false;
            // keep track of which buttons are involved in dragging
            Qt::MouseButton mbuttons[3] = {Qt::LeftButton, Qt::MidButton, Qt::RightButton};
            for(int i=0; i<3; ++i)
            {
                Qt::MouseButton btn = mbuttons[i];
                if((ev->buttons() & btn)==0)
                    continue;
                if(mDragButtons.contains(btn)==false)
                {
                    // see if we've dragged far enough yet
                    MouseClickEvent* cev = clickEventForButton(btn);

                    const double dist = distance(ev->screenPos(), cev->screenPos());
                    const qint64 dt = now - cev->time();
                    if(dist < mMoveDistance && dt<500)
                        continue;

                    // If this is the first button to be dragged, then init=True
                    init = init | (mDragButtons.size()==0);
                    mDragButtons.append(btn);
                }
            }

            // If we have dragged buttons, deliver a drag event
            if(mDragButtons.size()>0)
                if(sendDragEvent(ev, init))
                    ev->accept();
        }
    }
}

void QGraphicsScene2::mouseReleaseEvent(QGraphicsSceneMouseEvent *ev)
{
    if(mouseGrabberItem()==nullptr)
    {
        if(mDragButtons.contains(ev->button()))
        {
            // Send drag event
            if(sendDragEvent(ev, false, true))
                ev->accept();
            mDragButtons.removeAll(ev->button());
        }
        else
        {
            // Send click event
            MouseClickEvent* cev = clickEventForButton(ev->button());
            if(sendClickEvent(cev))
                ev->accept();
            mClickEvents.removeAll(cev);
            delete cev;
        }
    }

    if(ev->buttons()==Qt::NoButton)
    {
        mDragItem = nullptr;
        mDragButtons.clear();
        for(int i=0; i<mClickEvents.size(); ++i)
            delete mClickEvents[i];
        mClickEvents.clear();
        delete mLastDrag;
        mLastDrag = nullptr;
    }

    QGraphicsScene::mouseReleaseEvent(ev);

    // let items prepare for next click/drag
    sendHoverEvents(ev);
}


void QGraphicsScene2::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *ev)
{
    QGraphicsScene::mouseDoubleClickEvent(ev);
    if(mouseGrabberItem()==nullptr)
        mClickEvents.append(new MouseClickEvent(ev, true));
}

bool QGraphicsScene2::callMouseHover(QGraphicsItem *item, HoverEvent *ev)
{
    std::cout<<"Calling C++"<<std::endl;
    if(isGraphicsObject2(item))
        ((GraphicsObject*)item)->hoverEvent(ev);
    else if(isGraphicsWidget2(item))
        ((QGraphicsWidget2*)item)->hoverEvent(ev);
    else
        return false;
    return true;
}

bool QGraphicsScene2::callMouseClick(QGraphicsItem *item, MouseClickEvent *ev)
{
    std::cout<<"Calling C++"<<std::endl;
    if(isGraphicsObject2(item))
        ((GraphicsObject*)item)->mouseClickEvent(ev);
    else if(isGraphicsWidget2(item))
        ((QGraphicsWidget2*)item)->mouseClickEvent(ev);
    else
        return false;
    return true;
}

bool QGraphicsScene2::callMouseDrag(QGraphicsItem *item, MouseDragEvent *ev)
{
    std::cout<<"Calling C++"<<std::endl;
    if(isGraphicsObject2(item))
        ((GraphicsObject*)item)->mouseDragEvent(ev);
    else if(isGraphicsWidget2(item))
        ((QGraphicsWidget2*)item)->mouseDragEvent(ev);
    else
        return false;
    return true;
}


void QGraphicsScene2::sendHoverEvents(QEvent *ev, const bool exitOnly)
{
    // if exitOnly, then just inform all previously hovered items that the mouse has left.

    // if we are in mid-drag, do not allow items to accept the hover event.
    bool acceptable = false;
    QList<QGraphicsItem*> nearItems;
    HoverEvent* hevent = nullptr;

    if(exitOnly)
    {
        acceptable = false;
        hevent = new HoverEvent(nullptr, acceptable);
    }
    else
    {
        acceptable = ((QGraphicsSceneMouseEvent*)ev)->buttons() == Qt::NoButton;
        hevent = new HoverEvent((QGraphicsSceneMouseEvent*)ev, acceptable);
        nearItems = itemsNearEvent(hevent, Qt::IntersectsItemShape, Qt::DescendingOrder, true);
        emit sigMouseHover(nearItems);
    }

    QList<QGraphicsItem*> prevItems(mHoverItems);

    for(int i=0; i<nearItems.size(); ++i)
    {
        QGraphicsItem* item = nearItems[i];
        if(acceptsHoverEvents(item))
        {
            hevent->setCurrentItem(item);
            if(!mHoverItems.contains(item))
            {
                hevent->setEnter(true);
                mHoverItems.append(item);
            }
            else
            {
                prevItems.removeAt(i);
                hevent->setEnter(false);
            }

            callMouseHover(item, hevent);
        }
    }

    hevent->setEnter(false);
    hevent->setExit(true);
    for(int i=0; i<prevItems.size(); ++i)
    {
        QGraphicsItem* item = prevItems[i];
        hevent->setCurrentItem(item);
        this->callMouseHover(item, hevent);
        mHoverItems.removeAll(item);
    }

    // Update last hover event unless:
    //   - mouse is dragging (move+buttons); in this case we want the dragged
    //     item to continue receiving events until the drag is over
    //   - event is not a mouse event (QEvent.Leave sometimes appears here)
    if (ev->type() == QGraphicsSceneMouseEvent::GraphicsSceneMousePress ||
        (ev->type() == QGraphicsSceneMouseEvent::GraphicsSceneMouseMove && ((QGraphicsSceneMouseEvent*)ev)->buttons()==Qt::NoButton))
    {
        // save this so we can ask about accepted events later.
        if(mLastHoverEvent!=nullptr)
            delete mLastHoverEvent;
        mLastHoverEvent = hevent;
    }
    else
    {
        // Delete event
        delete hevent;
    }
}


bool QGraphicsScene2::sendDragEvent(QGraphicsSceneMouseEvent* ev, const bool init, const bool final)
{
    // Send a MouseDragEvent to the current dragItem or to
    // items near the beginning of the drag
    MouseDragEvent* devent = new MouseDragEvent(ev, mClickEvents[0], mLastDrag, init, final);
    if(init && mDragItem==nullptr)
    {
        QGraphicsItem* acceptedItem = nullptr;
        if(mLastHoverEvent!=nullptr)
            acceptedItem = mLastHoverEvent->getDragItem(devent->button(), nullptr);

        if(acceptedItem!=nullptr)
        {
            mDragItem = acceptedItem;
            devent->setCurrentItem(acceptedItem);
            //dynamic_cast<MouseEventsInterface*>(mDragItem)->mouseDragEvent(event); // To implement
            callMouseDrag(mDragItem, devent);
        }
        else
        {
            QList<QGraphicsItem*> nearItems = itemsNearEvent(devent);
            for (int i=0; i<nearItems.size(); ++i)
            {
                QGraphicsItem* item = nearItems[0];
                if(!item->isVisible() || !item->isEnabled())
                    continue;

                if(acceptsDragEvents(item))
                {
                    devent->setCurrentItem(item);
                    //dynamic_cast<MouseEventsInterface*>(item)->mouseDragEvent(event);  // To implement
                    callMouseDrag(mDragItem, devent);
                    if(devent->isAccepted())
                    {
                        mDragItem = item;
                        if(item->flags() & QGraphicsItem::ItemIsFocusable)
                            item->setFocus(Qt::MouseFocusReason);
                        break;
                    }
                }
            }
        }
    }
    else if(mDragItem!=nullptr)
    {
        devent->setCurrentItem(mDragItem);
        //dynamic_cast<MouseEventsInterface*>(mDragItem)->mouseDragEvent(devent); // To implement
        callMouseDrag(mDragItem, devent);
    }

    if(mLastDrag!=nullptr)
        delete mLastDrag;
    mLastDrag = devent;
    return devent->isAccepted();
}


bool QGraphicsScene2::sendClickEvent(MouseClickEvent *ev)
{
    // if we are in mid-drag, click events may only go to the dragged item.
    if(mDragItem!=nullptr)
    {
        ev->setCurrentItem(mDragItem);
        //dynamic_cast<MouseEventsInterface*>(mDragItem)->mouseClickEvent(ev); // To implement
        callMouseClick(mDragItem, ev);
    }
    else
    {
        // otherwise, search near the cursor
        QGraphicsItem* acceptedItem = nullptr;
        if(mLastHoverEvent!=nullptr)
            acceptedItem = mLastHoverEvent->getDragItem(ev->button(), nullptr);

        if(acceptedItem!=nullptr)
        {
            ev->setCurrentItem(acceptedItem);
            //dynamic_cast<MouseEventsInterface*>(acceptedItem)->mouseClickEvent(ev); // To implement
            callMouseClick(acceptedItem, ev);
        }
        else
        {
            QList<QGraphicsItem*> nearItems = itemsNearEvent(ev);
            for (int i=0; i<nearItems.size(); ++i)
            {
                QGraphicsItem* item = nearItems[0];
                if(!item->isVisible() || !item->isEnabled())
                    continue;
                if(acceptsClickEvents(item))
                {
                    ev->setCurrentItem(item);
                    //dynamic_cast<MouseEventsInterface*>(item)->mouseClickEvent(ev);  // To implement
                    callMouseClick(item, ev);
                    if(ev->isAccepted())
                    {
                        if(item->flags() & QGraphicsItem::ItemIsFocusable)
                            item->setFocus(Qt::MouseFocusReason);
                        break;
                    }
                }
            }
        }
    }

    emit sigMouseClicked(ev);
    return ev->isAccepted();
}


QList<QGraphicsItem *> QGraphicsScene2::itemsNearEvent(MouseEvent *event,
                                                       const Qt::ItemSelectionMode selMode,
                                                       const Qt::SortOrder sortOrder,
                                                       const bool hoverable)
{
    // Return an iterator that iterates first through the items that directly intersect point (in Z order)
    // followed by any other items that are within the scene's click radius.

    QGraphicsView* view = views()[0];
    QTransform tr = view->viewportTransform();
    QRectF rect = view->mapToScene(0, 0, 2*mClickRadius, 2*mClickRadius).boundingRect();

    QPointF point = event->buttonDownScenePos();
    double w = rect.width();
    double h = rect.height();
    QRectF rgn(point.x()-w, point.y()-h, 2.0*w, 2.0*h);

    QList<QGraphicsItem*> selItems = items(point, selMode, sortOrder, tr);

    QList<QGraphicsItem*> selItems2;
    for(int i=0; i<selItems.size(); ++i)
    {
        if(!hoverable)
            continue;
        QPainterPath shape = selItems[i]->shape();
        if(!shape.isEmpty() && shape.contains(point))
            selItems2.append(selItems[i]);
    }

    // Sort by descending Z-order (don't trust scene.itms() to do this either)
    // use 'absolute' z value, which is the sum of all item/parent ZValues
    std::sort(selItems2.begin(), selItems2.end(), absZvalueCompareDescending);

    return selItems2;
}
*/
