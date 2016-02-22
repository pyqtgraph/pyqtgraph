#ifndef GRAPHICSSCENE_H
#define GRAPHICSSCENE_H

#include <QGraphicsScene>
#include <QVector>
#include <QAction>
#include <QString>
#include <QMouseEvent>

#include "mouseevents/MouseClickEvent.h"
#include "mouseevents/MouseDragEvent.h"
#include "mouseevents/HoverEvent.h"

#include "internal/point_utils.h"
#include "Interfaces.h"

#include "QGraphicsObject2.h"
#include "QGraphicsWidget2.h"

class QGraphicsScene2: public QGraphicsScene, public ViewWidgetGetterInterface
{
    Q_OBJECT
public:
    explicit QGraphicsScene2(const double clickRadius=2.0, const double moveDistance=5.0, QObject *parent=nullptr);
    virtual ~QGraphicsScene2();

    void setClickRadius(const double clickRadius) { mClickRadius = clickRadius; }
    void setMoveDistance(const double moveDistance) { mMoveDistance = moveDistance; }

    double clickRadius() const { return mClickRadius; }
    double moveDistance() const { return mMoveDistance; }

    void render(QPainter* painter, const QRectF& target=QRectF(), const QRectF& source=QRectF(),
                Qt::AspectRatioMode aspectRatioMode=Qt::KeepAspectRatio)
    {
        prepareForPaint();
        QGraphicsScene::render(painter, target, source, aspectRatioMode);
    }

    void prepareForPaint()
    {
        emit sigPrepareForPaint();
    }

    virtual QGraphicsView* getViewWidget() const
    {
        return views()[0];
    }

    /*
    virtual void mousePressEvent(QGraphicsSceneMouseEvent *ev);

    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent* ev);

    virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent* ev);

    virtual void mouseDoubleClickEvent(QGraphicsSceneMouseEvent* ev);

    virtual void leaveEvent(QEvent *ev)
    {
        // inform items that mouse is gone
        if(mDragButtons.size()==0)
            sendHoverEvents(ev, true);
    }

    virtual bool callMouseHover(QGraphicsItem* item, HoverEvent* ev);
    virtual bool callMouseClick(QGraphicsItem* item, MouseClickEvent* ev);
    virtual bool callMouseDrag(QGraphicsItem* item, MouseDragEvent* ev);
    */

signals:

    void sigPrepareForPaint();

    void sigMouseMoved(const QPointF& pos);

    void sigMouseHover(const QList<QGraphicsItem*>& hitems);

    void sigMouseClicked(const MouseClickEvent* event);

protected:

    /*
    virtual void sendHoverEvents(QEvent *ev, const bool exitOnly=false);

    virtual bool sendDragEvent(QGraphicsSceneMouseEvent *ev, const bool init=false, const bool final=false);

    virtual bool sendClickEvent(MouseClickEvent* ev);

    QList<QGraphicsItem*> itemsNearEvent(MouseEvent* event,
                                           const Qt::ItemSelectionMode selMode=Qt::IntersectsItemShape,
                                           const Qt::SortOrder sortOrder=Qt::DescendingOrder,
                                           const bool hoverable=false);

    MouseClickEvent* clickEventForButton(const Qt::MouseButton btn)
    {
        for(int e=0; e<mClickEvents.size(); ++e)
        {
            if(mClickEvents[e]->button() == btn)
            {
                return mClickEvents[e];
            }
        }
        return nullptr;
    }

    static bool isGraphicsObject2(QGraphicsItem* item)
    {
        return item->type()==QGraphicsObject2::Type;
    }

    static bool isGraphicsWidget2(QGraphicsItem* item)
    {
        return item->type()==QGraphicsWidget2::Type;
    }

    static bool acceptsEvents(QGraphicsItem* item)
    {
        return isGraphicsObject2(item) || isGraphicsWidget2(item);
    }

    static bool acceptsHoverEvents(QGraphicsItem* item)
    {
        if(isGraphicsObject2(item))
            return ((QGraphicsObject2*)item)->acceptCustomMouseHoverEvent();
        else if (isGraphicsWidget2(item))
            return ((QGraphicsWidget2*)item)->acceptCustomMouseHoverEvent();
        return false;
    }

    static bool acceptsDragEvents(QGraphicsItem* item)
    {
        if(isGraphicsObject2(item))
            return ((QGraphicsObject2*)item)->acceptCustomMouseDragEvent();
        else if (isGraphicsWidget2(item))
            return ((QGraphicsWidget2*)item)->acceptCustomMouseDragEvent();
        return false;
    }

    static bool acceptsClickEvents(QGraphicsItem* item)
    {
        if(isGraphicsObject2(item))
            return ((QGraphicsObject2*)item)->acceptCustomMouseClickEvent();
        else if (isGraphicsWidget2(item))
            return ((QGraphicsWidget2*)item)->acceptCustomMouseClickEvent();
        return false;
    }
    */

protected:

    double mClickRadius;
    double mMoveDistance;

    /*
    QList<MouseClickEvent*> mClickEvents;
    QList<Qt::MouseButton> mDragButtons;

    QGraphicsItem* mDragItem;
    MouseDragEvent* mLastDrag;
    QList<QGraphicsItem*> mHoverItems;
    HoverEvent* mLastHoverEvent;
    */
};

#endif // GRAPHICSSCENE_H
