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

class QGraphicsScene2 : public QGraphicsScene
{
    Q_OBJECT
public:
    explicit QGraphicsScene2(const double clickRadius=2.0, const double moveDistance=5.0, QObject *parent=0);

    void setClickRadius(const double clickRadius) { mClickRadius = clickRadius; }
    void setMoveDistance(const double moveDistance) { mMoveDistance = moveDistance; }

    void render(QPainter* painter, const QRectF& target=QRectF(), const QRectF& source=QRectF(),
                Qt::AspectRatioMode aspectRatioMode=Qt::KeepAspectRatio)
    {
        prepareForPaint();
        QGraphicsScene::render(painter, target, source, aspectRatioMode);
    }

    virtual void mousePressEvent(QGraphicsSceneMouseEvent *ev);

    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent* ev);

    virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent* ev);

    virtual void mouseDoubleClickEvent(QGraphicsSceneMouseEvent* ev);

    QGraphicsView* getViewWidget() const
    {
        return views()[0];
    }

    // def addParentContextMenus(self, item, menu, event)
    // def getContextMenus(self, event)
    // def showExportDialog(self)

signals:

    void sigPrepareForPaint();

    void sigMouseMoved(const QPointF& pos);

    void sigMouseHover(const QVector<QGraphicsItem*>& hitems);

    void sigMouseClicked(const MouseClickEvent* event);

protected:

    void sendHoverEvents(QGraphicsSceneMouseEvent *ev, const bool exitOnly=false);

    bool sendDragEvent(QGraphicsSceneMouseEvent *ev, const bool init=false, const bool final=false);

    bool sendClickEvent(MouseClickEvent* ev);

    void leaveEvent(QGraphicsSceneMouseEvent *ev)
    {
        // inform items that mouse is gone
        if(mDragButtons.size()==0)
            sendHoverEvents(ev, true);
    }

    void prepareForPaint()
    {
        emit sigPrepareForPaint();
    }

    QVector<QGraphicsItem*> itemsNearEvent(MouseEvent* event,
                                           const Qt::ItemSelectionMode selMode=Qt::IntersectsItemShape,
                                           const Qt::SortOrder sortOrder=Qt::DescendingOrder,
                                           const bool hoverable=false);

private:

    MouseClickEvent* clickEventForButton(const Qt::MouseButton btn)
    {
        for(int e=0; e<mClickEvents.size(); ++e)
        {
            if(mClickEvents[e]->button() == btn)
            {
                return mClickEvents[e];
            }
        }
        return NULL;
    }

protected:

    double mClickRadius;
    double mMoveDistance;

    QVector<MouseClickEvent*> mClickEvents;
    QVector<Qt::MouseButton> mDragButtons;

    QGraphicsItem* mDragItem;
    MouseDragEvent* mLastDrag;
    QVector<QGraphicsItem*> mHoverItems;
    HoverEvent* mLastHoverEvent;
};

#endif // GRAPHICSSCENE_H
