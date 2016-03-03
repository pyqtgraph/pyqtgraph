#include "GraphicsViewBase.h"

GraphicsViewBase::GraphicsViewBase(QWidget *parent) :
    QGraphicsView(parent)
{
    setCacheMode(QGraphicsView::CacheBackground);

    // This might help, but it's probably dangerous in the general case..
    // setOptimizationFlag(DontSavePainterState, true)

    setBackgroundRole(QPalette::NoRole);

    setFocusPolicy(Qt::StrongFocus);
    setFrameShape(QFrame::NoFrame);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setTransformationAnchor(QGraphicsView::NoAnchor);
    setResizeAnchor(QGraphicsView::AnchorViewCenter);
    setViewportUpdateMode(QGraphicsView::MinimalViewportUpdate);
}

QRectF GraphicsViewBase::viewRect() const
{
    // Return the boundaries of the view in scene coordinates
    // easier to just return self.range ?
    return viewportTransform().inverted().mapRect(QRectF(rect()));
}
