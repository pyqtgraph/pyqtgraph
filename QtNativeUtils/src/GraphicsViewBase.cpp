#include "GraphicsViewBase.h"
#include "QGraphicsScene2.h"

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

QVector<Point> GraphicsViewBase::viewRange() const
{
    QRectF range = viewRect();
    QVector<Point> p;
    p << Point(range.left(), range.right())
      << Point(range.bottom(), range.top());
    return p;
}

void GraphicsViewBase::render(QPainter *painter, const QRectF &target,
                              const QRect &source, Qt::AspectRatioMode aspectRatioMode)
{
    QGraphicsScene2* s = qobject_cast<QGraphicsScene2*>(scene());
    s->prepareForPaint();
    QGraphicsView::render(painter, target, source, aspectRatioMode);
}

void GraphicsViewBase::setAntialiasing(const bool aa)
{
    setRenderHint(QPainter::Antialiasing, aa);
}

void GraphicsViewBase::paintEvent(QPaintEvent *event)
{
    QGraphicsScene2* s = qobject_cast<QGraphicsScene2*>(scene());
    s->prepareForPaint();
    QGraphicsView::paintEvent(event);
}

