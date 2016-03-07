#ifndef GRAPHICSVIEWBASE_H
#define GRAPHICSVIEWBASE_H

#include <QGraphicsView>
#include <QMouseEvent>

class GraphicsViewBase : public QGraphicsView
{
    Q_OBJECT
public:
    explicit GraphicsViewBase(QWidget *parent = 0);

    virtual QRectF viewRect() const;

    virtual QRectF visibleRange() const { return viewRect(); }

    void render(QPainter* painter, const QRectF& target=QRectF(),
                const QRect& source=QRect(),
                Qt::AspectRatioMode aspectRatioMode=Qt::KeepAspectRatio);

protected:

    virtual void paintEvent(QPaintEvent* event);

signals:

    void sigDeviceRangeChanged(GraphicsViewBase* view, const QRectF& r);
    void sigDeviceTransformChanged(GraphicsViewBase* view);
    void sigMouseReleased(QMouseEvent& ev);
    void sigSceneMouseMoved(const QPointF& p);
    void sigScaleChanged(GraphicsViewBase* view);

};

#endif // GRAPHICSVIEWBASE_H
