%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip

class GraphicsViewBase : public QGraphicsView
{

%TypeHeaderCode
#include "GraphicsViewBase.h"
%End

public:
    explicit GraphicsViewBase(QWidget *parent=0);

    virtual QRectF viewRect() const;

    virtual QRectF visibleRange() const;

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
