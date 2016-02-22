
%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip


class QGraphicsScene2: public QGraphicsScene
{

%TypeHeaderCode
#include <iostream>
#include "QGraphicsScene2.h"
%End


public:
    explicit QGraphicsScene2(const double clickRadius=2.0, const double moveDistance=5.0, QObject *parent /TransferThis/ =nullptr);
    virtual ~QGraphicsScene2();

    void setClickRadius(const double clickRadius);
    void setMoveDistance(const double moveDistance);

    double clickRadius() const;
    double moveDistance() const;

    void render(QPainter* painter, const QRectF& target=QRectF(), const QRectF& source=QRectF(),
                Qt::AspectRatioMode aspectRatioMode=Qt::KeepAspectRatio);

    void prepareForPaint();

    virtual QGraphicsView* getViewWidget() const;

signals:

    void sigPrepareForPaint();
    void sigMouseMoved(const QPointF& pos);
    void sigMouseHover(const QList<QGraphicsItem*>& hitems);
    void sigMouseClicked(const MouseClickEvent* event);
};
