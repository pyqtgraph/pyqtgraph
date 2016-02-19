
%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip




class QGraphicsObject2: public QGraphicsObject
{

%TypeHeaderCode
#include "QGraphicsObject2.h"
%End

public:
    QGraphicsObject2(QGraphicsItem* parent=nullptr);
    ~QGraphicsObject2();

    QGraphicsView* getViewWidget() const;

    void forgetViewWidget();

    QTransform deviceTransform() const;
    QTransform deviceTransform(const QTransform& viewportTransform) const;

    QList<QGraphicsItem*> getBoundingParents() const;

    QVector<Point> pixelVectors() const;

    QVector<Point> pixelVectors(const QPointF& direction) const;

    double pixelLength(const QPointF& direction, const bool ortho=false) const;

};
