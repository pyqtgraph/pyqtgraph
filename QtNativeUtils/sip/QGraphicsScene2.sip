
%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip


class QGraphicsScene2: public QGraphicsScene, public ViewWidgetGetterInterface
{

%TypeHeaderCode
#include "QGraphicsScene2.h"
%End

public:
    explicit QGraphicsScene2(const double clickRadius=2.0, const double moveDistance=5.0, QObject *parent=nullptr);

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
};
