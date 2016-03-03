#ifndef GRAPHICSVIEWBASE_H
#define GRAPHICSVIEWBASE_H

#include <QGraphicsView>

class GraphicsViewBase : public QGraphicsView
{
    Q_OBJECT
public:
    explicit GraphicsViewBase(QWidget *parent = 0);

signals:

public slots:

};

#endif // GRAPHICSVIEWBASE_H
