#ifndef QGRAPHICSWIDGET2_H
#define QGRAPHICSWIDGET2_H

#include <QGraphicsWidget>
#include <QGraphicsView>

#include "Point.h"
#include "ExtendedItem.h"

class QGraphicsWidget2: public QGraphicsWidget
{
public:
    QGraphicsWidget2(QGraphicsItem* parent=nullptr, Qt::WindowFlags wFlags=0);
    ~QGraphicsWidget2() {}

#define ENABLE_EXTENDEDTEM_CODE     1
#define BASE_GRAPHICSITEM_CLASS     QGraphicsObject
#include "ExtendedItem.h"
#undef ENABLE_EXTENDEDTEM_CODE
#undef BASE_GRAPHICSITEM_CLASS

};

#endif // QGRAPHICSWIDGET2_H
