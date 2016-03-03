#include "QGraphicsWidget2.h"

#include "ViewBoxBase.h"
#include "GraphicsViewBase.h"

QGraphicsWidget2::QGraphicsWidget2(QGraphicsItem* parent, Qt::WindowFlags wFlags) :
    QGraphicsWidget(parent, wFlags)
{
}

#define ENABLE_EXTENDEDTEM_CODE     1
#define BASE_GRAPHICSITEM_CLASS     QGraphicsWidget
#define GRAPHICSITEM_CLASS          QGraphicsWidget2
#include "ExtendedItem.cpp"
#undef ENABLE_EXTENDEDTEM_CODE
#undef BASE_GRAPHICSITEM_CLASS
#undef GRAPHICSITEM_CLASS
