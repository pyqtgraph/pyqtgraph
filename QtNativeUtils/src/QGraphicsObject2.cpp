#include "QGraphicsObject2.h"

#include "QGraphicsWidget2.h"
#include "ViewBoxBase.h"
#include "GraphicsViewBase.h"


#define ENABLE_EXTENDEDTEM_CODE     1
#define BASE_GRAPHICSITEM_CLASS     QGraphicsObject
#define GRAPHICSITEM_CLASS          QGraphicsObject2
#include "ExtendedItem.cpp"
#undef ENABLE_EXTENDEDTEM_CODE
#undef BASE_GRAPHICSITEM_CLASS
#undef GRAPHICSITEM_CLASS


