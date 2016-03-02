#include "QGraphicsObject2.h"

#include "ViewBoxBase.h"

/*
QTransform QGraphicsObject2::viewTransform() const
{
    if(mViewBoxIsViewWidget)
        return sceneTransform();

    ViewBoxBase* viewBox = getViewBox();
    if(!viewBox)
        return sceneTransform();


}
*/


/*
view = self.getViewBox()
if view is None:
    return None
try:
    if view.implements('ViewBox'):
        tr = self.itemTransform(view.innerSceneItem())
        if isinstance(tr, tuple):
            tr = tr[0]  # difference between pyside and pyqt
        return tr
except:
    pass

return self.sceneTransform()
*/



#define ENABLE_EXTENDEDTEM_CODE     1
#define BASE_GRAPHICSITEM_CLASS     QGraphicsObject
#define GRAPHICSITEM_CLASS          QGraphicsObject2
#include "ExtendedItem.cpp"
#undef ENABLE_EXTENDEDTEM_CODE
#undef BASE_GRAPHICSITEM_CLASS
#undef GRAPHICSITEM_CLASS


