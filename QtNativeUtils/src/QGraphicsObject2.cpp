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
