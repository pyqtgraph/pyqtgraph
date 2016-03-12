
#include "ExtendedItem.h"

#include "ItemDefines.h"
#include "ViewBoxBase.h"

QVector<Point> ExtendedItem::pixelVectors(const QPointF& direction) const
{
    // Return vectors in local coordinates representing the width and height of a view pixel.
    // If direction is specified, then return vectors parallel and orthogonal to it.

    // Return (None, None) if pixel size is not yet defined (usually because the item has not yet been displayed)
    // or if pixel size is below floating-point precision limit.

    QVector<Point> result(2, Point(0.0, 0.0));

    QTransform devTr = deviceTransform();
    QTransform dt(devTr.m11(), devTr.m12(), devTr.m21(), devTr.m22(), 0.0, 0.0);

    if(direction.manhattanLength()==0.0)
        return result;

    QLineF dirLine; // p1 and p2 are (0, 0)
    dirLine.setP2(direction);
    dirLine = dt.map(dirLine);
    if(dirLine.length()==0.0)
        return result; // pixel size cannot be represented on this scale

    QLineF normView(dirLine.unitVector());
    QLineF normOrtho(normView.normalVector());

    QTransform dti = dt.inverted();
    result[0] = Point(dti.map(normView).p2());
    result[1] = Point(dti.map(normOrtho).p2());

    return result;
}

QList<QGraphicsItem*> ExtendedItem::getBoundingParents() const
{
    // Return a list of parents to this item that have child clipping enabled.
    QGraphicsItem* p = mItemImpl->parentItem();
    QList<QGraphicsItem*> parents;

    while(p!=nullptr)
    {
        p = p->parentItem();
        if(p==nullptr)
            break;
        if(p->flags() & QGraphicsItem::ItemClipsChildrenToShape)
            parents.append(p);
    }

    return parents;
}

double ExtendedItem::transformAngle(QGraphicsItem* relativeItem) const
{
    if(relativeItem==nullptr)
        relativeItem = mItemImpl->parentItem();

    QTransform tr = mItemImpl->itemTransform(relativeItem);
    QLineF vec = tr.map(QLineF(0.0, 0.0, 1.0, 0.0));
    return vec.angleTo(QLineF(vec.p1(), vec.p1()+QPointF(1.0, 0.0)));
}



GraphicsViewBase* ExtendedItem::getViewWidget() const
{
    if(mView==nullptr)
    {
        QGraphicsScene* s = mItemImpl->scene();
        if(s==nullptr)
            return nullptr;
        QList<QGraphicsView*> views = s->views();
        if(views.size()==0)
            return nullptr;

        mView = qobject_cast<GraphicsViewBase*>(views[0]);
    }

    return mView;
}


ViewBoxBase* ExtendedItem::getViewBox() const
{
    if(mViewBox==nullptr && !mViewBoxIsViewWidget)
    {
        QGraphicsItem* p = (QGraphicsItem*)mItemImpl;
        while(p!=nullptr)
        {
            p = p->parentItem();
            if(p==nullptr)
            {
                GraphicsViewBase* view = getViewWidget();
                if(view==nullptr)
                    return nullptr;
                else
                {
                    mViewBoxIsViewWidget = true;
                    return nullptr;
                }
            }
            else if(p->type()==CustomItemTypes::TypeViewBox)
            {
                mViewBox = qgraphicsitem_cast<ViewBoxBase*>(p);
                return mViewBox;
            }
        }
    }
    return mViewBox;
}


QTransform ExtendedItem::viewTransform() const
{
    // Return the transform that maps from local coordinates to the item's ViewBox coordinates
    // If there is no ViewBox, return the scene transform.
    // Returns None if the item does not have a view.

    ViewBoxBase* viewBox = getViewBox();
    if(mViewBoxIsViewWidget || viewBox==nullptr)
        return mItemImpl->sceneTransform();

    return mItemImpl->itemTransform(viewBox->innerSceneItem());
}


QRectF ExtendedItem::viewRect() const
{
    // Return the bounds (in item coordinates) of this item's ViewBox or GraphicsWidget
    ViewBoxBase* viewBox = getViewBox();
    QRectF bounds;
    if(viewBox)
        bounds = viewBox->viewRect();
    else
        bounds = mView->viewRect();
    bounds = mapRectFromView(bounds);

    return bounds.normalized();
}


QList<QGraphicsItem*> ExtendedItem::allChildItems(QGraphicsItem* root) const
{
    QList<QGraphicsItem*> tree;
    if(root==nullptr)
        tree = mItemImpl->childItems();
    else
        tree = root->childItems();

    int index = 0;
    while(index<tree.size())
    {
        tree += tree[index]->childItems();
        index += 1;
    }
    return tree;
}


QPainterPath ExtendedItem::childrenShape() const
{
    QList<QGraphicsItem*> chItems = allChildItems();
    QPainterPath path;
    const int size = chItems.size();
    for(int i=0; i<size; ++i)
    {
        QGraphicsItem* c = chItems[i];
        path += mItemImpl->mapFromItem(c, c->shape());
    }
    return path;
}


QPointF ExtendedItem::pixelSize() const
{
    return deviceTransform().inverted().map(QPointF(1.0, 1.0));
}

double ExtendedItem::pixelWidth() const
{
    return deviceTransform().inverted().map(QLineF(0.0, 0.0, 1.0, 0.0)).length();
}

double ExtendedItem::pixelHeight() const
{
    return deviceTransform().inverted().map(QLineF(0.0, 0.0, 0.0, 1.0)).length();
}

QPointF ExtendedItem::viewPos() const
{
    return mapToView(mItemImpl->mapFromParent(mItemImpl->pos()));
}


void ExtendedItem::informViewBoundsChanged()
{
    // Inform this item's container ViewBox that the bounds of this item have changed.
    // This is used by ViewBox to react if auto-range is enabled.
    ViewBoxBase* viewBox = getViewBox();
    if(viewBox)
        viewBox->itemBoundsChanged(mItemImpl); // inform view so it can update its range if it wants
}


void ExtendedItem::_updateView()
{

}


void ExtendedItem::viewChanged()
{
    // Called when this item's view has changed
    // (ie, the item has been added to or removed from a ViewBox)
}


bool ExtendedItem::isViewBox(const ViewBoxBase* vb) const
{
    return mViewBox == vb;
}

bool ExtendedItem::isViewBox(const GraphicsViewBase* vb) const
{
    return (mViewBoxIsViewWidget && mView==vb);
}


void ExtendedItem::_replaceView(GraphicsViewBase* oldView, QGraphicsItem* item)
{
    if(item==nullptr)
        item = (QGraphicsItem*)mItemImpl;
    QList<QGraphicsItem*> children = item->childItems();
    const int count = children.size();
    for(int i=0; i<count; ++i)
    {
        ExtendedItem* itemObject = dynamic_cast<ExtendedItem*>(children[i]);
        if(itemObject)
            itemObject->_updateView();
        else
            _replaceView(oldView, children[i]);

        /*
        if(children[i]->type()<QGraphicsItem::UserType)
        {
            // We are dealing with standard item
            _replaceView(oldView, children[i]);
        }
        else
        {
            // _updateView()
            ExtendedItem* itemObject = dynamic_cast<ExtendedItem*>(children[i]);
            if(itemObject)
            {
                itemObject->_updateView();
            }
        }
        */
    }
}

void ExtendedItem::_replaceView(ViewBoxBase* oldView, QGraphicsItem* item)
{
    if(item==nullptr)
        item = (QGraphicsItem*)mItemImpl;
    QList<QGraphicsItem*> children = item->childItems();
    const int count = children.size();
    for(int i=0; i<count; ++i)
    {
        ExtendedItem* itemObject = dynamic_cast<ExtendedItem*>(children[i]);
        if(itemObject)
            itemObject->_updateView();
        else
            _replaceView(oldView, children[i]);

        /*
        if(children[i]->type()<QGraphicsItem::UserType)
        {
            // We are dealing with standard item
            _replaceView(oldView, children[i]);
        }
        else
        {
            // _updateView()
            ExtendedItem* itemObject = dynamic_cast<ExtendedItem*>(children[i]);
            if(itemObject)
            {
                itemObject->_updateView();
            }
        }
        */
    }
}


