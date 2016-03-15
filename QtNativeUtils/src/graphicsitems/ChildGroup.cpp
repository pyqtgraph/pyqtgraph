#include "ChildGroup.h"

ChildGroup::ChildGroup(QGraphicsItem *parent) :
    ItemGroup(parent)
{
    setFlag(ItemSendsGeometryChanges, true);
}

ChildGroup::~ChildGroup()
{
    mListeners.clear();
}

void ChildGroup::addListener(ItemChangedListener* listener)
{
    mListeners.append(listener);
}

QVariant ChildGroup::itemChange(QGraphicsItem::GraphicsItemChange change, const QVariant& value)
{
    if(change == ItemChildAddedChange || change == ItemChildRemovedChange)
    {
        const int size = mListeners.size();
        for(int i=0; i<size; ++i)
            mListeners[i]->itemsChanged();
    }

    return ItemGroup::itemChange(change, value);
}


/*
ret = ItemGroup.itemChange(self, change, value)
if change == self.ItemChildAddedChange or change == self.ItemChildRemovedChange:
    try:
        itemsChangedListeners = self.itemsChangedListeners
    except AttributeError:
        # It's possible that the attribute was already collected when the itemChange happened
        # (if it was triggered during the gc of the object).
        pass
    else:
        for listener in itemsChangedListeners:
            listener.itemsChanged()
return ret
*/
