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
