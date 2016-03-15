/***************************************************************************
 *   15/03/2016                                                            *
 *                                                                         *
 *   Copyright (C) 2016 Henesis s.r.l.                                     *
 *                                                                         *
 *   www.henesis.eu                                                        *
 *                                                                         *
 *   Alessandro Bacchini - alessandro.bacchini@henesis.eu                  *
 *                                                                         *
 ***************************************************************************/
#include "ItemGroup.h"

ItemGroup::ItemGroup(QGraphicsItem* parent) :
    GraphicsObject(parent)
{

}

ItemGroup::~ItemGroup()
{

}

QRectF ItemGroup::boundingRect() const
{
    return QRectF();
}

void ItemGroup::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    // Nothing to do
}

void ItemGroup::addItem(QGraphicsItem *item)
{
    item->setParentItem(this);
}
