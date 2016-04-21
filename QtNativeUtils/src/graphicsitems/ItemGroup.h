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
#ifndef ITEMGROUP_H
#define ITEMGROUP_H

#include "GraphicsObject.h"

/*!
 * \brief The ItemGroup class
 *
 * Replacement for QGraphicsItemGroup
 */
class ItemGroup: public GraphicsObject
{
    Q_OBJECT
public:
    explicit ItemGroup(QGraphicsItem* parent=nullptr);
    virtual ~ItemGroup();

    virtual QRectF boundingRect() const;

    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget=nullptr);

    void addItem(QGraphicsItem* item);
};

#endif // ITEMGROUP_H
