/***************************************************************************
 *   25/02/2016                                                            *
 *                                                                         *
 *   Copyright (C) 2016 Henesis s.r.l.                                     *
 *                                                                         *
 *   www.henesis.eu                                                        *
 *                                                                         *
 *   Alessandro Bacchini - alessandro.bacchini@henesis.eu                  *
 *                                                                         *
 ***************************************************************************/
#ifndef ITEMDEFINES_H
#define ITEMDEFINES_H

#include <QGraphicsItem>

enum CustomItemTypes
{
    TypeGraphicsObject = QGraphicsItem::UserType + 1,
    TypeGraphicsWidget = QGraphicsItem::UserType + 2,
    TypeViewBox = QGraphicsItem::UserType + 3,
    TypePlotItem = QGraphicsItem::UserType + 4
};


#endif // ITEMDEFINES_H
