#ifndef POINT_UTILS_H
#define POINT_UTILS_H

#include <QPointF>
#include <cmath>

static double length(const QPoint& p)
{
    return std::sqrt(p.x()*p.x() + p.y()*p.y());
}

static double length(const QPointF& p)
{
    return std::sqrt(p.x()*p.x() + p.y()*p.y());
}

static double distance(const QPoint& p0, const QPoint& p1)
{
    return length(p0 - p1);
}

static double distance(const QPointF& p0, const QPointF& p1)
{
    return length(p0 - p1);
}

static double distance(const QPointF& p0, const QPoint& p1)
{
    return length(p0 - QPointF(p1));
}

static double distance(const QPoint& p0, const QPointF& p1)
{
    return length(QPointF(p0) - p1);
}



#endif // POINT_UTILS_H
