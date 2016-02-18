#ifndef POINT_H
#define POINT_H


#include <stdexcept>
#include <iostream>
#include <cmath>
#include <string>
#include <strstream>

#include <QPointF>
#include <QSizeF>

class Point: public QPointF
{
public:
    Point():
        QPointF()
    {}

    Point(const double xy):
        QPointF(xy, xy)
    {}

    Point(const double x, const double y):
        QPointF(x, y)
    {}

    Point(const QPointF& p):
        QPointF(p.x(), p.y())
    {}

    Point(const QPoint& p):
        QPointF(p.x(), p.y())
    {}

    Point(const QSizeF& s):
        QPointF(s.width(), s.height())
    {}

    double& operator[](const int index) throw(std::runtime_error)
    {
        if(index==0)
            return rx();
        else if(index==1)
            return ry();
        throw std::runtime_error("Index not valid for Point: "+std::to_string(index));
    }

    double operator[](const int index) const throw(std::runtime_error)
    {
        if(index==0)
            return x();
        else if(index==1)
            return y();
        throw std::runtime_error("Index not valid for Point: "+std::to_string(index));
    }

    double length() const
    {
        return std::sqrt(x()*x() + y()*y());
    }

    Point norm() const
    {
        double l = length();
        return Point(x()/l, y()/l);
    }

    double angle(const Point& a) const
    {
        double n1 = length();
        double n2 = a.length();
        if(n1==0.0 || n2==0.0)
            return 0.0;

        double ang = std::atan2(y(),x()) - std::atan2(a.y(), a.x());
        return ang * 180.0 / M_PI;
    }

    double dot(const Point& a) const
    {
        return (x()*a.x() + y()*a.y());
    }

    double cross(const Point& a) const
    {
        return (x()*a.y() - y()*a.x());
    }

    Point proj(const Point& b) const
    {
        Point b1(b.norm());
        double d = dot(b1);
        b1.rx() *= d;
        b1.ry() *= d;
        return b1;
    }

    double min() const
    {
        return std::min(x(), y());
    }

    double max() const
    {
        return std::max(x(), y());
    }

    Point copy() const
    {
        return Point(x(), y());
    }

    Point toQPoint() const
    {
        return QPoint(x(), y());
    }


};



#endif // POINT_H
