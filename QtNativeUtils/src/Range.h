#ifndef RANGE_H
#define RANGE_H

#include <limits>
#include <stdexcept>

#include <QObject>
#include <QMetaType>
#include <QPointF>
#include <QList>
#include <QVector>
#include <QPointF>


class Range
{
public:
    Range() :
        mMin(NO_LIMIT),
        mMax(NO_LIMIT)
    {}

    Range(const double minVal, const double maxVal) :
        mMin(minVal),
        mMax(maxVal)
    {}

    /*
    Range(const QList<double>& l) :
        mMin(l[0]),
        mMax(l[1])
    {}

    Range(const QVector<double>& l) :
        mMin(l[0]),
        mMax(l[1])
    {}
    */

    Range(const QPointF& other) :
        mMin(other.x()),
        mMax(other.y())
    {}

    Range(const Range& other) :
        mMin(other.mMin),
        mMax(other.mMax)
    {}

    ~Range() {}

    Range& operator=(const Range& other)
    {
        mMin = other.mMin;
        mMax = other.mMax;
        return *this;
    }

    Range& operator=(const QPointF& other)
    {
        mMin = other.x();
        mMax = other.y();
        return *this;
    }

    double min() const { return mMin; }
    double max() const { return mMax; }

    void setMin(const double minVal) { mMin = minVal; }
    void setMax(const double maxVal) { mMax = maxVal; }
    void setRange(const double minVal, const double maxVal) { mMin = minVal; mMax = maxVal; }

    bool hasMin() const { return std::isfinite(mMin); }
    bool hasMax() const { return std::isfinite(mMax); }

    bool isValid() const { return mMin <= mMax; }
    bool isNull() const { return (!isValid() || mMin==mMax); }
    bool isFinite() const { return std::isfinite(mMin) && std::isfinite(mMax); }

    double& operator[](const int index) throw(std::runtime_error)
    {
        if(index==0)
            return mMin;
        else if(index==1)
            return mMax;
        throw std::runtime_error("Index not valid for Range: "+std::to_string(index));
    }

    double operator[](const int index) const throw(std::runtime_error)
    {
        if(index==0)
            return mMin;
        else if(index==1)
            return mMax;
        throw std::runtime_error("Index not valid for Range: "+std::to_string(index));
    }

    static void registerMetatype();

public:
    static constexpr double NO_LIMIT = std::numeric_limits<double>::quiet_NaN();
    static int metatypeId;

protected:

    double mMin;
    double mMax;
};


static bool operator==(const Range& r1, const Range& r2)
{
    if(r1.min()==r2.min() && r1.max()==r2.max())
        return true;

    if(!r1.isFinite() && !r2.isFinite())
        return true;

    return false;
}

static bool operator!=(const Range& r1, const Range& r2)
{
    return !(r1==r2);
}


Q_DECLARE_METATYPE(Range)

#endif // RANGE_H
