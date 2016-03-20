#ifndef RANGE_H
#define RANGE_H

#include <limits>

#include <QObject>
#include <QMetaType>
#include <QPointF>
#include <QList>
#include <QVector>


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

    Range(const QList<double>& l) :
        mMin(l[0]),
        mMax(l[1])
    {}

    Range(const QVector<double>& l) :
        mMin(l[0]),
        mMax(l[1])
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

    bool hasMin() const { return std::isfinite(mMin); }
    bool hasMax() const { return std::isfinite(mMax); }

    bool isValid() const { return mMin <= mMax; }
    bool isNull() const { return (!isValid() || mMin==mMax); }

    bool operator==(const Range& other)
    {
        if(other.mMin==mMin && other.mMax==mMax)
            return true;

        if(!std::isfinite(mMin) && !std::isfinite(mMax)
           && !std::isfinite(other.mMin) && !std::isfinite(other.mMax))
            return true;

        return false;
    }

    bool operator!=(const Range& other)
    {
        return !(*this==other);
    }


public:
    static constexpr double NO_LIMIT = std::numeric_limits<double>::quiet_NaN();

protected:

    double mMin;
    double mMax;
};


Q_DECLARE_METATYPE(Range);

#endif // RANGE_H
