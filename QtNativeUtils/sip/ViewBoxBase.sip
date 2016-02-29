%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip


class ViewBoxBase: public QGraphicsWidget2
{

%TypeHeaderCode
#include "ViewBoxBase.h"
%End

public:

    enum MouseMode
    {
        PanMode = 3,
        RectMode = 1
    };

    enum Axis
    {
        XAxis = 0,
        YAxis = 1,
        XYAxes = 2
    };

    ViewBoxBase(QGraphicsItem* parent=nullptr, Qt::WindowFlags wFlags=0,
                const bool invertX=false, const bool invertY=false);
    virtual ~ViewBoxBase();

    virtual int type() const;

    virtual void updateViewRange(const bool forceX=false, const bool forceY=false);

    bool matrixNeedsUpdate() const;
    void setMatrixNeedsUpdate(const bool on);

    bool autoRangeNeedsUpdate() const;
    void setAutoRangeNeedsUpdate(const bool on);

    /*!
     * \brief By default, the positive y-axis points upward on the screen. Use invertY(True) to reverse the y-axis.
     * \param b
     */
    void invertY(const bool b=true);
    bool yInverted() const;

    /*!
     * \brief By default, the positive x-axis points rightward on the screen. Use invertX(True) to reverse the x-axis.
     * \param b
     */
    void invertX(const bool b=true);
    bool xInverted() const;

    void setBackgroundColor(const QColor& color);
    QColor backgroundColor() const;
    void updateBackground();

    const QList<Point>& viewRange() const;
    const QList<Point>& targetRange() const;

    double aspectLocked() const;
    void setAspectLocked(const bool lock=true, const double ratio=1.0);

protected:

    void setViewRange(const Point& x, const Point& y);
    void setTargetRange(const Point& x, const Point& y);

signals:
    void sigYRangeChanged(ViewBoxBase* viewBox, const Point& range);
    void sigXRangeChanged(ViewBoxBase* viewBox, const Point& range);
    void sigRangeChangedManually(const bool mouseLeft, const bool mouseRight);
    void sigRangeChanged(ViewBoxBase* viewBox, const QList<Point>& range);
    void sigStateChanged(ViewBoxBase* viewBox);
    void sigTransformChanged(ViewBoxBase* viewBox);
    void sigResized(ViewBoxBase* viewBox);
};
