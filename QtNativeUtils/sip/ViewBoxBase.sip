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
                const bool invertX=false, const bool invertY=false) /HoldGIL/;
    virtual ~ViewBoxBase() /HoldGIL/;

    virtual int type() const /HoldGIL/;

    virtual void updateViewRange(const bool forceX=false, const bool forceY=false) /HoldGIL/;

    bool matrixNeedsUpdate() const /HoldGIL/;
    void setMatrixNeedsUpdate(const bool on) /HoldGIL/;

    bool autoRangeNeedsUpdate() const /HoldGIL/;
    void setAutoRangeNeedsUpdate(const bool on) /HoldGIL/;

    /*!
     * \brief By default, the positive y-axis points upward on the screen. Use invertY(True) to reverse the y-axis.
     * \param b
     */
    void invertY(const bool b=true) /HoldGIL/;
    bool yInverted() const /HoldGIL/;

    /*!
     * \brief By default, the positive x-axis points rightward on the screen. Use invertX(True) to reverse the x-axis.
     * \param b
     */
    void invertX(const bool b=true) /HoldGIL/;
    bool xInverted() const /HoldGIL/;

    void setBackgroundColor(const QColor& color) /HoldGIL/;
    QColor backgroundColor() const /HoldGIL/;
    void updateBackground() /HoldGIL/;

    const QList<Point>& viewRange() const /HoldGIL/;
    const QList<Point>& targetRange() const /HoldGIL/;
    SIP_PYOBJECT autoRangeEnabled() const [const QList<bool>& ()];
    %MethodCode
        const QList<bool> enabled = sipCpp->autoRangeEnabled();
        sipRes = Py_BuildValue("[OO]", enabled[0] ? Py_True : Py_False, enabled[1] ? Py_True : Py_False);
    %End

    double aspectLocked() const /HoldGIL/;
    void setAspectLocked(const bool lock=true, const double ratio=1.0) /HoldGIL/;

protected:

    void setViewRange(const Point& x, const Point& y) /HoldGIL/;
    void setTargetRange(const Point& x, const Point& y) /HoldGIL/;
    void setAutoRangeEnabled(const bool enableX, const bool enableY) /HoldGIL/;

signals:
    void sigYRangeChanged(ViewBoxBase* viewBox, const Point& range);
    void sigXRangeChanged(ViewBoxBase* viewBox, const Point& range);
    void sigRangeChangedManually(const bool mouseLeft, const bool mouseRight);
    void sigRangeChanged(ViewBoxBase* viewBox, const QList<Point>& range);
    void sigStateChanged(ViewBoxBase* viewBox);
    void sigTransformChanged(ViewBoxBase* viewBox);
    void sigResized(ViewBoxBase* viewBox);
};
