%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip


class ViewBoxBase: public GraphicsWidget, public ItemChangedListener
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
    virtual void updateAutoRange();
    virtual void updateMatrix();

    virtual void itemBoundsChanged(QGraphicsItem* item) /HoldGIL/;

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

    SIP_PYOBJECT viewRange() const [const QVector<Point>& ()];
    %MethodCode
        const QVector<Point>& p = sipCpp->viewRange();
        sipRes = Py_BuildValue("[[dd],[dd]]", p[0].x(), p[0].y(), p[1].x(), p[1].y());
    %End

    SIP_PYOBJECT targetRange() const [const QVector<Point>& ()];
    %MethodCode
        const QVector<Point>& p = sipCpp->targetRange();
        sipRes = Py_BuildValue("[[dd],[dd]]", p[0].x(), p[0].y(), p[1].x(), p[1].y());
    %End

    SIP_PYOBJECT autoRangeEnabled() const [const QList<bool>& ()];
    %MethodCode
        const QVector<bool> enabled = sipCpp->autoRangeEnabled();
        sipRes = Py_BuildValue("[OO]", enabled[0] ? Py_True : Py_False, enabled[1] ? Py_True : Py_False);
    %End

    void setAutoPan(const bool x=false, const bool y=false) /HoldGIL/;
    SIP_PYOBJECT autoPan() const [const QVector<bool>& ()];
    %MethodCode
        const QVector<bool> enabled = sipCpp->autoPan();
        sipRes = Py_BuildValue("[OO]", enabled[0] ? Py_True : Py_False, enabled[1] ? Py_True : Py_False);
    %End

    void setAutoVisible(const bool x=false, const bool y=false) /HoldGIL/;
    SIP_PYOBJECT autoVisible() const [const QVector<bool>& ()];
    %MethodCode
        const QVector<bool> enabled = sipCpp->autoVisible();
        sipRes = Py_BuildValue("[OO]", enabled[0] ? Py_True : Py_False, enabled[1] ? Py_True : Py_False);
    %End

    double aspectLocked() const /HoldGIL/;
    void setAspectLocked(const bool lock=true, const double ratio=1.0) /HoldGIL/;

    virtual QRectF viewRect() const /HoldGIL/;
    QRectF targetRect() const /HoldGIL/;

    GraphicsObject* innerSceneItem() const /HoldGIL/;

    virtual void itemsChanged() /HoldGIL/;

    ChildGroup* getChildGroup() const;

    QTransform childTransform();

public slots:

    void prepareForPaint();

protected:

    void setViewRange(const Point& x, const Point& y) /HoldGIL/;
    void setTargetRange(const Point& x, const Point& y) /HoldGIL/;
    void setAutoRangeEnabled(const bool enableX, const bool enableY) /HoldGIL/;

    void _resetTarget() /HoldGIL/;

    void setInnerSceneItem(GraphicsObject* innerItem) /HoldGIL/;

    virtual QVariant itemChange(GraphicsItemChange change, const QVariant& value);

signals:

    void sigYRangeChanged(const Point& range);
    void sigXRangeChanged(const Point& range);
    void sigRangeChangedManually(const bool mouseLeft, const bool mouseRight);
    void sigRangeChanged(const QList<Point>& range);
    void sigStateChanged(ViewBoxBase* viewBox);
    void sigTransformChanged();
    void sigResized();
};
