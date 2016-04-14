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

    ViewBoxBase(QGraphicsItem* parent=nullptr, Qt::WindowFlags wFlags=0, const QPen& border=QPen(Qt::NoPen),
                const bool invertX=false, const bool invertY=false, const bool enableMouse=true) /HoldGIL/;
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
        const QList<Range>& p = sipCpp->viewRange();
        sipRes = Py_BuildValue("[[dd],[dd]]", p[0].min(), p[0].max(), p[1].min(), p[1].max());
    %End

    SIP_PYOBJECT targetRange() const [const QVector<Point>& ()];
    %MethodCode
        const QList<Range>& p = sipCpp->targetRange();
        sipRes = Py_BuildValue("[[dd],[dd]]", p[0].min(), p[0].max(), p[1].min(), p[1].max());
    %End

    SIP_PYOBJECT autoRangeEnabled() const [const QList<bool>& ()];
    %MethodCode
        const QList<bool> enabled = sipCpp->autoRangeEnabled();
        sipRes = Py_BuildValue("[OO]", enabled[0] ? Py_True : Py_False, enabled[1] ? Py_True : Py_False);
    %End


    SIP_PYOBJECT autoPan() const [const QList<bool>& ()];
    %MethodCode
        const QList<bool> enabled = sipCpp->autoPan();
        sipRes = Py_BuildValue("[OO]", enabled[0] ? Py_True : Py_False, enabled[1] ? Py_True : Py_False);
    %End


    SIP_PYOBJECT autoVisible() const [const QList<bool>& ()];
    %MethodCode
        const QList<bool> enabled = sipCpp->autoVisible();
        sipRes = Py_BuildValue("[OO]", enabled[0] ? Py_True : Py_False, enabled[1] ? Py_True : Py_False);
    %End

    void setAutoPan(const bool x=false, const bool y=false) /HoldGIL/;
    void setAutoVisible(const bool x=false, const bool y=false) /HoldGIL/;

    double aspectLocked() const /HoldGIL/;
    void setAspectLocked(const bool lock=true, const double ratio=1.0) /HoldGIL/;

    SIP_PYOBJECT mouseEnabled() const [const QVector<bool>& ()];
    %MethodCode
        const QVector<bool> enabled = sipCpp->mouseEnabled();
        sipRes = Py_BuildValue("[OO]", enabled[0] ? Py_True : Py_False, enabled[1] ? Py_True : Py_False);
    %End
    void setMouseEnabled(const bool enabledOnX=true, const bool enabledOnY=true);

    virtual QRectF viewRect() const /HoldGIL/;
    QRectF targetRect() const /HoldGIL/;

    GraphicsObject* innerSceneItem() const /HoldGIL/;

    virtual void itemsChanged() /HoldGIL/;

    ChildGroup* getChildGroup() const;

    QTransform childTransform() const;

    const QList<QGraphicsItem*>& addedItems() const;

    void addItem(QGraphicsItem* item, const bool ignoreBounds=false);

    void removeItem(QGraphicsItem* item);

    void clear();

    double suggestPadding(const Axis ax) const;

    void enableAutoRange(const Axis axis=XYAxes, const bool enable=true);

    QPointF	mapToView(const QPointF& point) const;
    QPointF	mapToView(const QPoint& point) const;
    QPolygonF mapToView(const QRectF& rect) const;
    QPolygonF mapToView(const QPolygonF& polygon) const;
    QPainterPath mapToView(const QPainterPath& path) const;
    QPointF	mapToView(qreal x, qreal y) const;

    QPointF	mapFromView(const QPointF& point) const;
    QPointF	mapFromView(const QPoint& point) const;
    QPolygonF mapFromView(const QRectF& rect) const;
    QPolygonF mapFromView(const QPolygonF& polygon) const;
    QPainterPath mapFromView(const QPainterPath& path) const;
    QPointF	mapFromView(qreal x, qreal y) const;

    QPointF	mapSceneToView(const QPointF& point) const;
    QPointF	mapSceneToView(const QPoint& point) const;
    QPolygonF mapSceneToView(const QRectF& rect) const;
    QPolygonF mapSceneToView(const QPolygonF& polygon) const;
    QPainterPath mapSceneToView(const QPainterPath& path) const;
    QPointF	mapSceneToView(qreal x, qreal y) const;

    QPointF	mapViewToScene(const QPointF& point) const;
    QPointF	mapViewToScene(const QPoint& point) const;
    QPolygonF mapViewToScene(const QRectF& rect) const;
    QPolygonF mapViewToScene(const QPolygonF& polygon) const;
    QPainterPath mapViewToScene(const QPainterPath& path) const;
    QPointF	mapViewToScene(qreal x, qreal y) const;

    QPointF	mapFromItemToView(const QGraphicsItem* item, const QPointF& point) const;
    QPointF	mapFromItemToView(const QGraphicsItem* item, const QPoint& point) const;
    QPolygonF mapFromItemToView(const QGraphicsItem* item, const QRectF& rect) const;
    QPolygonF mapFromItemToView(const QGraphicsItem* item, const QPolygonF& polygon) const;
    QPainterPath mapFromItemToView(const QGraphicsItem* item, const QPainterPath& path) const;
    QPolygonF mapFromItemToView(const QGraphicsItem* item, qreal x, qreal y, qreal w, qreal h ) const;
    QPointF	mapFromItemToView(const QGraphicsItem* item, qreal x, qreal y) const;

    QPointF	mapFromViewToItem(const QGraphicsItem* item, const QPointF& point) const;
    QPointF	mapFromViewToItem(const QGraphicsItem* item, const QPoint& point) const;
    QPolygonF mapFromViewToItem(const QGraphicsItem* item, const QRectF& rect) const;
    QPolygonF mapFromViewToItem(const QGraphicsItem* item, const QPolygonF& polygon) const;
    QPainterPath mapFromViewToItem(const QGraphicsItem* item, const QPainterPath& path) const;
    QPolygonF mapFromViewToItem(const QGraphicsItem* item, qreal x, qreal y, qreal w, qreal h ) const;
    QPointF	mapFromViewToItem(const QGraphicsItem* item, qreal x, qreal y) const;

    QPointF	mapViewToDevice(const QPointF& point) const;
    QPointF	mapViewToDevice(const QPoint& point) const;
    QPolygonF mapViewToDevice(const QRectF& rect) const;
    QPolygonF mapViewToDevice(const QPolygonF& polygon) const;
    QPainterPath mapViewToDevice(const QPainterPath& path) const;
    QPointF	mapViewToDevice(qreal x, qreal y) const;

    QPointF	mapDeviceToView(const QPointF& point) const;
    QPointF	mapDeviceToView(const QPoint& point) const;
    QPolygonF mapDeviceToView(const QRectF& rect) const;
    QPolygonF mapDeviceToView(const QPolygonF& polygon) const;
    QPainterPath mapDeviceToView(const QPainterPath& path) const;
    QPointF	mapDeviceToView(qreal x, qreal y) const;

    QRectF itemBoundingRect(const QGraphicsItem* item) const;

    void setRange() /NoArgParser/;
    %MethodCode
        // rect=None, xRange=None, yRange=None, padding=None, update=True, disableAutoRange=True
        // PyObject *sipSelf, PyObject *sipArgs, PyObject *sipKwds
        char* keywords[] = {"rect", "xRange", "yRange", "padding", "update", "disableAutoRange", nullptr};
        PyObject* pyRect = Py_None;
        PyObject* pyXRange = Py_None;
        PyObject* pyYRange = Py_None;
        PyObject* pyPadding = Py_None;
        PyObject* pyUpdate = Py_True;
        PyObject* pyDisableAutoRange = Py_True;

        bool sipSelfWasArg = (!sipSelf || sipIsDerived((sipSimpleWrapper *)sipSelf));

        if(!sipSelfWasArg)
        {
            if(!PyArg_ParseTupleAndKeywords(sipArgs, sipKwds, "O|OOOOOO", keywords, &sipSelf, &pyRect, &pyXRange, &pyYRange, &pyPadding, &pyUpdate, &pyDisableAutoRange))
            {
                return NULL;
            }
        }
        else
        {
            if(!PyArg_ParseTupleAndKeywords(sipArgs, sipKwds, "|OOOOOO", keywords, &pyRect, &pyXRange, &pyYRange, &pyPadding, &pyUpdate, &pyDisableAutoRange))
            {
                return NULL;
            }
        }

        if(!sipCanConvertToType(sipSelf, sipType_ViewBoxBase, SIP_NOT_NONE))
        {
            return NULL;
        }

        int vbstate = 0, sipIsErr = 0;
        ViewBoxBase* viewBox = reinterpret_cast<ViewBoxBase*>(sipConvertToType(sipSelf, sipType_ViewBoxBase, 0, 0, &vbstate, &sipIsErr));

        double padding = ViewBoxBase::AutoPadding;
        if(pyPadding!=Py_None)
            padding = PyFloat_AS_DOUBLE(pyPadding);
        bool disableAutoRange = PyObject_IsTrue(pyDisableAutoRange);

        // try to call with rect
        if(sipCanConvertToType(pyRect, sipType_QRectF, SIP_NOT_NONE))
        {
            int state = 0;
            QRectF* rect = reinterpret_cast<QRectF*>(sipConvertToType(pyRect, sipType_QRectF, 0, SIP_NOT_NONE, &state, &sipIsErr));

            viewBox->setRange(*rect, padding, disableAutoRange);

            sipReleaseType(rect, sipType_QRectF, state);
            sipReleaseType(viewBox, sipType_ViewBoxBase, vbstate);
            Py_RETURN_NONE;
        }

        Range xRange;
        Range yRange;

        if(sipCanConvertToType(pyXRange, sipType_Range, SIP_NOT_NONE))
        {
            int xstate = 0;
            Range* r = reinterpret_cast<Range*>(sipConvertToType(pyXRange, sipType_Range, 0, SIP_NOT_NONE, &xstate, &sipIsErr));
            xRange.setRange(r->min(), r->max());
            sipReleaseType(r, sipType_Range, xstate);
        }

        if(sipCanConvertToType(pyYRange, sipType_Range, SIP_NOT_NONE))
        {
            int ystate = 0;
            Range* r = reinterpret_cast<Range*>(sipConvertToType(pyYRange, sipType_Range, 0, SIP_NOT_NONE, &ystate, &sipIsErr));
            yRange.setRange(r->min(), r->max());
            sipReleaseType(r, sipType_Range, ystate);
        }

        viewBox->setRange(xRange, yRange, padding, disableAutoRange);

        sipReleaseType(viewBox, sipType_ViewBoxBase, vbstate);

        Py_RETURN_NONE;

    %End

    void setXRange(const double minR, const double maxR, const double padding=ViewBoxBase::AutoPadding);
    void setYRange(const double minR, const double maxR, const double padding=ViewBoxBase::AutoPadding);

    void scaleBy()  /NoArgParser/;
    %MethodCode
        // s=None, center=None, x=None, y=None
        // PyObject *sipSelf, PyObject *sipArgs, PyObject *sipKwds

        char* keywords[] = {"s", "center", "x", "y", nullptr};
        PyObject* pyS = Py_None;
        PyObject* pyCenter = Py_None;
        PyObject* pyX = Py_None;
        PyObject* pyY = Py_None;

        bool sipSelfWasArg = (!sipSelf || sipIsDerived((sipSimpleWrapper *)sipSelf));

        if(!sipSelfWasArg)
        {
            if(!PyArg_ParseTupleAndKeywords(sipArgs, sipKwds, "O|OOOO", keywords, &sipSelf, &pyS, &pyCenter, &pyX, &pyY))
            {
                return NULL;
            }
        }
        else
        {
            if(!PyArg_ParseTupleAndKeywords(sipArgs, sipKwds, "|OOOO", keywords, &pyS, &pyCenter, &pyX, &pyY))
            {
                return NULL;
            }
        }

        if(!sipCanConvertToType(sipSelf, sipType_ViewBoxBase, SIP_NOT_NONE))
        {
            return NULL;
        }

        int vbstate = 0, sipIsErr = 0;
        ViewBoxBase* viewBox = reinterpret_cast<ViewBoxBase*>(sipConvertToType(sipSelf, sipType_ViewBoxBase, 0, 0, &vbstate, &sipIsErr));

        QPointF center;
        QPointF scale(1.0, 1.0);

        if(sipCanConvertToType(pyCenter, sipType_Point, SIP_NOT_NONE))
        {
            int state;
            Point* c = reinterpret_cast<Point*>(sipConvertToType(pyCenter, sipType_Point, 0, SIP_NOT_NONE, &state, &sipIsErr));
            center.setX(c->x());
            center.setY(c->y());
            sipReleaseType(c, sipType_Point, state);
        }
        else
            center = viewBox->targetRect().center();

        if(sipCanConvertToType(pyS, sipType_Point, SIP_NOT_NONE))
        {
            int state = 0;
            Point* s = reinterpret_cast<Point*>(sipConvertToType(pyS, sipType_Point, 0, SIP_NOT_NONE, &state, &sipIsErr));
            scale.setX(s->x());
            scale.setY(s->y());
            sipReleaseType(s, sipType_Point, state);
        }
        else
        {
            if(PyInt_Check(pyX) || PyFloat_Check(pyX))
                scale.setX(pyAsDouble(pyX));
            if(PyInt_Check(pyY) || PyFloat_Check(pyY))
                scale.setY(pyAsDouble(pyY));
        }

        viewBox->scaleBy(scale, center);

        sipReleaseType(viewBox, sipType_ViewBoxBase, vbstate);

        Py_RETURN_NONE;
    %End

    void translateBy()  /NoArgParser/;
    %MethodCode
        // t=None, x=None, y=None, y=None
        // PyObject *sipSelf, PyObject *sipArgs, PyObject *sipKwds

        char* keywords[] = {"t", "x", "y", nullptr};
        PyObject* pyT = Py_None;
        PyObject* pyX = Py_None;
        PyObject* pyY = Py_None;

        bool sipSelfWasArg = (!sipSelf || sipIsDerived((sipSimpleWrapper *)sipSelf));

        if(!sipSelfWasArg)
        {
            if(!PyArg_ParseTupleAndKeywords(sipArgs, sipKwds, "O|OOO", keywords, &sipSelf, &pyT, &pyX, &pyY))
            {
                return NULL;
            }
        }
        else
        {
            if(!PyArg_ParseTupleAndKeywords(sipArgs, sipKwds, "|OOO", keywords, &pyT, &pyX, &pyY))
            {
                return NULL;
            }
        }

        if(!sipCanConvertToType(sipSelf, sipType_ViewBoxBase, SIP_NOT_NONE))
        {
            return NULL;
        }

        int vbstate = 0, sipIsErr = 0;
        ViewBoxBase* viewBox = reinterpret_cast<ViewBoxBase*>(sipConvertToType(sipSelf, sipType_ViewBoxBase, 0, 0, &vbstate, &sipIsErr));

        QPointF dt(0.0, 0.0);
        if(sipCanConvertToType(pyT, sipType_Point, SIP_NOT_NONE))
        {
            int state = 0;
            Point* t = reinterpret_cast<Point*>(sipConvertToType(pyT, sipType_Point, 0, SIP_NOT_NONE, &state, &sipIsErr));
            dt.setX(t->x());
            dt.setY(t->y());
            sipReleaseType(t, sipType_Point, state);
        }
        else
        {
            if(PyInt_Check(pyX) || PyFloat_Check(pyX))
                dt.setX(pyAsDouble(pyX));
            if(PyInt_Check(pyY) || PyFloat_Check(pyY))
                dt.setY(pyAsDouble(pyY));
        }

        viewBox->translateBy(dt);

        sipReleaseType(viewBox, sipType_ViewBoxBase, vbstate);

        Py_RETURN_NONE;
    %End

    void setXLimits(const Range& rng);
    void setYLimits(const Range& rng);
    void setXRangeLimits(const Range& rng);
    void setYRangeLimits(const Range& rng);

    SIP_PYOBJECT xLimits() const [Range ()];
    %MethodCode
        Range rng = sipCpp->xLimits();
        sipRes = Py_BuildValue("[dd]", rng.min(), rng.max());
    %End

    SIP_PYOBJECT yLimits() const [Range ()];
    %MethodCode
        Range rng = sipCpp->yLimits();
        sipRes = Py_BuildValue("[dd]", rng.min(), rng.max());
    %End

    SIP_PYOBJECT xRangeLimits() const [Range ()];
    %MethodCode
        Range rng = sipCpp->xRangeLimits();
        sipRes = Py_BuildValue("[dd]", rng.min(), rng.max());
    %End

    SIP_PYOBJECT yRangeLimits() const [Range ()];
    %MethodCode
        Range rng = sipCpp->yRangeLimits();
        sipRes = Py_BuildValue("[dd]", rng.min(), rng.max());
    %End

    void setLimits() /NoArgParser/;
    %MethodCode
        // "xMin", "xMax", "yMin", "yMax", "minXRange", "maxXRange", "minYRange", "maxYRange"
        // PyObject *sipSelf, PyObject *sipArgs, PyObject *sipKwds

        bool sipSelfWasArg = (!sipSelf || sipIsDerived((sipSimpleWrapper *)sipSelf));

        if(!sipSelfWasArg)
        {
            if(!PyTuple_Size(sipArgs)==1)
                return NULL;

            sipSelf = PyTuple_GetItem(sipArgs, 0);
        }

        if(!sipCanConvertToType(sipSelf, sipType_ViewBoxBase, SIP_NOT_NONE))
        {
            return NULL;
        }

        int vbstate = 0, sipIsErr = 0;
        ViewBoxBase* viewBox = reinterpret_cast<ViewBoxBase*>(sipConvertToType(sipSelf, sipType_ViewBoxBase, 0, 0, &vbstate, &sipIsErr));

        Range rng = viewBox->xLimits();
        PyObject* val = PyDict_GetItemString(sipKwds, "xMin");
        if(val!=nullptr)
            rng.setMin((val==Py_None) ? Range::NO_LIMIT : pyAsDouble(val));
        val = PyDict_GetItemString(sipKwds, "xMax");
        if(val!=nullptr)
            rng.setMax((val==Py_None) ? Range::NO_LIMIT : pyAsDouble(val));
        viewBox->setXLimits(rng);

        rng = viewBox->yLimits();
        val = PyDict_GetItemString(sipKwds, "yMin");
        if(val!=nullptr)
            rng.setMin((val==Py_None) ? Range::NO_LIMIT : pyAsDouble(val));
        val = PyDict_GetItemString(sipKwds, "yMax");
        if(val!=nullptr)
            rng.setMax((val==Py_None) ? Range::NO_LIMIT : pyAsDouble(val));
        viewBox->setYLimits(rng);

        rng = viewBox->xRangeLimits();
        val = PyDict_GetItemString(sipKwds, "minXRange");
        if(val!=nullptr)
            rng.setMin((val==Py_None) ? Range::NO_LIMIT : pyAsDouble(val));
        val = PyDict_GetItemString(sipKwds, "maxXRange");
        if(val!=nullptr)
            rng.setMax((val==Py_None) ? Range::NO_LIMIT : pyAsDouble(val));
        viewBox->setXRangeLimits(rng);

        rng = viewBox->yRangeLimits();
        val = PyDict_GetItemString(sipKwds, "minYRange");
        if(val!=nullptr)
            rng.setMin((val==Py_None) ? Range::NO_LIMIT : pyAsDouble(val));
        val = PyDict_GetItemString(sipKwds, "maxYRange");
        if(val!=nullptr)
            rng.setMax((val==Py_None) ? Range::NO_LIMIT : pyAsDouble(val));
        viewBox->setYRangeLimits(rng);

        viewBox->updateViewRange();

        sipReleaseType(viewBox, sipType_ViewBoxBase, vbstate);

        Py_RETURN_NONE;
    %End

    ViewBoxBase::MouseMode mouseMode() const;
    void setMouseMode(const ViewBoxBase::MouseMode mode);

    void setWheelScaleFactor(const double factor);
    double wheelScaleFactor() const;

    void linkedWheelEvent(QGraphicsSceneWheelEvent* event, const Axis axis=XYAxes);

    //virtual void mouseDragEvent(MouseDragEvent* event);
    virtual void mouseDragEvent(MouseDragEvent* event, const Axis axis=XYAxes);

    QList<QGraphicsItem*> allChildren(QGraphicsItem* item=nullptr) const;

public slots:

    void prepareForPaint();

    void showAxRect(const QRectF& axRect);

    void scaleHistory(const int d);

protected:

    void setViewRange(const Range& x, const Range& y) /HoldGIL/;
    void setTargetRange(const Range& x, const Range& y) /HoldGIL/;
    void setAutoRangeEnabled(const bool enableX, const bool enableY) /HoldGIL/;

    void _resetTarget() /HoldGIL/;

    void setInnerSceneItem(GraphicsObject* innerItem) /HoldGIL/;

    virtual QVariant itemChange(GraphicsItemChange change, const QVariant& value);

    virtual void wheelEvent(QGraphicsSceneWheelEvent* event);
    virtual void keyPressEvent(QKeyEvent *event);

    void updateScaleBox(const QPointF& p1, const QPointF& p2);
    void hideScaleBox();

    void addToHistory(const QRectF& r);

signals:

    void sigYRangeChanged(const Range& range);
    void sigXRangeChanged(const Range& range);
    void sigRangeChangedManually(const bool mouseLeft, const bool mouseRight);
    void sigRangeChanged(const Range& xRange, const Range& yRange);
    void sigStateChanged(ViewBoxBase* viewBox);
    void sigTransformChanged();
    void sigResized();
};
