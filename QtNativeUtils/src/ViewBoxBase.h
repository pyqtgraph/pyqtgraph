#ifndef VIEWBOXBASE_H
#define VIEWBOXBASE_H

#include <QList>
#include <QGraphicsRectItem>
#include <QWidget>
#include <QWeakPointer>

#include "GraphicsWidget.h"
#include "Point.h"
#include "ItemDefines.h"
#include "GraphicsObject.h"
#include "Interfaces.h"
#include "graphicsitems/ChildGroup.h"
#include "Range.h"



class Limits
{
public:
    Limits()
    {}

    Range xLimits() const { return mXLimits; }
    Range yLimits() const { return mYLimits; }
    Range xRange() const { return mXRange; }
    Range yRange() const { return mYRange; }

    double xMin() const { return mXLimits.min(); }
    double xMax() const { return mXLimits.max(); }
    double yMin() const { return mYLimits.min(); }
    double yMax() const { return mYLimits.max(); }
    double minXRange() const { return mXRange.min(); }
    double maxXRange() const { return mXRange.max(); }
    double minYRange() const { return mYRange.min(); }
    double maxYRange() const { return mYRange.max(); }

    void setXLimits(const double xMin, const double xMax) { mXLimits.setRange(xMin, xMax); }
    void setYLimits(const double yMin, const double yMax) { mYLimits.setRange(yMin, yMax); }

    void setXRange(const double xMin, const double xMax) { mXRange.setRange(xMin, xMax); }
    void setYRange(const double yMin, const double yMax) { mYRange.setRange(yMin, yMax); }

    void setXMin(const double val) { mXLimits.setMin(val); }
    void setXMax(const double val) { mXLimits.setMax(val); }
    void setYMin(const double val) { mYLimits.setMin(val); }
    void setYMax(const double val) { mYLimits.setMax(val); }
    void setMinXRange(const double val) { mXRange.setMin(val); }
    void setMaxXRange(const double val) { mXRange.setMax(val); }
    void setMinYRange(const double val) { mYRange.setMin(val); }
    void setMaxYRange(const double val) { mYRange.setMax(val); }

protected:

    Range mXLimits; // Maximum and minimum visible X values
    Range mYLimits; // Maximum and minimum visible Y values
    Range mXRange;  // Maximum and minimum X range
    Range mYRange;  // Maximum and minimum Y range

};



/*!
 * \brief The ViewBoxBase class
 *
 * Box that allows internal scaling/panning of children by mouse drag.
 * This class is usually created automatically as part of a
 * :class:`PlotItem <pyqtgraph.PlotItem>` or :class:`Canvas <pyqtgraph.canvas.Canvas>`
 * or with :func:`GraphicsLayout.addViewBox() <pyqtgraph.GraphicsLayout.addViewBox>`.
 *
 * Features:
 *
 * - Scaling contents by mouse or auto-scale when contents change
 * - View linking--multiple views display the same data ranges
 * - Configurable by context menu
 * - Item coordinate mapping methods
 */
class ViewBoxBase: public GraphicsWidget, public ItemChangedListener
{
    Q_OBJECT
public:

    enum MouseMode
    {
        PanMode = 3,
        RectMode = 1
    };

    /*!
     * \brief ViewBoxBase
     *
     * \param parent Optional parent widget
     * \param wFlags
     * \param border Draw a border around the view
     * \param lockAspect The aspect ratio to lock the view coorinates to or 0.0 to allow the ratio to change
     * \param invertX Invert x axis
     * \param invertY Invert y axis
     * \param enableMouse Whether mouse can be used to scale/pan the view
     */
    ViewBoxBase(QGraphicsItem* parent=nullptr, Qt::WindowFlags wFlags=0, const QPen& border=QPen(Qt::NoPen),
                const double lockAspect=0.0,const bool invertX=false, const bool invertY=false,
                const bool enableMouse=true);
    virtual ~ViewBoxBase() {}

    enum { Type = CustomItemTypes::TypeViewBox };

    virtual int type() const
    {
        // Enable the use of qgraphicsitem_cast with this item.
        return Type;
    }

    /*!
     * \brief Update viewRange to match targetRange as closely as possible, given aspect ratio constraints.
     *
     * The force arguments are used to indicate which axis (if any) should be unchanged when
     * applying constraints.
     * \param forceX true for left unchanging x axis when applying contraints
     * \param forceY true for left unchanging y axis when applying contraints
     */
    void updateViewRange(const bool forceX=false, const bool forceY=false);

    /*!
     * \brief Update the internal transformation matrix
     * Make the childGroup's transform match the requested viewRange
     */
    void updateMatrix();

    virtual void itemBoundsChanged(QGraphicsItem* item);


    bool matrixNeedsUpdate() const { return mMatrixNeedsUpdate; }
    void setMatrixNeedsUpdate(const bool on) { mMatrixNeedsUpdate = on; }

    bool autoRangeNeedsUpdate() const { return mAutoRangeNeedsUpdate; }
    void setAutoRangeNeedsUpdate(const bool on) { mAutoRangeNeedsUpdate = on; }

    void invertY(const bool b=true);
    bool yInverted() const { return mYInverted; }

    void invertX(const bool b=true);
    bool xInverted() const { return mXInverted; }

    void setBackgroundColor(const QColor& color);
    QColor backgroundColor() const;
    void updateBackground();

    const QList<Range>& viewRange() const { return mViewRange; }
    const QList<Range>& targetRange() const { return mTargetRange; }
    const QList<bool>& autoRangeEnabled() const { return mAutoRangeEnabled; }

    void setAutoPan(const bool x=false, const bool y=false);
    const QList<bool>& autoPan() const { return mAutoPan; }

    void setAutoVisible(const bool x=false, const bool y=false);
    const QList<bool>& autoVisible() const { return mAutoVisibleOnly; }

    double aspectLocked() const { return mAspectLocked; }

    /*!
     * \brief Lock teh aspect ratio.
     * If the aspect ratio is locked, view scaling must always preserve the aspect ratio.
     * By default, the ratio is set to 1; x and y both have the same scaling.
     * This ratio can be overridden (xScale/yScale), or use 0.0 to lock in the current ratio.
     * \param lock true for locking the aspect
     * \param ratio New aspect ratio. 0.0 for setting the current aspect ratio.
     */
    void setAspectLocked(const bool lock=true, const double ratio=1.0);

    const QVector<bool>& mouseEnabled() const { return mMouseEnabled; }

    /*!
     * \brief Set whether each axis is enabled for mouse interaction
     *
     * This allows the user to pan/scale one axis of the view while leaving the other axis unchanged.
     *
     * \param enabledOnX true to eneble mouse interacion on x axis
     * \param enabledOnY true to eneble mouse interacion on y axis
     */
    void setMouseEnabled(const bool enabledOnX=true, const bool enabledOnY=true);

    /*!
     * \brief Bounding of the region visible within the ViewBox
     * \return The bounding of the region visible within the ViewBox
     */
    virtual QRectF viewRect() const;

    /*!
     * \brief Region which has been requested to be visible.
     *
     * This is not necessarily the same as the region that is *actually* visible.
     * Resizing and aspect ratio constraints can cause targetRect() and viewRect() to differ.
     *
     * \return The regiorn that hsa to be visible
     */
    QRectF targetRect() const;

    GraphicsObject* innerSceneItem() const;

    /*!
     * \brief Called when items are added/removed from childGroup
     */
    virtual void itemsChanged();

    ChildGroup* getChildGroup() const;

    /*!
     * \brief Transform that maps from child(item in the childGroup) coordinates to local coordinates.
     * This maps from inside the viewbox to outside
     * \return Transform
     */
    QTransform childTransform() const;

    QList<QGraphicsItem*> addedItems() const;

    /*!
     * \brief Add a QGraphicsItem to this view.
     *
     *  The view will include this item when determining how to set its range
     *  automatically unless ignoreBounds is True.
     *
     * \param item Item to add to teh view
     * \param ignoreBounds true to ignore the bounds of the item during autoscaling
     */
    void addItem(QGraphicsItem* item, const bool ignoreBounds=false);

    /*!
     * \brief Remove an item from this view.
     * \param item Item to remove
     */
    void removeItem(QGraphicsItem* item);

    void clear();

    double suggestPadding(const Axis ax) const;

    /*!
     * \brief Enable (or disable) auto-range for axis.
     *
     * When enabled, the axis will automatically rescale when items are added/removed or change their shape.
     *
     * \param axis Axis, which may be ViewBox.XAxis, ViewBox.YAxis, or ViewBox.XYAxes for both.
     * \param enable Enabling state
     */
    void enableAutoRange(const Axis axis=XYAxes, const bool enable=true);

    /*!
     * \brief Enable (or disable) auto-range for axis.
     *
     * When enabled, the axis will automatically rescale when items are added/removed or change their shape.
     *
     * \param QString Axis, which may be "x", "y", or "xy" for both.
     * \param enable Enabling state
     */
    void enableAutoRange(const QString& axis="xy", const bool enable=true);

    /*!
     * \brief Maps from the local coordinates of the ViewBox to the coordinate system displayed inside the ViewBox
     * \param point
     * \return
     */
    QPointF	mapToView(const QPointF& point) const { return childTransform().inverted().map(point); }
    /*!
     * \brief Maps from the local coordinates of the ViewBox to the coordinate system displayed inside the ViewBox
     * \param point
     * \return
     */
    QPointF	mapToView(const QPoint& point) const { return childTransform().inverted().map(QPointF(point)); }
    /*!
     * \brief Maps from the local coordinates of the ViewBox to the coordinate system displayed inside the ViewBox
     * \param rect
     * \return
     */
    QPolygonF mapToView(const QRectF& rect) const { return childTransform().inverted().map(rect); }
    /*!
     * \brief Maps from the local coordinates of the ViewBox to the coordinate system displayed inside the ViewBox
     * \param polygon
     * \return
     */
    QPolygonF mapToView(const QPolygonF& polygon) const { return childTransform().inverted().map(polygon); }
    /*!
     * \brief Maps from the local coordinates of the ViewBox to the coordinate system displayed inside the ViewBox
     * \param path
     * \return
     */
    QPainterPath mapToView(const QPainterPath& path) const { return childTransform().inverted().map(path); }
    /*!
     * \brief Maps from the local coordinates of the ViewBox to the coordinate system displayed inside the ViewBox
     * \param x
     * \param y
     * \return
     */
    QPointF	mapToView(qreal x, qreal y) const { return mapToView(QPointF(x, y)); }

    /*!
     * \brief Maps from the coordinate system displayed inside the ViewBox to the local coordinates of the ViewBox
     * \param point
     * \return
     */
    QPointF	mapFromView(const QPointF& point) const { return childTransform().map(point); }
    /*!
     * \brief Maps from the coordinate system displayed inside the ViewBox to the local coordinates of the ViewBox
     * \param point
     * \return
     */
    QPointF	mapFromView(const QPoint& point) const { return childTransform().map(QPointF(point)); }
    /*!
     * \brief Maps from the coordinate system displayed inside the ViewBox to the local coordinates of the ViewBox
     * \param rect
     * \return
     */
    QPolygonF mapFromView(const QRectF& rect) const { return childTransform().map(rect); }
    /*!
     * \brief Maps from the coordinate system displayed inside the ViewBox to the local coordinates of the ViewBox
     * \param polygon
     * \return
     */
    QPolygonF mapFromView(const QPolygonF& polygon) const { return childTransform().map(polygon); }
    /*!
     * \brief Maps from the coordinate system displayed inside the ViewBox to the local coordinates of the ViewBox
     * \param path
     * \return
     */
    QPainterPath mapFromView(const QPainterPath& path) const { return childTransform().map(path); }
    /*!
     * \brief Maps from the coordinate system displayed inside the ViewBox to the local coordinates of the ViewBox
     * \param x
     * \param y
     * \return
     */
    QPointF	mapFromView(qreal x, qreal y) const { return mapFromView(QPointF(x, y)); }

    /*!
     * \brief Maps from scene coordinates to the coordinate system displayed inside the ViewBox
     * \param point
     * \return
     */
    QPointF	mapSceneToView(const QPointF& point) const { return mapToView(mapFromScene(point)); }
    /*!
     * \brief Maps from scene coordinates to the coordinate system displayed inside the ViewBox
     * \param point
     * \return
     */
    QPointF	mapSceneToView(const QPoint& point) const { return mapToView(mapFromScene(QPointF(point))); }
    /*!
     * \brief Maps from scene coordinates to the coordinate system displayed inside the ViewBox
     * \param rect
     * \return
     */
    QPolygonF mapSceneToView(const QRectF& rect) const { return mapToView(mapFromScene(rect)); }
    /*!
     * \brief Maps from scene coordinates to the coordinate system displayed inside the ViewBox
     * \param polygon
     * \return
     */
    QPolygonF mapSceneToView(const QPolygonF& polygon) const { return mapToView(mapFromScene(polygon)); }
    /*!
     * \brief Maps from scene coordinates to the coordinate system displayed inside the ViewBox
     * \param path
     * \return
     */
    QPainterPath mapSceneToView(const QPainterPath& path) const { return mapToView(mapFromScene(path)); }
    /*!
     * \brief Maps from scene coordinates to the coordinate system displayed inside the ViewBox
     * \param x
     * \param y
     * \return
     */
    QPointF	mapSceneToView(qreal x, qreal y) const { return mapSceneToView(QPointF(x, y)); }

    /*!
     * \brief Maps from the coordinate system displayed inside the ViewBox to scene coordinates
     * \param point
     * \return
     */
    QPointF	mapViewToScene(const QPointF& point) const { return mapToScene(mapFromView(point)); }
    /*!
     * \brief Maps from the coordinate system displayed inside the ViewBox to scene coordinates
     * \param point
     * \return
     */
    QPointF	mapViewToScene(const QPoint& point) const { return mapToScene(mapFromView(QPointF(point))); }
    /*!
     * \brief Maps from the coordinate system displayed inside the ViewBox to scene coordinates
     * \param rect
     * \return
     */
    QPolygonF mapViewToScene(const QRectF& rect) const { return mapToScene(mapFromView(rect)); }
    /*!
     * \brief Maps from the coordinate system displayed inside the ViewBox to scene coordinates
     * \param polygon
     * \return
     */
    QPolygonF mapViewToScene(const QPolygonF& polygon) const { return mapToScene(mapFromView(polygon)); }
    /*!
     * \brief Maps from the coordinate system displayed inside the ViewBox to scene coordinates
     * \param path
     * \return
     */
    QPainterPath mapViewToScene(const QPainterPath& path) const { return mapToScene(mapFromView(path)); }
    /*!
     * \brief Maps from the coordinate system displayed inside the ViewBox to scene coordinates
     * \param x
     * \param y
     * \return
     */
    QPointF	mapViewToScene(qreal x, qreal y) const { return mapViewToScene(QPointF(x, y)); }

    /*!
     * \brief Maps from the local coordinate system of *item* to the view coordinates
     * \param item
     * \param point
     * \return
     */
    QPointF	mapFromItemToView(const QGraphicsItem* item, const QPointF& point) const { return mChildGroup->mapFromItem(item, point); }
    /*!
     * \brief Maps from the local coordinate system of *item* to the view coordinates
     * \param item
     * \param point
     * \return
     */
    QPointF	mapFromItemToView(const QGraphicsItem* item, const QPoint& point) const { return mChildGroup->mapFromItem(item, QPointF(point)); }
    /*!
     * \brief Maps from the local coordinate system of *item* to the view coordinates
     * \param item
     * \param rect
     * \return
     */
    QPolygonF mapFromItemToView(const QGraphicsItem* item, const QRectF& rect) const { return mChildGroup->mapFromItem(item, rect); }
    /*!
     * \brief Maps from the local coordinate system of *item* to the view coordinates
     * \param item
     * \param polygon
     * \return
     */
    QPolygonF mapFromItemToView(const QGraphicsItem* item, const QPolygonF& polygon) const { return mChildGroup->mapFromItem(item, polygon); }
    /*!
     * \brief Maps from the local coordinate system of *item* to the view coordinates
     * \param item
     * \param path
     * \return
     */
    QPainterPath mapFromItemToView(const QGraphicsItem* item, const QPainterPath& path) const { return mChildGroup->mapFromItem(item, path); }
    /*!
     * \brief Maps from the local coordinate system of *item* to the view coordinates
     * \param item
     * \param x
     * \param y
     * \param w
     * \param h
     * \return
     */
    QPolygonF mapFromItemToView(const QGraphicsItem* item, qreal x, qreal y, qreal w, qreal h ) const { return mChildGroup->mapFromItem(item, x, y, w, h); }
    /*!
     * \brief Maps from the local coordinate system of *item* to the view coordinates
     * \param item
     * \param x
     * \param y
     * \return
     */
    QPointF	mapFromItemToView(const QGraphicsItem* item, qreal x, qreal y) const { return mapFromItemToView(item, QPointF(x, y)); }

    /*!
     * \brief Maps from view coordinates to the local coordinate system of item.
     * \param item
     * \param point
     * \return
     */
    QPointF	mapFromViewToItem(const QGraphicsItem* item, const QPointF& point) const { return mChildGroup->mapToItem(item, point); }
    /*!
     * \brief Maps from view coordinates to the local coordinate system of item.
     * \param item
     * \param point
     * \return
     */
    QPointF	mapFromViewToItem(const QGraphicsItem* item, const QPoint& point) const { return mChildGroup->mapToItem(item, QPointF(point)); }
    /*!
     * \brief Maps from view coordinates to the local coordinate system of item.
     * \param item
     * \param rect
     * \return
     */
    QPolygonF mapFromViewToItem(const QGraphicsItem* item, const QRectF& rect) const { return mChildGroup->mapToItem(item, rect); }
    /*!
     * \brief Maps from view coordinates to the local coordinate system of item.
     * \param item
     * \param polygon
     * \return
     */
    QPolygonF mapFromViewToItem(const QGraphicsItem* item, const QPolygonF& polygon) const { return mChildGroup->mapToItem(item, polygon); }
    /*!
     * \brief Maps from view coordinates to the local coordinate system of item.
     * \param item
     * \param path
     * \return
     */
    QPainterPath mapFromViewToItem(const QGraphicsItem* item, const QPainterPath& path) const { return mChildGroup->mapToItem(item, path); }
    /*!
     * \brief Maps from view coordinates to the local coordinate system of item.
     * \param item
     * \param x
     * \param y
     * \param w
     * \param h
     * \return
     */
    QPolygonF mapFromViewToItem(const QGraphicsItem* item, qreal x, qreal y, qreal w, qreal h ) const { return mChildGroup->mapToItem(item, x, y, w, h); }
    /*!
     * \brief Maps from view coordinates to the local coordinate system of item.
     * \param item
     * \param x
     * \param y
     * \return
     */
    QPointF	mapFromViewToItem(const QGraphicsItem* item, qreal x, qreal y) const { return mapFromViewToItem(item, QPointF(x, y)); }

    QPointF	mapViewToDevice(const QPointF& point) const { return mapToDevice(mapFromView(point)); }
    QPointF	mapViewToDevice(const QPoint& point) const { return mapToDevice(mapFromView(QPointF(point))); }
    QPolygonF mapViewToDevice(const QRectF& rect) const { return mapToDevice(mapFromView(rect)); }
    QPolygonF mapViewToDevice(const QPolygonF& polygon) const { return mapToDevice(mapFromView(polygon)); }
    QPainterPath mapViewToDevice(const QPainterPath& path) const { return mapToDevice(mapFromView(path)); }
    QPointF	mapViewToDevice(qreal x, qreal y) const { return mapViewToDevice(QPointF(x, y)); }

    QPointF	mapDeviceToView(const QPointF& point) const { return mapToView(mapFromDevice(point)); }
    QPointF	mapDeviceToView(const QPoint& point) const { return mapToView(mapFromDevice(QPointF(point))); }
    QPolygonF mapDeviceToView(const QRectF& rect) const { return mapToView(mapFromDevice(rect)); }
    QPolygonF mapDeviceToView(const QPolygonF& polygon) const { return mapToView(mapFromDevice(polygon)); }
    QPainterPath mapDeviceToView(const QPainterPath& path) const { return mapToView(mapFromDevice(path)); }
    QPointF	mapDeviceToView(qreal x, qreal y) const { return mapDeviceToView(QPointF(x, y)); }

    /*!
     * \brief Bounding rect of the item in view coordinates
     * \param item
     * \return The bounding rect of the item in view coordinates
     */
    QRectF itemBoundingRect(const QGraphicsItem* item) const;

    /*!
     * \brief Set the visible range of the ViewBox.
     *
     *
     * \param xRange The range that should be visible along the x-axis
     * \param yRange The range that should be visible along the y-axis
     * \param padding Expand the view by a fraction of the requested range.
     *                By default (AutoPadding), this value is set between 0.02 and 0.1 depending on
     *                the size of the ViewBox.
     * \param disableAutoRange If True, auto-ranging is disabled. Otherwise, it is left unchanged
     */
    void setRange(const Range& xRange=Range(), const Range& yRange=Range(), const double padding=AutoPadding, const bool disableAutoRange=true);

    /*!
     * \brief Set the visible range of the ViewBox.
     *
     * \param rect The full range that should be visible in the view box.
     * \param padding Expand the view by a fraction of the requested range.
     *                By default (AutoPadding), this value is set between 0.02 and 0.1 depending on
     *                the size of the ViewBox.
     * \param disableAutoRange If True, auto-ranging is disabled. Otherwise, it is left unchanged
     */
    void setRange(const QRectF& rect, const double padding=AutoPadding, const bool disableAutoRange=true);

    /*!
     * \brief Set the visible X range of the view to [*min*, *max*].
     * \param minR Minimum value of the range
     * \param maxR Maximum value of the range
     * \param padding Expand the view by a fraction of the requested range.
     *                By default (AutoPadding), this value is set between 0.02 and 0.1 depending on
     *                the size of the ViewBox.
     */
    void setXRange(const double minR, const double maxR, const double padding=AutoPadding);

    /*!
     * \brief Set the visible Y range of the view to [*min*, *max*].
     * \param minR Minimum value of the range
     * \param maxR Maximum value of the range
     * \param padding Expand the view by a fraction of the requested range.
     *                By default (AutoPadding), this value is set between 0.02 and 0.1 depending on
     *                the size of the ViewBox.
     */
    void setYRange(const double minR, const double maxR, const double padding=AutoPadding);

    /*!
     * \brief Scale around given center point (or center of view).
     *
     * \param s Scale factor
     * \param center Center point
     */
    void scaleBy(const QPointF& s, const QPointF& center);

    /*!
     * \brief Scale around given center point.
     *
     * This allows the other axis to be left unaffected.
     *
     * Note:
     * Using a scale factor of 1.0 may cause slight changes due to floating-point error.
     *
     * \param x Scale factor on x axis
     * \param y Scale factor on y axis
     * \param center Center point
     */
    void scaleBy(const double x, const double y, const QPointF& center) { scaleBy(QPointF(x, y), center); }

    /*!
     * \brief Scale around center of view.
     *
     * \param s Scale factor
     */
    void scaleBy(const QPointF& s);

    /*!
     * \brief Scale around center of view.
     *
     * This allows the other axis to be left unaffected.
     *
     * Note:
     * Using a scale factor of 1.0 may cause slight changes due to floating-point error.
     *
     * \param x Scale factor on x axis
     * \param y Scale factor on y axis
     */
    void scaleBy(const double x, const double y) { scaleBy(QPointF(x, y)); }

    /*!
     * \brief Translate the view by.
     *
     * \param t Translate distance
     */
    void translateBy(const QPointF& t);

    /*!
     * \brief Translate the view by.
     *
     * x or y may be specified independently, leaving the other
     * axis unchanged.
     * Note:
     * Using a translation of 0 may still cause small changes due to floating-point error.
     *
     * \param x Translation distance on x axis
     * \param y Translation distance factor on y axis
     */
    void translateBy(const double x, const double y) { translateBy(QPointF(x, y)); }

    /*!
     * \brief Set the padding limits that constrain the possible view ranges.
     * \param xMin Minumum value in the x range
     * \param xMax Maximum value in the x range
     */
    void setXLimits(const double xMin, const double xMax) { mLimits.setXLimits(xMin, xMax); }

    /*!
     * \brief Set the padding limits that constrain the possible view ranges.
     * \param rng Allowed X range
     */
    void setXLimits(const Range& rng) { mLimits.setXLimits(rng.min(), rng.max()); }

    /*!
     * \brief Set the padding limits that constrain the possible view ranges.
     * \param yMin Minumum value in the y range
     * \param yMax Maximum value in the y range
     */
    void setYLimits(const double yMin, const double yMax) { mLimits.setYLimits(yMin, yMax); }

    /*!
     * \brief Set the padding limits that constrain the possible view ranges.
     * \param rng Allowed Y range
     */
    void setYLimits(const Range& rng) { mLimits.setYLimits(rng.min(), rng.max()); }

    /*!
     * \brief Set the scaling limits that constrain the possible view ranges.
     * \param xMin Minimum allowed left-to-right span across the view.
     * \param xMax Maximum allowed left-to-right span across the view.
     */
    void setXRangeLimits(const double xMin, const double xMax) { mLimits.setXRange(xMin, xMax); }

    /*!
     * \brief Set the scaling limits that constrain the possible view ranges.
     * \param rng Range allowed left-to-right span across the view.
     */
    void setXRangeLimits(const Range& rng) { mLimits.setXRange(rng.min(), rng.max()); }

    /*!
     * \brief Set the scaling limits that constrain the possible view ranges.
     * \param yMin Minimum allowed top-to-bottom span across the view.
     * \param yMax Maximum allowed top-to-bottom span across the view.
     */
    void setYRangeLimits(const double yMin, const double yMax) { mLimits.setYRange(yMin, yMax); }

    /*!
     * \brief Set the scaling limits that constrain the possible view ranges.
     * \param rng Range allowed left-to-right span across the view.
     */
    void setYRangeLimits(const Range& rng) { mLimits.setYRange(rng.min(), rng.max()); }

    Range xLimits() const { return mLimits.xLimits(); }
    Range yLimits() const { return mLimits.yLimits(); }

    Range xRangeLimits() const { return mLimits.xRange(); }
    Range yRangeLimits() const { return mLimits.yRange(); }

    MouseMode mouseMode() const { return mMouseMode; }

    /*!
     * \brief Set the mouse behavior for zooming an panning
     *
     * Set the mouse interaction mode. *mode* must be either ViewBox::PanMode or ViewBox::RectMode.
     * In PanMode, the left mouse button pans the view and the right button scales.
     * In RectMode, the left button draws a rectangle which updates the visible region
     * (this mode is more suitable for single-button mice)
     *
     * \param mode Mouse mode for zooming and panning
     */
    void setMouseMode(const MouseMode mode);

    void setWheelScaleFactor(const double factor) { mWheelScaleFactor = factor; }
    double wheelScaleFactor() const { return mWheelScaleFactor; }

    void linkedWheelEvent(QGraphicsSceneWheelEvent* event, const Axis axis=XYAxes);

    virtual void mouseDragEvent(MouseDragEvent* event);
    virtual void mouseDragEvent(MouseDragEvent* event, const Axis axis);

    /*!
     * \brief List of all children and grandchildren of this ViewBox
     * \param item Parent item. nullptr for the viewbox
     * \return All children and grandchildren of this ViewBox
     */
    QList<QGraphicsItem*> allChildren(QGraphicsItem* item=nullptr) const;

    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget=nullptr);

    /*!
     * \brief Screen geometry of the viewbox
     * \return The screen geometry of the viewbox
     */
    QRectF screenGeometry() const;

    /*!
     * \brief Width and Height of a screen pixel in view coordinates
     * \return The Width and Height of a screen pixel in view coordinates
     */
    Point viewPixelSize() const;

    void blockLink(const bool block) { mLinksBlocked = block; }
    bool linksBlocked() const { return mLinksBlocked; }

    /*!
     * \brief Link X or Y axes of two views and unlink any previously connected axes.
     * If view is nullptr, the axis is left unlinked.
     * \param axis Axis to link
     * \param view Other view to link with
     */
    void linkView(const Axis axis, ViewBoxBase* view);

    /*!
     * \brief Link this view's X axis to another view.
     * If view is nullptr, the axis is left unlinked.
     * \param view Other view to link with
     */
    void setXLink(ViewBoxBase* view);

    /*!
     * \brief Link this view's Y axis to another view.
     * If view is nullptr, the axis is left unlinked.
     * \param view Other view to link with
     */
    void setYLink(ViewBoxBase* view);

    // TODO: create a overload for the PlotItem that is a container of a ViewBox.
    //void linkView(const Axis axis, PlotItem* view);
    //void setXLink(PlotItem* view);
    //void setYLink(PlotItem* view);

    /*!
     * \brief Linked view for axis
     * \param axis Axis
     * \return The linked view or nullptr
     */
    ViewBoxBase* linkedView(const Axis axis) const;

    /*!
     * \brief Bounding range of all children.
     * Values may be invalid ranges if there are no specific bounds for an axis.
     * \param frac Fraction of teh data to consider
     * \param orthoRange Orthogonal (perpendicular) range to consider
     * \param items Items to consider.
     * \return The bounding ranges
     */
    QVector<Range> childrenBounds(const QPointF& frac, const Range& orthoRange, const QList<QGraphicsItem*>& items) const;
    /*!
     * \brief Bounding range of all children.
     * Values may be invalid ranges if there are no specific bounds for an axis.
     * \param frac Fraction of teh data to consider
     * \param orthoRange Orthogonal (perpendicular) range to consider
     * \return The bounding ranges
     */
    QVector<Range> childrenBounds(const QPointF& frac, const Range& orthoRange) const { return childrenBounds(frac, orthoRange, mAddedItems); }
    /*!
     * \brief Bounding range of all children.
     * Values may be invalid ranges if there are no specific bounds for an axis.
     * \param frac Fraction of teh data to consider
     * \return The bounding ranges
     */
    QVector<Range> childrenBounds(const QPointF& frac) const { return childrenBounds(frac, Range(), mAddedItems); }
    /*!
     * \brief Bounding range of all children.
     * Values may be invalid ranges if there are no specific bounds for an axis.
     * \return The bounding ranges
     */
    QVector<Range> childrenBounds() const { return childrenBounds(QPointF(1.0, 1.0), Range(), mAddedItems); }

    /*!
     * \brief Bounding rect of all children.
     * \param frac Fraction of teh data to consider
     * \param orthoRange Orthogonal (perpendicular) range to consider
     * \param items Items to consider.
     * \return Teh bounding rect of the items
     */
    QRectF childrenBoundingRect(const QPointF& frac, const Range& orthoRange, const QList<QGraphicsItem*>& items) const;
    /*!
     * \brief Bounding rect of all children.
     * \param frac Fraction of teh data to consider
     * \param orthoRange Orthogonal (perpendicular) range to consider
     * \return Teh bounding rect of the items
     */
    QRectF childrenBoundingRect(const QPointF& frac, const Range& orthoRange) const { return childrenBoundingRect(frac, orthoRange, mAddedItems); }
    /*!
     * \brief Bounding rect of all children.
     * \param frac Fraction of teh data to consider
     * \return Teh bounding rect of the items
     */
    QRectF childrenBoundingRect(const QPointF& frac) const { return childrenBoundingRect(frac, Range(), mAddedItems); }
    /*!
     * \brief Bounding rect of all children.
     * \return Teh bounding rect of the items
     */
    QRectF childrenBoundingRect() const { return childrenBoundingRect(QPointF(1.0, 1.0), Range(), mAddedItems); }

    /*!
     * \brief Set the range of the view box to make all children visible.
     *
     * Note that this is not the same as enableAutoRange, which causes the view to
     * automatically auto-range whenever its contents are changed.
     *
     * \param padding Expand the view by a fraction of the requested range.
     *                By default (AutoPadding), this value is set between 0.02 and 0.1 depending on
     *                the size of the ViewBox.
     */
    void autoRange(const double padding=AutoPadding);

    /*!
     * \brief Set the range of the view box to make all children visible.
     *
     * Note that this is not the same as enableAutoRange, which causes the view to
     * automatically auto-range whenever its contents are changed.
     *
     * \param items List of items to consider when determining the visible range.
     * \param padding Expand the view by a fraction of the requested range.
     *                By default (AutoPadding), this value is set between 0.02 and 0.1 depending on
     *                the size of the ViewBox.
     */
    void autoRange(const QList<QGraphicsItem*>& items, const double padding=AutoPadding);
\
    /*!
     * \brief Disables auto-range.
     *
     * See: enableAutoRange()
     *
     * \param ax Axis
     */
    void disableAutoRange(const Axis ax=XYAxes);

public slots:

    void prepareForPaint();

    void showAxRect(const QRectF& axRect);

    void scaleHistory(const int d);

    void linkedViewChanged(ViewBoxBase* view, const Axis axis);

    /*!
     * \brief called when x range of linked view has changed
     */
    void linkedXChanged();

    /*!
     * \brief called when y range of linked view has changed
     */
    void linkedYChanged();

    void updateAutoRange();

protected:

    void setViewRange(const Range& x, const Range& y);
    void setTargetRange(const Range& x, const Range& y);
    void setAutoRangeEnabled(const bool enableX, const bool enableY);

    /*!
     * \brief Reset target range to exactly match current view range.
     *
     * This is used during mouse interaction to prevent unpredictable
     * behavior (because the user is unaware of targetRange).
     */
    void _resetTarget();

    void setInnerSceneItem(GraphicsObject* innerItem);

    virtual QVariant itemChange(GraphicsItemChange change, const QVariant& value);

    virtual void wheelEvent(QGraphicsSceneWheelEvent* event);

    /*!
     * \brief This method should capture key presses in the current view box.
     * Key presses are used only when mouse mode is RectMode
     * The following events are implemented:
     * - ctrl-A : zooms out to the default "full" view of the plot
     * - ctrl-+ : moves forward in the zooming stack (if it exists)
     * - ctrl-- : moves backward in the zooming stack (if it exists)
     * \param event
     */
    virtual void keyPressEvent(QKeyEvent *event);

    void updateScaleBox(const QPointF& p1, const QPointF& p2);
    void hideScaleBox();

    void addToHistory(const QRectF& r);

    virtual void resizeEvent(QGraphicsSceneResizeEvent* event);

signals:

    void sigYRangeChanged(const Range& range);
    void sigXRangeChanged(const Range& range);
    void sigRangeChangedManually(const bool mouseLeft, const bool mouseRight);
    void sigRangeChanged(const Range& xRange, const Range& yRange);
    void sigStateChanged(ViewBoxBase* viewBox);
    void sigTransformChanged();
    void sigResized();

public:

    static constexpr double AutoPadding = std::numeric_limits<double>::quiet_NaN();

protected:

    bool mMatrixNeedsUpdate;
    bool mAutoRangeNeedsUpdate;

    bool mXInverted;
    bool mYInverted;

    QList<Range> mViewRange;    // actual range viewed
    QList<Range> mTargetRange;  // child coord. range visible [[xmin, xmax], [ymin, ymax]]
    double mAspectLocked;   // 0.0: aspect unlocked, double for the aspect ratio
    QList<bool> mAutoRangeEnabled;
    QList<bool> mAutoPan;  // whether to only pan (do not change scaling) when auto-range is enabled
    QList<bool> mAutoVisibleOnly;  // whether to auto-range only to the visible portion of a plot
    QVector<bool> mMouseEnabled;    //

    MouseMode mMouseMode;

    QGraphicsRectItem* mBackground = nullptr;

    GraphicsObject* mInnerSceneItem = nullptr;

    QGraphicsRectItem* mRbScaleBox = nullptr;

    ChildGroup* mChildGroup;

    QList<QGraphicsItem*> mAddedItems;

    QPen mBorder;

    Limits mLimits;

    double mWheelScaleFactor;

    QVector<QRectF> mAxHistory;
    int mAxHistoryPointer;

    bool mLinksBlocked;

    QVector<QWeakPointer<ViewBoxBase> > mLinkedViews;

    bool mUpdatingRange;
};

#endif // VIEWBOXBASE_H
