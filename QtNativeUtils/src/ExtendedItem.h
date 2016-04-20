#ifndef EXTENDED_ITEM_H
#define EXTENDED_ITEM_H

#include <QGraphicsObject>
#include <QHash>
#include <QVariant>

#include "GraphicsViewBase.h"
#include "Point.h"
#include "mouseevents/MouseClickEvent.h"
#include "mouseevents/HoverEvent.h"
#include "mouseevents/MouseDragEvent.h"
#include "Range.h"


class GraphicsViewBase;
class ViewBoxBase;

/*!
 * \brief The ExtendedItem class
 *
 * Abstract class providing useful methods to GraphicsObject and GraphicsWidget.
 *
 * A note about Qt's GraphicsView framework:
 * The GraphicsView system places a lot of emphasis on the notion that the graphics within
 * the scene should be device independent--you should be able to take the same graphics
 * and display them on screens of different resolutions, printers, export to SVG, etc.
 * This is nice in principle, but causes me a lot of headache in practice.
 * It means that I have to circumvent all the device-independent expectations any time
 * I want to operate in pixel coordinates rather than arbitrary scene coordinates.
 * A lot of the code in GraphicsItem is devoted to this task--keeping track of view widgets
 * and device transforms, computing the size and shape of a pixel in local item coordinates,
 * etc.
 * Note that in item coordinates, a pixel does not have to be square or even rectangular, so
 * just asking how to increase a bounding rect by 2px can be a rather complex task.
 */
class ExtendedItem
{

public:

    enum Axis
    {
        XAxis = 0,
        YAxis = 1,
        XYAxes = 2
    };


public:
    ExtendedItem(QGraphicsObject* impl) : mItemImpl(impl) {}
    virtual ~ExtendedItem() {}

    /*!
     * \brief GraphicsView taht contains this item
     * \return The GraphicsView
     */
    virtual GraphicsViewBase* getViewWidget() const;

    virtual void forgetViewWidget()
    {
        mView = nullptr;
    }

    QList<QGraphicsItem*> getBoundingParents() const;

    QVector<Point> pixelVectors() const
    {
        return pixelVectors(QPointF(1.0, 0.0));
    }

    QVector<Point> pixelVectors(const QPointF& direction) const;

    double pixelLength(const QPointF& direction, const bool ortho=false) const
    {
        QVector<Point> p = pixelVectors(direction);
        if(ortho)
            return p[1].length();
        return p[0].length();
    }

    virtual QTransform deviceTransform() const = 0;

    QPointF	mapFromDevice(const QPointF& point) const { return deviceTransform().inverted().map(point); }
    QPointF	mapFromDevice(const QPoint& point) const { return deviceTransform().inverted().map(QPointF(point)); }
    QPolygonF mapFromDevice(const QRectF& rect) const { return deviceTransform().inverted().map(rect); }
    QPolygonF mapFromDevice(const QPolygonF& polygon) const { return deviceTransform().inverted().map(polygon); }
    QPainterPath mapFromDevice(const QPainterPath& path) const { return deviceTransform().inverted().map(path); }
    QPointF	mapFromDevice(qreal x, qreal y) const { return mapFromDevice(QPointF(x, y)); }

    QPointF	mapToDevice(const QPointF& point) const { return deviceTransform().map(point); }
    QPointF	mapToDevice(const QPoint& point) const { return deviceTransform().map(QPointF(point)); }
    QPolygonF mapToDevice(const QRectF& rect) const { return deviceTransform().map(rect); }
    QPolygonF mapToDevice(const QPolygonF& polygon) const { return deviceTransform().map(polygon); }
    QPainterPath mapToDevice(const QPainterPath& path) const { return deviceTransform().map(path); }
    QPointF	mapToDevice(qreal x, qreal y) const { return mapToDevice(QPointF(x, y)); }

    QRectF mapRectToDevice(const QRectF& rect) const { return deviceTransform().mapRect(rect); }
    QRect mapRectToDevice(const QRect& rect) const { return deviceTransform().mapRect(rect); }

    QRectF mapRectFromDevice(const QRectF& rect) const { return deviceTransform().inverted().mapRect(rect); }
    QRect mapRectFromDevice(const QRect& rect) const { return deviceTransform().inverted().mapRect(rect); }

    QPointF	mapToView(const QPointF& point) const { return viewTransform().map(point); }
    QPointF	mapToView(const QPoint& point) const { return viewTransform().map(QPointF(point)); }
    QPolygonF mapToView(const QRectF& rect) const { return viewTransform().map(rect); }
    QPolygonF mapToView(const QPolygonF& polygon) const { return viewTransform().map(polygon); }
    QPainterPath mapToView(const QPainterPath& path) const { return viewTransform().map(path); }
    QPointF	mapToView(qreal x, qreal y) const { return mapToView(QPointF(x, y)); }

    QRectF mapRectToView(const QRectF& rect) const { return viewTransform().mapRect(rect); }
    QRect mapRectToView(const QRect& rect) const { return viewTransform().mapRect(rect); }

    QPointF	mapFromView(const QPointF& point) const { return viewTransform().inverted().map(point); }
    QPointF	mapFromView(const QPoint& point) const { return viewTransform().inverted().map(QPointF(point)); }
    QPolygonF mapFromView(const QRectF& rect) const { return viewTransform().inverted().map(rect); }
    QPolygonF mapFromView(const QPolygonF& polygon) const { return viewTransform().inverted().map(polygon); }
    QPainterPath mapFromView(const QPainterPath& path) const { return viewTransform().inverted().map(path); }
    QPointF	mapFromView(qreal x, qreal y) const { return mapFromView(QPointF(x, y)); }

    QRectF mapRectFromView(const QRectF& rect) const { return viewTransform().inverted().mapRect(rect); }
    QRect mapRectFromView(const QRect& rect) const { return viewTransform().inverted().mapRect(rect); }

    /*!
     * \brief Angle between this item and a relativeItem
     * \param relativeItem The relative item. If nullptr, use parent item.
     * \return The angle between this item and another item
     */
    double transformAngle(QGraphicsItem* relativeItem=nullptr) const;

    virtual void mouseClickEvent(MouseClickEvent* event) { event->ignore(); }
    virtual void hoverEvent(HoverEvent* event) { event->ignore(); }
    virtual void mouseDragEvent(MouseDragEvent* event) { event->ignore(); }

    /*!
     * \brief Parent ViewBox
     * \return The parent ViewBox or nullptr is not exists
     */
    virtual ViewBoxBase* getNativeViewBox() const;

    /*!
     * \brief First parent ViewBox or GraphicsView
     *
     * Return the first ViewBox or GraphicsView which bounds this item's visible space.
     * If this item is not contained within a ViewBox, then the GraphicsView is returned.
     * If the item is contained inside nested ViewBoxes, then the inner-most ViewBox is returned.
     * The result is cached; clear the cache with forgetViewBox()
     *
     * \return The first parent ViewBox or GraphicsView
     */
    QObject* getViewBox() const
    {
        getNativeViewBox();
        if(mViewBox!=nullptr)
            return (QObject*)mViewBox;
        else if(mView!=nullptr)
            return (QObject*)mView;

        return nullptr;
    }

    /*!
     * \brief Clear the ViewBox cache
     */
    virtual void forgetViewBox()
    {
        mViewBox = nullptr;
        mViewBoxIsViewWidget = false;
    }

    /*!
     * \brief Transform from local to view coordinates
     *
     * Return the transform that maps from local coordinates to the item's ViewBox coordinates.
     * If there is no ViewBox, return the scene transform.
     *
     * \return The transform from local to view coordinates
     */
    virtual QTransform viewTransform() const;

    /*!
     * \brief Bounds item's ViewBox or GraphicsWidget
     *
     * \return The bounds (in item coordinates) of this item's ViewBox or GraphicsWidget
     */
    virtual QRectF viewRect() const;

    /*!
     * \brief List of the entire item tree descending from this item.
     * \param root Item used as parent. If nullptr, use this.
     * \return The list of the entire item tree descending from this item.
     */
    QList<QGraphicsItem*> allChildItems(QGraphicsItem* root=nullptr) const;

    /*!
     * \brief Union of the shapes of all descendants of this item in local coordinates.
     * \return The union of the shapes of all descendants of this item in local coordinates.
     */
    QPainterPath childrenShape() const;

    QPointF pixelSize() const;
    double pixelWidth() const;
    double pixelHeight() const;

    QPointF viewPos() const;

    /*!
     * \brief Inform this item's container ViewBox that the bounds of this item have changed.
     *
     * This is used by ViewBox to react if auto-range is enabled.
     */
    virtual void informViewBoundsChanged();

    /*!
     * \brief Called to see whether this item has a new view to connect to
     *
     * This is called from GraphicsObject.itemChange or GraphicsWidget.itemChange.
     *
     * It is possible this item has moved to a different ViewBox or widget;
     * clear out previously determined references to these.
     */
    virtual void _updateView();

    void _replaceView(QGraphicsItem* item=nullptr);

    bool isViewBox(const ViewBoxBase* vb) const;
    bool isViewBox(const GraphicsViewBase* vb) const;

    /*!
     * \brief Called when the item's parent has changed.
     *
     * Handles connecting / disconnecting from ViewBox signals
     * to make sure viewRangeChanged works properly. It should generally be
     * extended, not overridden.
     */
    void parentIsChanged();

    /*!
     * \brief Called whenever the view coordinates of the ViewBox containing this item have changed.
     * \param xRange New Range of the ViewBox in x coordinates
     * \param yRange New Range of the ViewBox in y coordinates
     */
    virtual void viewRangeChanged(const Range& xRange, const Range& yRange) = 0;

    /*!
     * \brief Called whenever the transformation matrix of the view has changed.
     *
     * Eg, the view range has changed or the view has been resized.
     */
    virtual void viewTransformChanged() = 0;

    void setExportMode(const bool isExporting, const QVariantHash& opt=QVariantHash());

    const QVariantHash& getExportMode() const;

protected:

    virtual void viewChanged();

    virtual void disconnectView(ViewBoxBase* view) = 0;
    virtual void disconnectView(GraphicsViewBase* view) = 0;

    virtual void connectView(ViewBoxBase* view) = 0;
    virtual void connectView(GraphicsViewBase* view) = 0;

protected:

    mutable GraphicsViewBase* mView = nullptr;
    mutable ViewBoxBase* mViewBox = nullptr;
    mutable bool mViewBoxIsViewWidget = false;

    QVariantHash mExportOptions;

private:

    QGraphicsObject* mItemImpl;


};

#endif // ENABLE_EXTENDEDTEM_CODE

