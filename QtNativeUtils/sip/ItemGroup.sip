
%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip


class ItemGroup: public GraphicsObject
{


%TypeHeaderCode
#include "ItemGroup.h"
%End


public:
    explicit ItemGroup(QGraphicsItem* parent=nullptr);
    virtual ~ItemGroup();

    virtual QRectF boundingRect() const;

    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget=nullptr);

    void addItem(QGraphicsItem* item);
};
