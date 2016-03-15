%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip

class ChildGroup : public ItemGroup
{

%TypeHeaderCode
#include "ChildGroup.h"
%End

public:
    explicit ChildGroup(QGraphicsItem *parent = 0);
    virtual ~ChildGroup();

    void addListener(ItemChangedListener* listener /KeepReference/);

protected:

    virtual QVariant itemChange(GraphicsItemChange change, const QVariant& value);
};
