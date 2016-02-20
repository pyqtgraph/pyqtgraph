
%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip



class ViewBoxInterface
{

%TypeHeaderCode
#include "Interfaces.h"
%End

public:

};




class ViewBoxGetterInterface
{

%TypeHeaderCode
#include "Interfaces.h"
%End

public:

    virtual ViewBoxInterface* getViewBox() const;
    virtual void forgetViewBox();

};





class ViewWidgetGetterInterface
{

%TypeHeaderCode
#include "Interfaces.h"
%End

public:
    virtual QGraphicsView* getViewWidget() const;
    virtual void forgetViewWidget();
};
