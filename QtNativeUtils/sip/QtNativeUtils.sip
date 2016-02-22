
%Module(name=QtNativeUtils, keyword_arguments="Optional")


%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip


%PostInitialisationCode
init_package();
%End

%ModuleCode
#include "QtNativeUtils.h"
%End


%Include Exceptions.sip

//%Include Interfaces.sip

%Include Point.sip
%Include MouseEvent.sip
%Include MouseClickEvent.sip
%Include HoverEvent.sip
%Include MouseDragEvent.sip

%Include QGraphicsObject2.sip
%Include QGraphicsWidget2.sip


%Include QGraphicsScene2.sip
%Include ViewBoxBase.sip
