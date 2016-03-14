
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

%Include ExtendedItem.sip

%Include Point.sip
%Include MouseEvent.sip
%Include MouseClickEvent.sip
%Include HoverEvent.sip
%Include MouseDragEvent.sip

%Include GraphicsObject.sip
%Include GraphicsWidget.sip


%Include QGraphicsScene2.sip
%Include ViewBoxBase.sip
%Include PlotItemBase.sip
%Include GraphicsViewBase.sip
