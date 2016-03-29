#include "Range.h"

int Range::metatypeId = 0;

void Range::registerMetatype()
{
    if(Range::metatypeId==0)
        Range::metatypeId = qRegisterMetaType<Range>("Range");
}
