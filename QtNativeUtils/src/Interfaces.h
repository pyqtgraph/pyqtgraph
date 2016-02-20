#ifndef INTERFACES_H
#define INTERFACES_H


#include <QGraphicsView>

class ViewBoxInterface
{
public:

};




class ViewBoxGetterInterface
{
public:

    virtual ViewBoxInterface* getViewBox() const
    {
        return nullptr;
    }

    virtual void forgetViewBox()
    {
        mViewBox = nullptr;
    }

protected:

    ViewBoxInterface* mViewBox;

};





class ViewWidgetGetterInterface
{
public:
    virtual QGraphicsView* getViewWidget() const
    {
        return nullptr;
    }

    virtual void forgetViewWidget() {}
};





#endif // INTERFACES_H
