Mouse Interaction
=================

Most applications that use pyqtgraph's data visualization will generate widgets that can be interactively scaled, panned, and otherwise configured using the mouse. This section describes mouse interaction with these widgets.


2D Graphics
-----------

In pyqtgraph, most 2D visualizations follow the following mouse interaction:
    
* **Left button:** Interacts with items in the scene (select/move objects, etc). If there are no movable objects under the mouse cursor, then dragging with the left button will pan the scene instead.
* **Right button drag:** Scales the scene. Dragging left/right scales horizontally; dragging up/down scales vertically (although some scenes will have their x/y scales locked together). If there are x/y axes visible in the scene, then right-dragging over the axis will _only_ affect that axis.
* **Right button click:** Clicking the right button in most cases will show a context menu with a variety of options depending on the object(s) under the mouse cursor.
* **Middle button (or wheel) drag:** Dragging the mouse with the wheel pressed down will always pan the scene (this is useful in instances where panning with the left button is prevented by other objects in the scene).
* **Wheel spin:** Zooms the scene in and out.
    
For machines where dragging with the right or middle buttons is difficult (usually Mac), another mouse interaction mode exists. In this mode, dragging with the left mouse button draws a box over a region of the scene. After the button is released, the scene is scaled and panned to fit the box. This mode can be accessed in the context menu or by calling::
    
    pyqtgraph.setConfigOption('leftButtonPan', False)


Context Menu
------------

Right-clicking on most scenes will show a context menu with various options for changing the behavior of the scene. Some of the options available in this menu are:
    
* Enable/disable automatic scaling when the data range changes
* Link the axes of multiple views together
* Enable disable mouse interaction per axis
* Explicitly set the visible range values

The exact set of items available in the menu depends on the contents of the scene and the object clicked on.
    
    
3D Graphics
-----------

3D visualizations use the following mouse interaction:

* **Left button drag:** Rotates the scene around a central point
* **Middle button drag:** Pan the scene by moving the central "look-at" point within the x-y plane
* **Middle button drag + CTRL:** Pan the scene by moving the central "look-at" point along the z axis
* **Wheel spin:** zoom in/out
* **Wheel + CTRL:** change field-of-view angle

And keyboard controls:

* Arrow keys rotate around central point, just like dragging the left mouse button
