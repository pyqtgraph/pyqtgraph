import os
dirs = [
  ('graphicsItems', 'graphicsItems'),
  ('3dgraphics', 'opengl/items'),
  ('widgets', 'widgets'),
]

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
for a, b in dirs:
     rst = [os.path.splitext(x)[0].lower() for x in os.listdir(os.path.join(path, 'documentation', 'source', a))]
     py = [os.path.splitext(x)[0].lower() for x in os.listdir(os.path.join(path, b))]
     print a
     for x in set(py) - set(rst):
         print "    ", x
