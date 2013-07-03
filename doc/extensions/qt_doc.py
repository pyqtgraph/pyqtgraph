"""
Extension for building Qt-like documentation.

 - Method lists preceding the actual method documentation
 - Inherited members documented separately
 - Members inherited from Qt have links to qt-project documentation
 - Signal documentation

"""



def setup(app):
    # probably we will be making a wrapper around autodoc
    app.setup_extension('sphinx.ext.autodoc')
    
    # would it be useful to define a new domain?
    #app.add_domain(QtDomain) 
    
    ## Add new configuration options
    app.add_config_value('todo_include_todos', False, False)

    ## Nodes are the basic objects representing documentation directives
    ## and roles
    app.add_node(Todolist)
    app.add_node(Todo,
                 html=(visit_todo_node, depart_todo_node),
                 latex=(visit_todo_node, depart_todo_node),
                 text=(visit_todo_node, depart_todo_node))

    ## New directives like ".. todo:"
    app.add_directive('todo', TodoDirective)
    app.add_directive('todolist', TodolistDirective)
    
    ## Connect callbacks to specific hooks in the build process
    app.connect('doctree-resolved', process_todo_nodes)
    app.connect('env-purge-doc', purge_todos)
    

from docutils import nodes
from sphinx.util.compat import Directive
from sphinx.util.compat import make_admonition


# Just a general node
class Todolist(nodes.General, nodes.Element):
    pass

# .. and its directive
class TodolistDirective(Directive):
    # all directives have 'run' method that returns a list of nodes
    def run(self):
        return [Todolist('')]




# Admonition classes are like notes or warnings
class Todo(nodes.Admonition, nodes.Element):
    pass

def visit_todo_node(self, node):
    self.visit_admonition(node)

def depart_todo_node(self, node):
    self.depart_admonition(node)    

class TodoDirective(Directive):

    # this enables content in the directive
    has_content = True

    def run(self):
        env = self.state.document.settings.env
    
        # create a new target node for linking to
        targetid = "todo-%d" % env.new_serialno('todo')
        targetnode = nodes.target('', '', ids=[targetid])

        # make the admonition node
        ad = make_admonition(Todo, self.name, [('Todo')], self.options,
                             self.content, self.lineno, self.content_offset,
                             self.block_text, self.state, self.state_machine)

        # store a handle in a global list of all todos
        if not hasattr(env, 'todo_all_todos'):
            env.todo_all_todos = []
        env.todo_all_todos.append({
            'docname': env.docname,
            'lineno': self.lineno,
            'todo': ad[0].deepcopy(),
            'target': targetnode,
        })

        # return both the linking target and the node itself
        return [targetnode] + ad


# env data is persistent across source files so we purge whenever the source file has changed.
def purge_todos(app, env, docname):
    if not hasattr(env, 'todo_all_todos'):
        return
    env.todo_all_todos = [todo for todo in env.todo_all_todos
                          if todo['docname'] != docname]
                          

# called at the end of resolving phase; we will convert temporary nodes
# into finalized nodes
def process_todo_nodes(app, doctree, fromdocname):
    if not app.config.todo_include_todos:
        for node in doctree.traverse(Todo):
            node.parent.remove(node)

    # Replace all todolist nodes with a list of the collected todos.
    # Augment each todo with a backlink to the original location.
    env = app.builder.env

    for node in doctree.traverse(Todolist):
        if not app.config.todo_include_todos:
            node.replace_self([])
            continue

        content = []

        for todo_info in env.todo_all_todos:
            para = nodes.paragraph()
            filename = env.doc2path(todo_info['docname'], base=None)
            description = (
                ('(The original entry is located in %s, line %d and can be found ') %
                (filename, todo_info['lineno']))
            para += nodes.Text(description, description)

            # Create a reference
            newnode = nodes.reference('', '')
            innernode = nodes.emphasis(('here'), ('here'))
            newnode['refdocname'] = todo_info['docname']
            newnode['refuri'] = app.builder.get_relative_uri(
                fromdocname, todo_info['docname'])
            newnode['refuri'] += '#' + todo_info['target']['refid']
            newnode.append(innernode)
            para += newnode
            para += nodes.Text('.)', '.)')

            # Insert into the todolist
            content.append(todo_info['todo'])
            content.append(para)

        node.replace_self(content)
        
