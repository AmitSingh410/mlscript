# ast_nodes.py

class Node:
    """Base class for all Abstract Syntax Tree nodes."""
    pass

class Number(Node):
    """Represents a literal integer or float value."""
    def __init__(self, token):
        self.token = token
        self.value = token[1]

class StringLiteral(Node):
    """Represents a string literal."""
    def __init__(self, token):
        self.token = token
        self.value = token[1] 

class Variable(Node):
    """Represents a variable identifier."""
    def __init__(self, token):
        self.token = token
        self.name = token[1]

class BinOp(Node):
    """Represents a binary operation (e.g., +, -, *, /, ==)."""
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class Assign(Node):
    """Represents a variable assignment (e.g., x = 5)."""
    def __init__(self, left, expr):
        self.left = left  # This is a Variable node
        self.expr = expr

class PrintStatement(Node):
    """Represents a print statement."""
    def __init__(self, expr):
        self.expr = expr

class Block(Node):
    """Represents a block of statements { ... }."""
    def __init__(self, statements):
        self.statements = statements

class IfStatement(Node):
    """Represents an if-else statement."""
    def __init__(self, condition, if_block, else_block=None):
        self.condition = condition
        self.if_block = if_block
        self.else_block = else_block

class WhileStatement(Node):
    """Represents a while loop."""
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class ForStatement(Node):
    """Represents a for loop."""
    def __init__(self, variable, iterable, body):
        self.variable = variable
        self.iterable = iterable 
        self.body = body


class FunctionDef(Node):
    """Represents a function definition."""
    def __init__(self, name_token, params, body):
        self.name = name_token[1]
        self.params = params  # List of Variable nodes
        self.body = body      # Block node

class FunctionCall(Node):
    """Represents a function call."""
    def __init__(self, name, args):
        self.name = name
        self.args = args

class ReturnStatement(Node):
    """Represents a return statement."""
    def __init__(self, expr):
        self.expr = expr