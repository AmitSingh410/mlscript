from .parser import Parser
from .lexer import tokenize
from .ast_nodes import *
import mlscript

class ReturnSignal(Exception):
    def __init__(self, value):
        self.value = value

class MLObject:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return str(self.value)

class Interpreter:
    def __init__(self):
        self.e = mlscript.Evaluator()
        self.functions = {}

    def run(self, code):
        tokens = tokenize(code)
        statements = Parser(tokens,code).parse()
        for stmt in statements:
            self.visit(stmt)

    def visit(self, node):
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.no_visit_method)
        return visitor(node)
    
    def visit_IndexAssign(self, node):
        collection = self.visit(node.collection)
        index = self.visit(node.index_expr)
        value = self.visit(node.value_expr)

        line_num = node.token[2] # Get the line number from the node

        if isinstance(collection, (list, dict)):
            try:
                collection[index] = value
            except (IndexError, KeyError) as e:
                # Now includes the line number
                raise Exception(f"Runtime Error on line {line_num}: {e}")
        else:
            # Now includes the line number
            raise Exception(f"Runtime Error on line {line_num}: Index assignment is only supported for lists and dictionaries.")
        
    def visit_IndexAccess(self, node):
        collection = self.visit(node.collection)
        index = self.visit(node.index_expr)
        try:
            return collection[index]
        except (IndexError, KeyError, TypeError) as e:
            line_num = node.token[2]
            raise Exception(f"Runtime Error on line {line_num}: {e}")

    def no_visit_method(self, node):
        raise Exception(f"No visit_{type(node).__name__} method defined")

    def visit_Assign(self, node):
        value = self.visit(node.expr)
        if isinstance(value,(bool,list,dict)):
            self.e.assign_variable(node.left.name, MLObject(value))
        else:
            self.e.assign_variable(node.left.name,value)

    def visit_UnaryOp(self, node):
        value = self.visit(node.expr)
        if node.op == '-':
            return -value
        elif node.op == '+':
            return value
        
    def visit_StringLiteral(self, node):
        return node.value
    
    def visit_ListLiteral(self, node):
        return [self.visit(elem) for elem in node.elements]
    
    def visit_DictLiteral(self, node):
        py_dict = {}
        for key_node, value_node in node.pairs:
            key = self.visit(key_node)
            value = self.visit(value_node)
            py_dict[key] = value
        return py_dict
    
    def visit_BooleanLiteral(self, node):
        return node.value

    def visit_PrintStatement(self, node):
        value = self.visit(node.expr)
        if isinstance(value,bool):
            print(str(value).lower())
        else:
            print(value)

    def visit_Block(self, node):
        for statement in node.statements:
            self.visit(statement)

    def visit_IfStatement(self, node):
        condition_value = self.visit(node.condition)
        if condition_value:
            self.visit(node.if_block)
        elif node.else_block:
            self.visit(node.else_block)

    def visit_WhileStatement(self, node):
        while self.visit(node.condition):
            self.visit(node.body)

    def visit_ForStatement(self, node):
        iterable_value = self.visit(node.iterable)
        
        iterator = None
        if isinstance(iterable_value, (list, str)):
            iterator = iterable_value
        else:
            line_num = node.iterable.token[2]
            raise Exception(f"Runtime Error on line {line_num}: 'for' loop can only iterate over a list, string, or range.")

        self.e.enter_scope()
        for item in iterator:
            self.e.assign_variable(node.variable.name, item)
            self.visit(node.body)
        self.e.exit_scope()

    def visit_FunctionDef(self, node):
        self.functions[node.name] = node

    def visit_ReturnStatement(self, node):
        value = self.visit(node.expr)
        raise ReturnSignal(value)

    def visit_Number(self, node):
        return node.value

    def visit_Variable(self, node):
        try:
            value = self.e.get_variable(node.name)
            if isinstance(value,MLObject):
                return value.value
            return value
        except Exception as e:
            line_num = node.token[2] 
            raise Exception(f"Runtime Error on line {line_num}: {e}")

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        op = node.op
        if op in ('+', '-', '*', '/'):
            return self.e.evaluate(op, left, right)
        elif op == '==': return left == right
        elif op == '!=': return left != right
        elif op == '<':  return left < right
        elif op == '<=': return left <= right
        elif op == '>':  return left > right
        elif op == '>=': return left >= right
        else:
            raise Exception(f"Unsupported binary operator: {op}")

    def visit_FunctionCall(self, node):
        line_num = node.token[2]
        
        # First, check if the callee is a simple variable name.
        if isinstance(node.callee, Variable):
            callee_name = node.callee.name
            
            # Handle Built-in functions by name
            builtins = {'len', 'range', 'min', 'max', 'sum'}
            if callee_name in builtins:
                args = [self.visit(arg) for arg in node.args] # Evaluate args for built-ins
                if callee_name == 'len':
                    if len(args) != 1: raise Exception(f"Runtime Error on line {line_num}: len() expects 1 argument, but received {len(args)}")
                    value = args[0]
                    if isinstance(value, (str, list, dict)): return len(value)
                    raise Exception(f"Runtime Error on line {line_num}: len() is only supported for strings, lists, and dictionaries.")
                
                if callee_name == 'range':
                    return list(range(*args))

                if callee_name in ('min', 'max', 'sum'):
                    if len(args) != 1: raise Exception(f"Runtime Error on line {line_num}: {callee_name}() expects 1 argument (a list), but received {len(args)}")
                    data = args[0]
                    if not isinstance(data, list): raise Exception(f"Runtime Error on line {line_num}: Argument to {callee_name}() must be a list.")
                    if not data: raise Exception(f"Runtime Error on line {line_num}: {callee_name}() arg is an empty sequence.")
                    if callee_name == 'min': return min(data)
                    if callee_name == 'max': return max(data)
                    if callee_name == 'sum': return sum(data)
            
            # Handle User-defined functions by name
            if callee_name in self.functions:
                func_def = self.functions[callee_name]
                args = [self.visit(arg) for arg in node.args] # Evaluate args for user functions

                if len(args) != len(func_def.params):
                    raise Exception(f"Runtime Error on line {line_num}: Function '{callee_name}' expects {len(func_def.params)} arguments, but received {len(args)}")
                
                self.e.enter_scope()
                try:
                    for param, arg_value in zip(func_def.params, args):
                        self.e.assign_variable(param.name, arg_value)
                    self.visit(func_def.body)
                except ReturnSignal as ret:
                    return ret.value
                finally:
                    self.e.exit_scope()
                return None

            # If the name is not a known built-in or user function, it's an error.
            raise Exception(f"Runtime Error on line {line_num}: Undefined function '{callee_name}'")

        # Case 2: The callee is a complex expression (e.g., my_list[0]())
        else:
            # For now, we can keep this restricted until we need it.
            raise Exception(f"Runtime Error on line {line_num}: Dynamic function execution is not yet supported.")