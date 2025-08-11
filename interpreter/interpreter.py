import importlib
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
        
        # Define all built-in objects that should exist in the global scope
        self.global_scope = {
            'tensor': mlscript.Tensor,
            'matmul': self.e.matmul,
            'len': len,
            'range': range,
            'min': min,
            'max': max,
            'sum': sum,
        }
        
        # Load the built-ins into the C++ Evaluator's global scope
        for name, value in self.global_scope.items():
            self.e.assign_variable(name, value)

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
        value = self.visit(node.value_expr)
        indices = [self.visit(expr) for expr in node.index_expr]
        index = indices[0] if len(indices) == 1 else tuple(indices)

        line_num = node.token[2] # Get the line number from the node

        try:
            collection[index] = value
        except (IndexError, KeyError) as e:
            # Now includes the line number
            raise Exception(f"Runtime Error on line {line_num}: {e}")
        
    def visit_IndexAccess(self, node):
        collection = self.visit(node.collection)
        indices = [self.visit(expr) for expr in node.index_expr]
        index = indices[0] if len(indices) == 1 else tuple(indices)
        try:
            return collection[index]
        except (IndexError, KeyError, TypeError) as e:
            line_num = node.token[2]
            raise Exception(f"Runtime Error on line {line_num}: {e}")

    def no_visit_method(self, node):
        raise Exception(f"No visit_{type(node).__name__} method defined")

    def visit_Assign(self, node):
        value = self.visit(node.expr)
        self.e.assign_variable(node.left.name, value)


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
        if isinstance(iterable_value, (list, str,range)):
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
            return self.e.get_variable(node.name)
        except Exception as e:
            line_num = node.token[2] 
            raise Exception(f"Runtime Error on line {line_num}: {e}")
        
    def visit_AttributeAccess(self, node):
        obj = self.visit(node.obj)
        try:
            return getattr(obj, node.attribute)
        except AttributeError:
            line_num = node.token[2]
            raise Exception(f"Runtime Error on line {line_num}: Object '{obj}' has no attribute '{node.attribute}'") 

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
        
        # First, handle user-defined functions as a special case because they
        # manipulate the interpreter's scope stack directly.
        if isinstance(node.callee, Variable) and node.callee.name in self.functions:
            func_def = self.functions[node.callee.name]
            args = [self.visit(arg) for arg in node.args]

            if len(args) != len(func_def.params):
                raise Exception(f"Runtime Error on line {line_num}: Function '{func_def.name}' expects {len(func_def.params)} arguments, but received {len(args)}")
            
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

        # For all other functions (built-ins, imported functions), resolve the callee
        # expression to a callable Python object.
        callee_obj = self.visit(node.callee)
        
        if not callable(callee_obj):
            callee_repr = node.callee.name if isinstance(node.callee, Variable) else 'expression'
            raise Exception(f"Runtime Error on line {line_num}: '{callee_repr}' is not a function.")

        args = [self.visit(arg) for arg in node.args]
        
        try:
            return callee_obj(*args)
        except Exception as e:
            raise Exception(f"Runtime Error on line {line_num}: Error during function call: {e}")
        
    def visit_SliceNode(self, node):
        start = self.visit(node.start) if node.start else None
        stop = self.visit(node.stop) if node.stop else None
        step = self.visit(node.step) if node.step else None
        return slice(start,stop,step)
    
    def visit_ImportStatement(self, node):
        try:
            module = importlib.import_module(node.module_name)
            self.e.assign_variable(node.alias, module)
        except ImportError as e:
            line_num = node.token[2]
            raise Exception(f"Runtime Error on line {line_num}: {e}")