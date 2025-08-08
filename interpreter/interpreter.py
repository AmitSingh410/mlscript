from .parser import Parser
from .lexer import tokenize
from .ast_nodes import *
import mlscript

class ReturnSignal(Exception):
    def __init__(self, value):
        self.value = value

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
    
    def visit_BooleanLiteral(self, node):
        return node.value

    def visit_PrintStatement(self, node):
        value = self.visit(node.expr)
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
            return self.e.get_variable(node.name)
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
        if node.name == 'len':
            if len(node.args) != 1:
                raise Exception(f"Runtime Error on line {line_num}: len() expects 1 argument, but received {len(node.args)}")
            value = self.visit(node.args[0])
            if isinstance(value, (str, list)):
                return len(value)
            else:
                raise Exception(f"Runtime Error on line {line_num}: len() is only supported for strings and lists.")
        
        if node.name == 'range':
            args = [self.visit(arg) for arg in node.args]
            return list(range(*args))

        if node.name not in self.functions:
            raise Exception(f"Runtime Error on line {line_num}: Undefined function '{node.name}'")
        
        func_def = self.functions[node.name]
        
        if len(node.args) != len(func_def.params):
            raise Exception(f"Runtime Error on line {line_num}: Function '{node.name}' expects {len(func_def.params)} arguments, but received {len(node.args)}")
        
        self.e.enter_scope()
        
        try:
            for param, arg_node in zip(func_def.params, node.args):
                arg_value = self.visit(arg_node)
                self.e.assign_variable(param.name, arg_value)
            
            self.visit(func_def.body)

        except ReturnSignal as ret:
            return ret.value
        finally:
            self.e.exit_scope()
            
        return None