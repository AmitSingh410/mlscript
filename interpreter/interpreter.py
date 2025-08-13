import importlib
from .parser import Parser
from .lexer import tokenize
from .ast_nodes import *
import mlscript

class ReturnSignal(Exception):
    def __init__(self, value):
        self.value = value

class MlscriptThrow(Exception):
    def __init__(self, value):
        self.value = value

class BreakSignal(Exception):
    """Signal to break out of a loop."""
    pass

class ContinueSignal(Exception):
    """Signal to continue to the next iteration of a loop."""
    pass

class MLObject:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return str(self.value)

class NoGradManager:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.original_state = None

    def __enter__(self):
        self.evaluator.set_grad_enabled(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.evaluator.set_grad_enabled(True)

class Interpreter:
    def __init__(self):
        self.e = mlscript.Evaluator()
        self.functions = {}
        
        self.global_scope = {
            'tensor': mlscript.Tensor,
            'matmul': self.e.matmul,
            'len': len,
            'range': range,
            'min': min,
            'max': max,
            'sum': sum,
            'no_grad': NoGradManager(self.e),
        }
        
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

        line_num = node.token[2]

        try:
            collection[index] = value
        except (IndexError, KeyError) as e:
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
            try:
                self.visit(node.body)
            except BreakSignal:
                break
            except ContinueSignal:
                continue

    def visit_ForStatement(self, node):
        iterable_value = self.visit(node.iterable)
        
        iterator = None
        if isinstance(iterable_value, (list, str,range,tuple)):
            iterator = iterable_value
        else:
            line_num = node.iterable.token[2]
            raise Exception(f"Runtime Error on line {line_num}: 'for' loop can only iterate over a list, string, tuple or range.")

        self.e.enter_scope()
        for item in iterator:
            self.e.assign_variable(node.variable.name, item)
            try:
                self.visit(node.body)
            except BreakSignal:
                break
            except ContinueSignal:
                continue
        self.e.exit_scope()

    def visit_WithStatement(self,node):
        context_manager = self.visit(node.context_expr)
        if not (hasattr(context_manager, '__enter__') and hasattr(context_manager, '__exit__')):
            line_num = node.context_expr.token[2]
            raise Exception(f"Runtime Error on line {line_num}: 'with' statement requires a context manager.")
        context_manager.__enter__()
        try:
            self.visit(node.body)
        finally:
            context_manager.__exit__(None, None, None)

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
        
        op = node.op_token[1]
        line_num = node.op_token[2]

        if op in ('+', '-', '*', '/'):
            return self.e.evaluate(op, left, right)
        elif op == '==': return left == right
        elif op == '!=': return left != right
        elif op == '<':  return left < right
        elif op == '<=': return left <= right
        elif op == '>':  return left > right
        elif op == '>=': return left >= right
        elif op == 'in':
            if isinstance(right, (list, str, dict, tuple)):
                return left in right
            else:
                raise Exception(f"Runtime Error on line {line_num}: The 'in' operator can only be used with lists, strings, dictionaries, or tuples.")
        elif op == 'not in':
            if isinstance(right, (list, str, dict, tuple)):
                return left not in right
            else:
                raise Exception(f"Runtime Error on line {line_num}: The 'not in' operator can only be used with lists, strings, dictionaries, or tuples.")
        else:
            raise Exception(f"Unsupported binary operator: {op} on line: {line_num}" )

    def visit_FunctionCall(self, node):
        line_num = node.token[2]
        
        if isinstance(node.callee, Variable) and node.callee.name in self.functions:
            func_def = self.functions[node.callee.name]
            args = [self.visit(arg) for arg in node.args]
            
            num_args = len(args)
            num_params = len(func_def.params)
            
            min_required_args = sum(1 for _, default in func_def.params if default is None)

            if num_args < min_required_args:
                raise Exception(f"Runtime Error on line {line_num}: Function '{func_def.name}' missing required arguments. Expected at least {min_required_args}, but received {num_args}.")
            
            if num_args > num_params:
                raise Exception(f"Runtime Error on line {line_num}: Function '{func_def.name}' takes at most {num_params} arguments, but {num_args} were given.")

            self.e.enter_scope()
            try:
                for i, (param_node, default_node) in enumerate(func_def.params):
                    if i < num_args:
                        self.e.assign_variable(param_node.name, args[i])
                    else:
                        default_value = self.visit(default_node)
                        self.e.assign_variable(param_node.name, default_value)
                
                self.visit(func_def.body)

            except ReturnSignal as ret:
                return ret.value
            finally:
                self.e.exit_scope()
            return None

        callee_obj = self.visit(node.callee)
        
        if isinstance(callee_obj, NoGradManager):
            raise Exception(f"Runtime Error on line {line_num}: 'no_grad' must be used in a 'with' statement, not called as a function.")

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
        
    def visit_TupleLiteral(self, node):
        elements =[]
        for elem in node.elements:
            elements.append(self.visit(elem))
        return tuple(elements)
    
    def visit_ThrowStatement(self, node):
        value_to_throw = self.visit(node.expr)
        raise MlscriptThrow(value_to_throw)

    def visit_TryCatch(self, node):
        try:
            try:
                self.visit(node.try_block)
            except MlscriptThrow as e:
                if node.catch_block:
                    self.e.enter_scope()
                    # Assign the caught error to the specified variable
                    self.e.assign_variable(node.catch_variable.name, e.value)
                    self.visit(node.catch_block)
                    self.e.exit_scope()
                else:
                    # If there's no catch block, the error continues up
                    raise e
        finally:
            # The finally block runs no matter what
            if node.finally_block:
                self.visit(node.finally_block)

    def visit_BreakStatement(self, node):
        raise BreakSignal()
    
    def visit_ContinueStatement(self, node):
        raise ContinueSignal()