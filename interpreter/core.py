import importlib
from .parser import Parser
from .lexer import tokenize
from .ast_nodes import *
from interpreter import mlscript 

class C3_MRO:
    @staticmethod
    def resolve(cls):
        """Calculates the Method Resolution Order for a class using the C3 algorithm."""
        mro = [cls]
        parent_mros = [p.mro for p in cls.parents]
        mro.extend(C3_MRO._merge(parent_mros))
        return mro

    @staticmethod
    def _merge(mro_list):
        """Merges a list of parent MROs according to C3 linearization."""
        if not any(mro_list):
            return []
        
        result = []
        # Make a copy of the lists to avoid modifying them in place
        mro_list_copy = [list(mro) for mro in mro_list]

        while True:
            # Filter out empty MROs from our copy
            mro_list_copy = [mro for mro in mro_list_copy if mro]
            if not mro_list_copy:
                break

            # Find a "good head" that does not appear in the TAIL of any other MRO
            good_head = None
            for mro in mro_list_copy:
                head = mro[0]
                is_good_head = True
                for other_mro in mro_list_copy:
                    if head in other_mro[1:]:
                        is_good_head = False
                        break
                if is_good_head:
                    good_head = head
                    break
            
            if good_head is None:
                raise Exception("Cannot create a consistent Method Resolution Order (MRO).")

            result.append(good_head)

            # Remove the good head from the front of all lists in our copy
            for mro in mro_list_copy:
                if mro[0] == good_head:
                    del mro[0]
        
        return result


class MlscriptClass:
    def __init__(self, name, parents, methods):
        self.name = name
        self.parents = parents
        self.methods = methods
        self.mro = C3_MRO.resolve(self)

    def __call__(self, interpreter, args, kwargs):
        instance = MlscriptInstance(self)
        initializer_tuple = self.find_method("init")
        if initializer_tuple:
            method_node, defining_class = initializer_tuple
            bound_init = MlscriptBoundMethod(instance, method_node, defining_class)
            bound_init(interpreter, args,kwargs)
        elif len(args) > 0 or len(kwargs) > 0:
            raise Exception(f"Error: '{self.name}' constructor takes no arguments, but {len(args)} were given.")
        return instance

    def find_method(self, name):
        for cls in self.mro:
            if name in cls.methods:
                return (cls.methods[name], cls)
        return None

    def __repr__(self):
        return f"<class '{self.name}'>"

class MlscriptBoundMethod:
    def __init__(self, instance, func_def_node, defining_class):
        self.instance = instance
        self.func_def_node = func_def_node
        self.defining_class = defining_class

    def __call__(self, interpreter, args, kwargs):
        return interpreter._call_function(self.func_def_node, args, kwargs,instance=self.instance, defining_class=self.defining_class)
    
class MlscriptInstance:
    def __init__(self,klass):
        self.klass = klass
        self.fields = {}

    def __repr__(self):
        return f"<{self.klass.name} instance>"
    
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
        self.mlscript = mlscript
        self.e = mlscript.Evaluator()
        self.functions = {}
        self.method_context_stack = []

        self.global_scope = {
            'tensor': mlscript.Tensor,
            'matmul': self.e.matmul,
            'len': len,
            'range': range,
            'min': min,
            'max': max,
            'sum': sum,
            'no_grad': NoGradManager(self.e),
            'Dense': mlscript.Dense,
            'ReLU': mlscript.ReLU,
            'Sigmoid': mlscript.Sigmoid,
            'Flatten': mlscript.Flatten,
            'MSELoss': mlscript.MSELoss,
            'CrossEntropyLoss': mlscript.CrossEntropyLoss,
            'SGD': mlscript.SGD,
            'Adam': mlscript.Adam
        }
        
        for name, value in self.global_scope.items():
            self.e.assign_variable(name, value)

    def run(self, code):
        tokens = tokenize(code)
        statements = Parser(tokens,code).parse()
        for stmt in statements:
            self.visit(stmt)

    def visit(self, node):
        if isinstance(node,NetworkLiteral):
            return self.visit_NetworkLiteral(node)
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
        values = []
        for expr in node.exprs:
            value = self.visit(expr)
            if isinstance(value, bool):
                values.append(str(value).lower())
            else:
                values.append(value)
        
        print(*values)

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
        attribute_name = node.attribute

        if isinstance(obj, MlscriptInstance):
            if attribute_name in obj.fields:
                return obj.fields[attribute_name]
            
            method_node = obj.klass.find_method(attribute_name)
            if method_node:
                return MlscriptBoundMethod(obj, method_node)
            
            line_num = node.token[2]
            raise Exception(f"Runtime Error on line {line_num}: Object of type '{obj.klass.name}' has no attribute or method '{attribute_name}'")
        
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
            try:
                return self.e.evaluate(op, left, right)
            except Exception as e:
                raise Exception(f"Runtime Error on line {line_num}: {e}")
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
        args = [self.visit(arg) for arg in node.args]
        kwargs = {key: self.visit(value) for key,value in node.kwargs.items()}

        if isinstance(node.callee, Variable) and node.callee.name in self.functions:
            func_def = self.functions[node.callee.name]
            return self._call_function(func_def, args, kwargs)

        callee_obj = self.visit(node.callee)

        if isinstance(callee_obj, MlscriptClass):
            return callee_obj(self, args, kwargs)
        
        if isinstance(callee_obj, MlscriptBoundMethod):
            return callee_obj(self, args, kwargs)
        
        if isinstance(callee_obj, NoGradManager):
            raise Exception(f"Runtime Error on line {line_num}: 'no_grad' must be used in a 'with' statement, not called as a function.")

        if not callable(callee_obj):
            callee_repr = node.callee.name if isinstance(node.callee, Variable) else 'expression'
            raise Exception(f"Runtime Error on line {line_num}: '{callee_repr}' is not a function.")

        try:
            return callee_obj(*args, **kwargs)
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
    
    def visit_ClassDef(self,node):
        parents = []
        for parent_node in node.parents:
            parent_obj = self.visit(parent_node)
            if not isinstance(parent_obj, MlscriptClass):
                line_num = node.parent.token[2]
                raise Exception(f"Runtime Error on line {line_num}: 'inherits' must be followed by a class name.")
            parents.append(parent_obj)

        class_name = node.name
        methods = {method.name: method for method in node.methods}
        klass = MlscriptClass(class_name,parents, methods)
        self.e.assign_variable(class_name,klass)
        return None
    
    def visit_AttributeAccess(self, node):
        obj = self.visit(node.obj)
        attribute_name = node.attribute
        
        if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], MlscriptClass):
            instance, search_start_class = obj
            method_tuple = search_start_class.find_method(attribute_name)
            if not method_tuple:
                line_num = node.token[2]
                raise Exception(f"Runtime Error on line {line_num}: No method '{attribute_name}' found in superclass chain.")
            
            method_node, defining_class = method_tuple
            return MlscriptBoundMethod(instance, method_node, defining_class)

        if isinstance(obj, MlscriptInstance):
            if attribute_name in obj.fields:
                return obj.fields[attribute_name]
            
            method_tuple = obj.klass.find_method(attribute_name)
            if method_tuple:
                method_node, defining_class = method_tuple
                return MlscriptBoundMethod(obj, method_node, defining_class)
            
            line_num = node.token[2]
            raise Exception(f"Runtime Error on line {line_num}: Object of type '{obj.klass.name}' has no attribute or method '{attribute_name}'")

        try:
            return getattr(obj, attribute_name)
        except AttributeError:
            line_num = node.token[2]
            raise Exception(f"Runtime Error on line {line_num}: Object '{obj}' has no attribute '{attribute_name}'")
        
    def visit_SuperNode(self, node):
        if not self.method_context_stack:
            line_num = node.token[2]
            raise Exception(f"Runtime Error on line {line_num}: 'super' can only be used inside a class method.")

        instance, defining_class = self.method_context_stack[-1]

        instance_mro = instance.klass.mro
        try:
            idx = instance_mro.index(defining_class)
            search_start_class = instance_mro[idx + 1]
            return (instance, search_start_class)
        except (ValueError, IndexError):
            line_num = node.token[2]
            raise Exception(f"Runtime Error on line {line_num}: 'super' could not find a valid superclass in the MRO for class '{defining_class.name}'.")
        
    def visit_AttributeAssign(self, node):
        obj = self.visit(node.obj)
        if not isinstance(obj, MlscriptInstance):
            line_num = node.token[2]
            raise Exception(f"Runtime Error on line {line_num}: Cannot assign attribute '{node.attribute}' to non-instance object '{obj}'.")
        
        value = self.visit(node.value_expr)
        obj.fields[node.attribute] = value
        return value
    
    def _call_function(self, func_def, args, kwargs, instance=None, defining_class=None):
        params = list(func_def.params)
        param_names = [p[0].name for p in params]
        final_args = {}

        # 1. Handle instance 'self' - it's a special positional argument
        if instance:
            if not param_names:
                raise Exception(f"Error: Method '{func_def.name}' has no 'self' parameter.")
            final_args[param_names[0]] = instance
            # Remove 'self' from the list of params to match against
            param_names.pop(0) 
            params.pop(0)

        # 2. Match positional args to remaining params
        for i, arg_value in enumerate(args):
            if i >= len(param_names):
                raise Exception(f"Error: Function '{func_def.name}' received too many positional arguments.")
            param_name = param_names[i]
            if param_name in kwargs:
                 raise Exception(f"Error: Function '{func_def.name}' got multiple values for argument '{param_name}'")
            final_args[param_name] = arg_value
        
        # 3. Handle keyword arguments
        for key, value in kwargs.items():
            if key not in param_names:
                raise Exception(f"Error: Function '{func_def.name}' got an unexpected keyword argument '{key}'")
            if key in final_args:
                raise Exception(f"Error: Function '{func_def.name}' got multiple values for argument '{key}'")
            final_args[key] = value

        if instance:
            self.method_context_stack.append((instance, defining_class))

        self.e.enter_scope()
        try:
            for param_node, default_node in params:
                param_name = param_node.name
                if param_name in final_args:
                    self.e.assign_variable(param_name, final_args[param_name])
                elif default_node is not None:
                    default_value = self.visit(default_node)
                    self.e.assign_variable(param_name, default_value)
                elif instance and param_name == param_names[0]:
                    continue # Skip 'self' if it wasn't provided
                else:
                    raise Exception(f"Error: Function '{func_def.name}' missing required argument: '{param_name}'")

            return self.visit(func_def.body)

        except ReturnSignal as ret:
            return ret.value
        finally:
            self.e.exit_scope()
            if instance:
                self.method_context_stack.pop()

        return None
    
    def visit_NetworkLiteral(self,node):
        attrs = node.attributes

        # Building Model Architecture
        input_shape = self.visit(attrs['input'])
        architecture = self.mlscript.Sequential()
        current_shape = input_shape

        layer_list_node = attrs['layers']
        if not isinstance(layer_list_node,ListLiteral):
            raise Exception("The 'layers' attribute in a network block must be a list.")
        
        for layer_call_node in layer_list_node.elements:
            if not isinstance(layer_call_node,FunctionCall):
                raise Exception("Elements in the 'layers' list must be layer calls, e.g., Dense(...)")
            
            layer_class = self.visit(layer_call_node.callee)
            args = [self.visit(arg) for arg in layer_call_node.args]

            if layer_class == self.mlscript.Dense:
                if len(args)!=1:
                    raise Exception(f"Dense() in a network block expects 1 argument (output_Features), got {len(args)}.")
                output_features = args[0]
                layer_instance = layer_class(current_shape, output_features)
                current_shape = output_features
            else:
                layer_instance = layer_class(*args)

            architecture.add_module(layer_instance)

        # Building the Optimizer 
        optimizer_node = attrs['optimizer']
        if not isinstance(optimizer_node,FunctionCall):
            raise Exception("The 'optimizer' attribute must be a function call, e.g., Adam(...)")
        
        optimizer_class = self.visit(optimizer_node.callee)

        optimizer_pos_args = [self.visit(arg) for arg in optimizer_node.args]
        optimizer_kwargs = {key: self.visit(value) for key, value in optimizer_node.kwargs.items()}

        model_params = architecture.parameters()
        optimizer = optimizer_class(model_params, *optimizer_pos_args, **optimizer_kwargs)

        # Building the Loss functions
        loss_fn_node = attrs['loss']
        if not isinstance(loss_fn_node,FunctionCall):
            raise Exception("The 'loss' attribute must be a function call, e.g., CrossEntropyLoss()")
        
        loss_fn_class = self.visit(loss_fn_node.callee)
        loss_fn_args = [self.visit(arg) for arg in loss_fn_node.args]
        loss_fn = loss_fn_class(*loss_fn_args)

        # Final Assembly
        return self.mlscript.AssembledModel(architecture,optimizer,loss_fn)    