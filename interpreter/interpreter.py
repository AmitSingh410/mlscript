from .parser import Parser
from .lexer import tokenize, TokenType
from .ast_nodes import Assign, Print, BinOp, Number, Var
import mlscript

class Interpreter:
    def __init__(self):
        self.e = mlscript.Evaluator()

    def run(self, code):
        tokens = tokenize(code)
        ast = Parser(tokens).parse()
        for stmt in ast:
            if isinstance(stmt, Assign):
                # Pass the actual numeric value to set_var
                value = self.eval_expr(stmt.expr)
                self.e.set_var(stmt.name, value)
            elif isinstance(stmt, Print):
                value = self.eval_expr(stmt.expr)
                print(value)

    def eval_expr(self, expr):
        if isinstance(expr, Number):
            # Return the number directly (it's already an int or float)
            return expr.value
        elif isinstance(expr, Var):
            # C++ will return an int or float via the variant
            return self.e.get_var(expr.name)
        elif isinstance(expr, BinOp):
            # Evaluate expressions to numbers first
            left = self.eval_expr(expr.left)
            right = self.eval_expr(expr.right)
            # Pass the numbers and operator string to C++
            return self.e.eval_op(left, expr.op, right)
        else:
            raise Exception(f"Unsupported expression type: {type(expr).__name__}")