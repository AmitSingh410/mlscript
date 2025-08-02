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
                value = self.eval_expr(stmt.expr)
                self.e.set_var(stmt.name, str(value))  # Ensure string
            elif isinstance(stmt, Print):
                value = self.eval_expr(stmt.expr)
                print(value)

    def eval_expr(self, expr):
        if isinstance(expr, Number):
            return str(expr.value)  # Ensure string return
        elif isinstance(expr, Var):
            return self.e.eval_expr(expr.name)
        elif isinstance(expr, BinOp):
            left = str(self.eval_expr(expr.left))
            right = str(self.eval_expr(expr.right))
            op = str(self.token_to_str(expr.op))
            return self.e.eval_op(left, op, right)
        else:
            raise Exception(f"Unsupported expression type: {type(expr).__name__}")
        
    def token_to_str(self, token):
        return {
            TokenType.PLUS: '+',
            TokenType.MINUS: '-',
            TokenType.MUL: '*',
            TokenType.DIV: '/',
        }[token]
    
