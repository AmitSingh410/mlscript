from .lexer import TokenType
from .ast_nodes import *

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self): return self.tokens[self.pos][0]
    def next(self): self.pos += 1; return self.tokens[self.pos-1]

    def parse(self):
        stmts = []
        while self.peek() != TokenType.EOF:
            stmts.append(self.statement())
        return stmts

    def statement(self):
        if self.peek() == TokenType.IDENT:
            name = self.next()[1]
            self.expect(TokenType.ASSIGN)
            expr = self.expr()
            return Assign(name, expr)
        elif self.peek() == TokenType.PRINT:
            self.next()
            self.expect(TokenType.LPAREN)
            expr = self.expr()
            self.expect(TokenType.RPAREN)
            return Print(expr)
        else:
            raise SyntaxError("Expected a statement")

    def expr(self):
        node = self.term()
        while self.peek() in (TokenType.PLUS, TokenType.MINUS):
            op_token = self.next()
            op = op_token[1] # Store the character ('+' or '-') directly
            right = self.term()
            node = BinOp(node, op, right)
        return node

    def term(self):
        node = self.factor()
        while self.peek() in (TokenType.MUL, TokenType.DIV):
            op_token = self.next()
            op = op_token[1] # Store the character ('*' or '/') directly
            right = self.factor()
            node = BinOp(node, op, right)
        return node

    def factor(self):
        tok, val = self.tokens[self.pos]
        if tok in (TokenType.INTEGER, TokenType.FLOAT):
            self.next()
            return Number(val)
        elif tok == TokenType.IDENT:
            self.next()
            return Var(val)
        elif tok == TokenType.LPAREN:
            self.next()
            expr = self.expr()
            self.expect(TokenType.RPAREN)
            return expr
        else:
            raise SyntaxError("Expected a number or identifier")

    def expect(self, expected):
        if self.peek() != expected:
            raise SyntaxError(f"Expected {expected}, got {self.peek()}")
        self.next()