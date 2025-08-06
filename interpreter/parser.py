from .lexer import TokenType
from .ast_nodes import *

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]

    def advance(self):
        """Advance the token pointer and update the current token."""
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]

    def eat(self, token_type):
        """Consume the current token if it matches the expected type."""
        if self.current_token[0] == token_type:
            self.advance()
        else:
            raise SyntaxError(f"Expected {token_type}, but got {self.current_token[0]}")

    def parse(self):
        """Parse a list of statements."""
        statements = []
        while self.current_token[0] != TokenType.EOF:
            statements.append(self.statement())
        return statements

    def statement(self):
        """Parse a single statement."""
        token_type = self.current_token[0]

        if token_type == TokenType.PRINT:
            return self.print_statement()
        elif token_type == TokenType.DEF:
            return self.function_definition()
        elif token_type == TokenType.IF:
            return self.if_statement()
        elif token_type == TokenType.RETURN:
            self.advance()
            expr = self.comparison_expression()
            return ReturnStatement(expr)
        elif token_type == TokenType.IDENT:
            # Lookahead to distinguish between assignment and function call
            if len(self.tokens) > self.pos + 1 and self.tokens[self.pos + 1][0] == TokenType.ASSIGN:
                return self.assignment_statement()
            else:
                # An expression used as a statement (e.g., a function call)
                return self.comparison_expression()
        else:
            raise SyntaxError(f"Unexpected token {self.current_token} at start of statement")

    def print_statement(self):
        self.eat(TokenType.PRINT)
        self.eat(TokenType.LPAREN)
        expr = self.comparison_expression()
        self.eat(TokenType.RPAREN)
        return PrintStatement(expr)

    def assignment_statement(self):
        ident_token = self.current_token
        self.eat(TokenType.IDENT)
        self.eat(TokenType.ASSIGN)
        expr = self.comparison_expression()
        return Assign(Variable(ident_token), expr)

    def if_statement(self):
        self.eat(TokenType.IF)
        self.eat(TokenType.LPAREN)
        condition = self.comparison_expression()
        self.eat(TokenType.RPAREN)
        if_block = self.block()
        else_block = None
        if self.current_token[0] == TokenType.ELSE:
            self.eat(TokenType.ELSE)
            else_block = self.block()
        return IfStatement(condition, if_block, else_block)

    def function_definition(self):
        self.eat(TokenType.DEF)
        name_token = self.current_token
        self.eat(TokenType.IDENT)
        self.eat(TokenType.LPAREN)
        params = []
        if self.current_token[0] != TokenType.RPAREN:
            params.append(Variable(self.current_token))
            self.eat(TokenType.IDENT)
            while self.current_token[0] == TokenType.COMMA:
                self.eat(TokenType.COMMA)
                params.append(Variable(self.current_token))
                self.eat(TokenType.IDENT)
        self.eat(TokenType.RPAREN)
        body = self.block()
        return FunctionDef(name_token, params, body)

    def block(self):
        """Parses a block of statements enclosed in curly braces."""
        self.eat(TokenType.LBRACE)
        statements = []
        while self.current_token[0] not in (TokenType.RBRACE, TokenType.EOF):
            statements.append(self.statement())
        self.eat(TokenType.RBRACE)
        return Block(statements)

    def comparison_expression(self):
        """Parses comparison operators (==, !=, <, >, etc.)."""
        node = self.expr()
        op_types = [
            TokenType.EQ, TokenType.NE, TokenType.LT,
            TokenType.LTE, TokenType.GT, TokenType.GTE
        ]
        while self.current_token[0] in op_types:
            op_token = self.current_token
            self.eat(op_token[0])
            node = BinOp(node, op_token[1], self.expr())
        return node

    def expr(self):
        """Parses addition and subtraction."""
        node = self.term()
        while self.current_token[0] in (TokenType.PLUS, TokenType.MINUS):
            op_token = self.current_token
            self.eat(op_token[0])
            node = BinOp(node, op_token[1], self.term())
        return node

    def term(self):
        """Parses multiplication and division."""
        node = self.factor()
        while self.current_token[0] in (TokenType.MUL, TokenType.DIV):
            op_token = self.current_token
            self.eat(op_token[0])
            node = BinOp(node, op_token[1], self.factor())
        return node

    def factor(self):
        """Parses numbers, identifiers, function calls, and parenthesized expressions."""
        token_type, token_value = self.current_token

        if token_type in (TokenType.INTEGER, TokenType.FLOAT):
            self.advance()
            return Number((token_type, token_value))
        elif token_type == TokenType.IDENT:
            # Lookahead to see if it's a function call
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1][0] == TokenType.LPAREN:
                return self.call_expression()
            else:
                self.advance()
                return Variable((token_type, token_value))
        elif token_type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            node = self.comparison_expression()
            self.eat(TokenType.RPAREN)
            return node
        else:
            raise SyntaxError(f"Unexpected token {self.current_token} in expression")

    def call_expression(self):
        """Parses a function call."""
        name_token = self.current_token
        self.eat(TokenType.IDENT)
        self.eat(TokenType.LPAREN)
        args = []
        if self.current_token[0] != TokenType.RPAREN:
            args.append(self.comparison_expression())
            while self.current_token[0] == TokenType.COMMA:
                self.eat(TokenType.COMMA)
                args.append(self.comparison_expression())
        self.eat(TokenType.RPAREN)
        return FunctionCall(name_token[1], args)