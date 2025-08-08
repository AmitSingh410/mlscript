from ast import UnaryOp
from .lexer import TokenType
from .ast_nodes import *

class Parser:
    def __init__(self, tokens,code):
        self.tokens = tokens
        self.code_lines = code.split('\n')
        self.pos = 0
        self.current_token = self.tokens[self.pos]

    def advance(self):
        """Advance the token pointer and update the current token."""
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]

    def error(self, expected_type):
        token_type, token_value, line_num = self.current_token
        line = self.code_lines[line_num - 1]

        error_message = f"""SyntaxError: Expected {expected_type}, but got {token_type} at line {line_num}: {line.strip()}"""
        raise Exception(error_message)


    def eat(self, token_type):
        """Consume the current token if it matches the expected type."""
        if self.current_token[0] == token_type:
            self.advance()
        else:
            self.error(token_type)

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
        elif token_type == TokenType.FUN:
            return self.function_definition()
        elif token_type == TokenType.IF:
            return self.if_statement()
        elif token_type == TokenType.WHILE:
            return self.while_statement()
        elif token_type == TokenType.FOR:
            return self.for_statement()
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
        cases = []
        # Parse the initial 'if'
        self.eat(TokenType.IF)
        self.eat(TokenType.LPAREN)
        condition = self.comparison_expression()
        self.eat(TokenType.RPAREN)
        body = self.block()
        cases.append((condition, body))

        # Parse all 'elif' blocks
        while self.current_token[0] == TokenType.ELIF:
            self.eat(TokenType.ELIF)
            self.eat(TokenType.LPAREN)
            condition = self.comparison_expression()
            self.eat(TokenType.RPAREN)
            body = self.block()
            cases.append((condition, body))

        # Parse the final 'else' block, if it exists
        else_body = None
        if self.current_token[0] == TokenType.ELSE:
            self.eat(TokenType.ELSE)
            else_body = self.block()

        # Build the nested IfStatement node from the cases
        # Start from the last case and work backwards
        if else_body:
            node = else_body
        else:
            node = None
            
        for condition, body in reversed(cases):
            node = IfStatement(condition, body, node)
        
        return node
    
    def while_statement(self):
        self.eat(TokenType.WHILE)
        self.eat(TokenType.LPAREN)
        condition = self.comparison_expression()
        self.eat(TokenType.RPAREN)
        body = self.block()
        return WhileStatement(condition, body)
    
    def for_statement(self):
        self.eat(TokenType.FOR)
        variable_node = Variable(self.current_token)
        self.eat(TokenType.IDENT)
        self.eat(TokenType.IN)

        iterable_node = self.call_expression() # Assuming iterable is a function call, like range()

        body = self.block()
        return ForStatement(variable_node, iterable_node, body)

    def function_definition(self):
        self.eat(TokenType.FUN)
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
        token_type, token_value, line_num = self.current_token

        if token_type in (TokenType.PLUS, TokenType.MINUS):
            self.advance()
            return UnaryOp(token_value,self.factor())
        
        if token_type == TokenType.STRING:
            self.advance()
            return StringLiteral((token_type, token_value, line_num))

        if token_type in (TokenType.INTEGER, TokenType.FLOAT):
            self.advance()
            return Number((token_type, token_value,line_num))
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