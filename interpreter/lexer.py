# lexer.py

import re
from enum import Enum, auto

class TokenType(Enum):
    NUMBER = auto()
    IDENT = auto()
    ASSIGN = auto()
    PLUS = auto()
    MINUS = auto()
    MUL = auto()
    DIV = auto()
    LPAREN = auto()
    RPAREN = auto()
    PRINT = auto()
    EOF = auto()

token_spec = [
    (TokenType.PRINT, r'print\b'),
    (TokenType.IDENT, r'[a-zA-Z_]\w*'),
    (TokenType.NUMBER, r'\d+'),
    (TokenType.ASSIGN, r'='),
    (TokenType.PLUS, r'\+'),
    (TokenType.MINUS, r'-'),
    (TokenType.MUL, r'\*'),
    (TokenType.DIV, r'/'),
    (TokenType.LPAREN, r'\('),
    (TokenType.RPAREN, r'\)'),
    ("SKIP", r'[ \t\n]+'),
    ("MISMATCH", r'.'),
]

master_pattern = re.compile('|'.join(f'(?P<{tok.name}>{pat})' for tok, pat in token_spec if isinstance(tok, TokenType)))

def tokenize(code):
    tokens = []
    for mo in master_pattern.finditer(code):
        kind = mo.lastgroup
        value = mo.group()
        if kind == "SKIP":
            continue
        elif kind == "MISMATCH":
            raise SyntaxError(f"Unexpected character {value}")
        else:
            token_type = TokenType[kind]
            if token_type == TokenType.NUMBER:
                value = int(value)
            tokens.append((token_type, value))
    tokens.append((TokenType.EOF, None))
    return tokens
