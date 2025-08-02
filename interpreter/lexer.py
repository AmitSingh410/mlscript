import re
from enum import Enum, auto

class TokenType(Enum):
    INTEGER = auto()
    FLOAT = auto()
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
    (TokenType.PRINT,   r'print\b'),
    (TokenType.FLOAT,   r'\d+\.\d*'),      # Rule for floats
    (TokenType.INTEGER, r'\d+'),           # Rule for integers
    (TokenType.IDENT,   r'[a-zA-Z_]\w*'),
    (TokenType.ASSIGN,  r'='),
    (TokenType.PLUS,    r'\+'),
    (TokenType.MINUS,   r'-'),
    (TokenType.MUL,     r'\*'),
    (TokenType.DIV,     r'/'),
    (TokenType.LPAREN,  r'\('),
    (TokenType.RPAREN,  r'\)'),
    ("SKIP",            r'[ \t\n]+'),
    ("MISMATCH",        r'.'),
]

master_pattern = re.compile('|'.join(f'(?P<{tok.name}>{pat})' for tok, pat in token_spec if isinstance(tok, TokenType)))

def tokenize(code):
    tokens = []
    for mo in master_pattern.finditer(code):
        kind = TokenType[mo.lastgroup]
        value = mo.group()
        if kind == TokenType.FLOAT:
            value = float(value)
        elif kind == TokenType.INTEGER:
            value = int(value)
        tokens.append((kind, value))
    tokens.append((TokenType.EOF, None))
    return tokens