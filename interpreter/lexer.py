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
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    PRINT = auto()
    DEF = auto()
    RETURN = auto()
    IF = auto()
    ELSE = auto()
    EQ = auto()
    NE = auto()
    GT = auto()
    GTE = auto()
    LT = auto()
    LTE = auto()
    EOF = auto()

token_spec = [
    (TokenType.IF,      r'if\b'),
    (TokenType.ELSE,    r'else\b'),
    (TokenType.DEF,     r'def\b'),
    (TokenType.RETURN,  r'return\b'),
    (TokenType.PRINT,   r'print\b'),
    (TokenType.FLOAT,   r'\d+\.\d+'),
    (TokenType.INTEGER, r'\d+'),
    (TokenType.IDENT,   r'[a-zA-Z_]\w*'),
    (TokenType.EQ,      r'=='),
    (TokenType.NE,      r'!='),
    (TokenType.GTE,     r'>='),
    (TokenType.LTE,     r'<='),
    (TokenType.GT,      r'>'),
    (TokenType.LT,      r'<'),
    (TokenType.ASSIGN,  r'='),
    (TokenType.PLUS,    r'\+'),
    (TokenType.MINUS,   r'-'),
    (TokenType.MUL,     r'\*'),
    (TokenType.DIV,     r'/'),
    (TokenType.LPAREN,  r'\('),
    (TokenType.RPAREN,  r'\)'),
    (TokenType.LBRACE,  r'\{'),
    (TokenType.RBRACE,  r'\}'),
    (TokenType.COMMA,   r','),
    ("SKIP",            r'[ \t\n]+'),
    ("MISMATCH",        r'.'),
]

token_regex = '|'.join(f'(?P<{spec[0].name}>{spec[1]})' for spec in token_spec if spec[0] != "SKIP" and spec[0] != "MISMATCH")

def tokenize(code):
    tokens = []
    line_num = 1
    line_start = 0
    for mo in re.finditer(token_regex, code):
        kind_str = mo.lastgroup
        value = mo.group()

        kind = TokenType[kind_str]

        if kind == TokenType.FLOAT:
            value = float(value)
        elif kind == TokenType.INTEGER:
            value = int(value)

        tokens.append((kind, value))

    tokens.append((TokenType.EOF, None))
    return tokens