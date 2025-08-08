import re
from enum import Enum, auto

class TokenType(Enum):
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
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
    FUN = auto()
    RETURN = auto()
    IF = auto()
    ELIF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    IN = auto()
    EQ = auto()
    NE = auto()
    GT = auto()
    GTE = auto()
    LT = auto()
    LTE = auto()
    EOF = auto()

token_spec = [
    ("COMMENT",       r'//.*'),
    ("NEWLINE",        r'\n'),
    ("SKIP",            r'[ \t]+'),
    (TokenType.IF,      r'if\b'),
    (TokenType.ELIF,    r'elif\b'),
    (TokenType.ELSE,    r'else\b'),
    (TokenType.WHILE,   r'while\b'),
    (TokenType.FOR,     r'for\b'),
    (TokenType.IN,      r'in\b'),
    (TokenType.FUN,     r'fun\b'),
    (TokenType.RETURN,  r'return\b'),
    (TokenType.PRINT,   r'print\b'),
    (TokenType.FLOAT,   r'\d+\.\d+'),
    (TokenType.INTEGER, r'\d+'),
    (TokenType.STRING,  r'"[^"]*"'),
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
    
    ("MISMATCH",        r'.'),
]

token_regex = '|'.join(f'(?P<{spec[0].name if isinstance(spec[0], Enum) else spec[0]}>{spec[1]})' for spec in token_spec)
 
def tokenize(code):
    tokens = []
    line_num = 1

    for mo in re.finditer(token_regex, code):
        kind_str = mo.lastgroup
        value = mo.group()

        if kind_str == "COMMENT":
            continue
        if kind_str == "NEWLINE":
            line_num += 1
            continue
        if kind_str == "SKIP" or kind_str == "MISMATCH":
            continue

        kind = TokenType[kind_str]

        if kind == TokenType.FLOAT:
            value = float(value)
        elif kind == TokenType.INTEGER:
            value = int(value)
        elif kind == TokenType.STRING:
            value = value[1:-1]

        tokens.append((kind, value, line_num))

    tokens.append((TokenType.EOF, None,line_num))
    return tokens