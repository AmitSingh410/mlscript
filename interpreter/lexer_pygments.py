from pygments.lexer import RegexLexer, bygroups, words
from pygments.token import *

class MlscriptLexer(RegexLexer):
    """A Pygments lexer for the MLScript language."""

    name = 'MLScript'
    aliases = ['mlscript', 'ms']
    filenames = ['*.ms']
    mimetypes = ['text/x-mlscript']

    tokens = {
        'root': [
            (r'//.*', Comment.Single),
            (r'\n', Text),
            (r'\s+', Text),

            (words((
                'if', 'elif', 'else', 'while', 'for', 'in', 'with', 'try', 
                'catch', 'finally', 'throw', 'break', 'continue', 'fun', 
                'class', 'super', 'inherits', 'return', 'print', 'import', 'as',
                'network', 'not'), suffix=r'\b'), Keyword),
            
            (words(('true', 'false'), suffix=r'\b'), Keyword.Constant),
            (r'\d+\.\d+', Number.Float),
            (r'\d+', Number.Integer),
            (r'"(?:\\.|[^"\\])*"', String),

            (r'[a-zA-Z_]\w*', Name),

            (r'==|!=|>=|<=|>|<', Operator),
            (r'=', Operator),
            (r'[+\-*/]', Operator),
            (r'\(|\)|\[|\]|\{|\}', Punctuation),
            (r'[:,.]', Punctuation),
        ]
    }
