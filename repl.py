from lexer import tokenize
from parser import Parser
from interpreter import Interpreter

def repl():
    interp= Interpreter()
    while True:
        try:
            line=input("mlscript> ")
            tokens = tokenize(line)
            parser = Parser(tokens)
            stmts = parser.parse()
            for stmt in stmts:
                interp.eval(stmt)
        except Exception as e:
            print(f"Error: {e}")