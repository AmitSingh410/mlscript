import sys
import mlscript
from interpreter.interpreter import Interpreter, ReturnSignal

def run_from_file(filepath):
    interp = Interpreter()
    try:
        with open(filepath, 'r') as f:
            code = f.read()
            interp.run(code)
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
    except Exception as e:
        print(f"An error occurred while executing the file: {e}")

def start_repl():
    interp = Interpreter()
    print("mlscript v0.2 -- interactive REPL")
    print("Type 'quit' or 'exit' to leave.")

    buffer = ""
    prompt = "mlscript> "

    while True:
        try:
            line = input(prompt)

            if line.strip().lower() in ("quit", "exit"):
                break

            buffer += line

            if buffer.count('{') > buffer.count('}'):
                prompt = "...       "
                buffer += "\n"
                continue
            
            if not buffer.strip():
                buffer = ""
                continue
            
            interp.run(buffer)

            buffer = ""
            prompt = "mlscript> "

        except ReturnSignal:
             print("SyntaxError: 'return' can only be used inside a function.")
             buffer = ""
             prompt = "mlscript> "
        except Exception as e:
            print(f"Error: {e}")
            buffer = ""
            prompt = "mlscript> "

def main():
    # sys.argv is a list of command-line arguments.
    # sys.argv[0] is the script name itself (main.py)
    # If there is another argument, it's the file to run.
    if len(sys.argv) > 1:
        run_from_file(sys.argv[1])
    else:
        start_repl()

if __name__ == "__main__":
    main()