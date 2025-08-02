from interpreter.interpreter import Interpreter

if __name__ == "__main__":
    interp = Interpreter()
    while True:
        try:
            line = input("mlscript> ")
            if line.strip() in ("exit","quit"): break
            interp.run(line)
        except Exception as e:
            print("Error:", e)