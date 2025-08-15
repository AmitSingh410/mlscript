from ipykernel.kernelbase import Kernel
import io
from contextlib import redirect_stdout

from interpreter.interpreter import Interpreter
from interpreter.lexer import tokenize
from interpreter.parser import Parser

class MLScriptKernel(Kernel):
    implementation = 'mlscript'
    implementation_version = '0.8'
    language = 'mlscript'
    language_version = '0.8'
    language_info = {
        'name': 'mlscript',
        'mimetype': 'text/plain',
        'file_extension': '.ms',
    }
    banner = "mlscript Kernel"

    def do_execute(self, code, silent, store_history=True, user_expressions=None, allow_stdin=False):
        if not code.strip():
            return {'status': 'ok', 'execution_count': self.execution_count,
                    'payload': [], 'user_expressions': {}}

        interp = Interpreter()
        f = io.StringIO()

        try:
            with redirect_stdout(f):
                interp.run(code)
            
            output = f.getvalue()
            if not silent:
                stream_content = {'name': 'stdout', 'text': output}
                self.send_response(self.iopub_socket, 'stream', stream_content)
        
        except Exception as e:
            error_content = {'name': 'stderr', 'text': str(e)}
            self.send_response(self.iopub_socket, 'stream', error_content)
            return {'status': 'error', 'execution_count': self.execution_count,
                    'ename': type(e).__name__, 'evalue': str(e), 'traceback': []}

        return {'status': 'ok', 'execution_count': self.execution_count,
                'payload': [], 'user_expressions': {}}

if __name__ == '__main__':
    from ipykernel.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(kernel_class=MLScriptKernel)