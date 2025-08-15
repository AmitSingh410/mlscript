import sys
import json
import os 
import shutil
from jupyter_client.kernelspec import KernelSpecManager
from IPython.utils.tempdir import TemporaryDirectory

kernel_json = {
    "argv" : [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "MlScript",
    "language": "mlscript",
}

def install_my_kernel_spec(user=True):
    with TemporaryDirectory() as td:
        os.chmod(td, 0o755)
        with open(os.path.join(td, 'kernel.json'), 'w') as f:
            json.dump(kernel_json, f,sort_keys=True)

        print('Installing Jupyter kernel spec')
        KernelSpecManager().install_kernel_spec(td, 'mlscript', user=user, replace =True)

def _is_root():
    try:
        return os.getuid() == 0
    except AttributeError:
        return False
    
def main(argv=None):
    if _is_root():
        print("Running as root, installing for all users.")
        install_my_kernel_spec(user=False)
    else:
        print("Running as non-root, installing for current user.")
        install_my_kernel_spec(user=True)
    
    print("mlscript kernel successfully installed.")

if __name__ == "__main__":
    main()