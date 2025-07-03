import subprocess
import sys

commands = [
    "pip install numpy==1.26.4",
    "pip install torch==2.3.1+cu121 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121",
    "pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html",
    "pip install torch-sparse -f https://data.pyg.org/whl/torch-2.3.1+cu121.html",
    "pip install torch-cluster -f https://data.pyg.org/whl/torch-2.3.1+cu121.html",
    "pip install torch-geometric"
]

def install_packages():
    for command in commands:
        print(f"\nExecuting: {command}\n" + "-" * 50)
        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            print(f"\nError occurred while executing: {command}\nExiting.")
            sys.exit(1)

if __name__ == "__main__":
    install_packages()
    print("\nAll packages installed successfully! 🚀")
