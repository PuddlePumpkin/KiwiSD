import subprocess
import os
os.chdir(str(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))))
while True:
    print("\nStarting kiwi process...\n")
    subprocess.run(["python", "src/kiwiSP.py"])