"""Launch GSF - Geology Source Finder. Double-click this file or run: python run.py"""

import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    result = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app.py"],
        check=True,
    )
except subprocess.CalledProcessError as exc:
    print(f"\n--- GSF exited with error code {exc.returncode} ---")
    input("Press Enter to close this window...")
except FileNotFoundError:
    print("\nError: Python or Streamlit not found.")
    print("Make sure you have installed the dependencies:")
    print(f"  {sys.executable} -m pip install -r requirements.txt")
    input("\nPress Enter to close this window...")
except KeyboardInterrupt:
    pass
except Exception as exc:
    print(f"\nUnexpected error: {exc}")
    input("Press Enter to close this window...")
