"""Launch GSF - Geology Source Finder. Double-click this file or run: python run.py"""

import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=False)
