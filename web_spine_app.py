# Streamlit Cloud entry point for the existing deployment.
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("web_app.py")), run_name="__main__")
