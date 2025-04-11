import sys
import os

def run_streamlit_app():
    # Get the path to your Streamlit app
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    
    # Try the new import structure first
    try:
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", app_path]
        sys.exit(stcli.main())
    except ImportError:
        # Fall back to old import structure
        try:
            from streamlit.cli import main
            sys.argv = ["streamlit", "run", app_path]
            sys.exit(main())
        except ImportError:
            print("Error: Streamlit is not installed or has an incompatible version.")
            print("Please install streamlit with: pip install streamlit==1.30.0")
            sys.exit(1)

if __name__ == "__main__":
    run_streamlit_app()