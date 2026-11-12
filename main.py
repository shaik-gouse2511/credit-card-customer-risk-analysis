"""
Main entrypoint to run the Streamlit Credit Card Customer Risk Analysis Dashboard.
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    app_path = os.path.join(current_dir, "app.py")
    print("=" * 70)
    print("  LAUNCHING STREAMLIT CREDIT CARD CUSTOMER RISK ANALYSIS DASHBOARD")
    print("=" * 70)
    print(f"  Target: {app_path}\n")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
    except KeyboardInterrupt:
        print("\n[*] Streamlit app stopped.")
