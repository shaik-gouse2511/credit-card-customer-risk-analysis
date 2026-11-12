"""
Credit Card Customer Risk Analysis - Interactive Dashboard Server
==================================================================
Runs a local server and automatically launches the interactive web dashboard.
"""

import http.server
import socketserver
import webbrowser
import os
import sys
import threading
import time

PORT = 8501

def ensure_data():
    """Verify that dashboard data is compiled; if not, build it."""
    if not os.path.exists("outputs/dashboard_data.js"):
        print("[*] Generating dashboard data files...")
        import subprocess
        subprocess.run([sys.executable, "build_dashboard_data.py"], check=True)

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Keep console clean, only log errors or startup
        pass

def start_server():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ensure_data()

    handler = QuietHandler
    # Allow port reuse
    socketserver.TCPServer.allow_reuse_address = True
    
    current_port = PORT
    server = None
    for attempt in range(5):
        try:
            server = socketserver.TCPServer(("", current_port), handler)
            break
        except OSError:
            current_port += 1

    url = f"http://localhost:{current_port}/dashboard.html"
    
    print("=" * 70)
    print("  CREDIT CARD CUSTOMER RISK ANALYSIS - INTERACTIVE DASHBOARD")
    print("=" * 70)
    print(f"\n  >> Dashboard is running at: {url}")
    print("  >> Press Ctrl+C in this terminal to stop the server.\n")
    print("  Features Included:")
    print("  - [1] Executive Portfolio KPIs & Risk Tier Segmentation")
    print("  - [2] Customer Risk Explorer (Search, multi-filter & profile modal)")
    print("  - [3] Interactive Live 'What-If' Risk Calculator & Underwriting Advice")
    print("  - [4] Machine Learning Model Benchmarks (Random Forest Champion)")
    print("  - [5] Institutional Credit Policy Matrix & Academic Viva Defense Guide")
    print("=" * 70)

    # Open browser automatically after a short pause
    def open_browser():
        time.sleep(1.0)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")

if __name__ == "__main__":
    start_server()

