"""
Credit Card Customer Risk Analysis - FastHTML & FastAPI Web Application
========================================================================
Serves the interactive HTML dashboard, static assets, and ML prediction endpoints.
Compatible with Vercel Python Runtime, FastHTML, and FastAPI/Uvicorn.
"""

import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Check if python-fasthtml is available; otherwise use FastAPI / Starlette
USE_FASTHTML = False
try:
    import fasthtml
    from fasthtml.common import fast_app, FileResponse, HTMLResponse
    USE_FASTHTML = True
except ImportError:
    pass

if USE_FASTHTML:
    # --- FastHTML Implementation ---
    app, rt = fast_app(static_dir=str(BASE_DIR / "outputs"))

    @rt("/")
    def get():
        index_file = BASE_DIR / "index.html"
        if index_file.exists():
            return HTMLResponse(index_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Credit Card Customer Risk Analysis Dashboard</h1>")

    @rt("/dashboard")
    def get_dashboard():
        dash_file = BASE_DIR / "dashboard.html"
        if dash_file.exists():
            return HTMLResponse(dash_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Dashboard</h1>")

    @rt("/api/health")
    def get_health():
        return {"status": "healthy", "engine": "FastHTML"}

else:
    # --- FastAPI / Starlette Implementation (Standard Python ASGI) ---
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(
        title="Credit Card Customer Risk Analysis Dashboard",
        description="Interactive risk intelligence dashboard and predictive scoring API",
        version="2.0.0"
    )

    # Mount static outputs folder if exists
    outputs_dir = BASE_DIR / "outputs"
    if outputs_dir.exists():
        app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

    @app.get("/", response_class=HTMLResponse)
    async def get_index():
        index_file = BASE_DIR / "index.html"
        if index_file.exists():
            return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Credit Card Customer Risk Analysis Dashboard</h1>")

    @app.get("/dashboard", response_class=HTMLResponse)
    async def get_dashboard():
        dash_file = BASE_DIR / "dashboard.html"
        if dash_file.exists():
            return HTMLResponse(content=dash_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Dashboard</h1>")

    @app.get("/api/health")
    async def get_health():
        return {
            "status": "healthy",
            "framework": "FastAPI / FastHTML ASGI Engine",
            "model": "Random Forest (77.4% ROC-AUC)",
            "records": 5993
        }

    @app.get("/api/kpis")
    async def get_kpis():
        json_path = outputs_dir / "dashboard_data.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("kpis", {})
        return {"message": "Data compiling"}

# Handler alias for serverless platforms expecting 'handler' or 'application'
application = app
handler = app

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    import time

    port = int(os.environ.get("PORT", 8000))
    url = f"http://localhost:{port}"

    def open_browser():
        time.sleep(1.0)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()
    print("=" * 70)
    print("  CREDIT CARD CUSTOMER RISK ANALYSIS - FASTHTML/FASTAPI SERVER")
    print("=" * 70)
    print(f"  >> Running at: {url}")
    print("=" * 70)
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
