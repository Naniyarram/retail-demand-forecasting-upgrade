"""
run_api.py

Simple API entry point for local demos.

This starts a web server and keeps running untilyou stop it C.
"""

import uvicorn


def main():
    """
    Start the forecasting API.
    """

    print("Starting Retail Demand Forecasting API at ""http://127.0.0.1:8000")
    print("Press Ctrl+C to stop the server." )

    uvicorn.run("pipeline.api.app:app",host="127.0.0.1", port=8000,reload=False )


if __name__ == "__main__":
    main()
