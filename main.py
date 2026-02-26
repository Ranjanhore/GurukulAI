import sys

@app.get("/")
def root():
    return {
        "ok": True,
        "python_version": sys.version
    }
