import uvicorn
import argparse

if __name__ == "__main__":
    argparse.ArgumentParser(description="Persistent Python execution API").parse_args()
    uvicorn.run("main:create_app", factory=True, host="0.0.0.0", port=8000, workers=1)
