import importlib.util
import sys
import threading
import time
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "best.pt"


def require(module_name, pip_name=None):
    if importlib.util.find_spec(module_name) is None:
        name = pip_name or module_name
        raise RuntimeError(f"Missing dependency: {name}. Install it with: python -m pip install {name}")


def verify_runtime():
    for module, package in [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("ultralytics", "ultralytics"),
        ("supervision", "supervision"),
        ("multipart", "python-multipart"),
    ]:
        require(module, package)

    if not MODEL.exists():
        raise RuntimeError(f"Model not found: {MODEL}")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"GPU OK: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU not active: CUDA unavailable. SportVision will still run on CPU.")
    except Exception as exc:
        print(f"GPU check skipped: {exc}")


def open_browser_later(url):
    time.sleep(1.2)
    webbrowser.open(url)


def main():
    verify_runtime()
    import uvicorn

    url = "http://127.0.0.1:8000"
    threading.Thread(target=open_browser_later, args=(url,), daemon=True).start()
    print("SPORTVISION launching")
    print(f"Open URL: {url}")
    uvicorn.run("app.sportvision_server:app", host="127.0.0.1", port=8000, reload=False, app_dir=str(ROOT))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SPORTVISION failed to launch: {exc}")
        if sys.stdin.isatty():
            input("Press Enter to close...")
        sys.exit(1)
