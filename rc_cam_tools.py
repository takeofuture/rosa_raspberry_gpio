# rc_cam_tools.py
import time
import cv2
from langchain_core.tools import tool

@tool
def capture_image(dev: int = 0, width: int = 1280, height: int = 720, save_path: str = "/tmp/cap.jpg") -> dict:
    """
    Capture one frame from /dev/video<dev> and save it.
    IMPORTANT: Do NOT return base64 to LLM (keeps context small).
    """
    cap = cv2.VideoCapture(int(dev), cv2.CAP_V4L2)
    if not cap.isOpened():
        return {"ok": False, "error": f"VideoCapture open failed: dev={dev}"}

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return {"ok": False, "error": f"Frame read failed: dev={dev}", "size": [width, height]}

    ok_save = cv2.imwrite(save_path, frame)
    if not ok_save:
        return {"ok": False, "error": "cv2.imwrite failed", "path": save_path}

    h, w = frame.shape[:2]
    return {
        "ok": True,
        "dev": dev,
        "size": [w, h],
        "path": save_path,
        "ts": time.time(),
    }
