# app_rccam_rosa.py
# Streamlit app: RC ROSA (GPIO) + USB camera on Raspberry Pi
#
# IMPORTANT:
# - Do NOT send base64 image data to LLM (context explosion).
# - capture_image tool should return only small dict (ok/path/size/ts), NOT image_b64.
# - Streamlit displays image by reading the saved file path.
#
# Run:
#   streamlit run app_rccam_rosa.py --server.address 0.0.0.0 --server.port 8501

import streamlit as st
import json
import re
from config import OPENAI_API_KEY,LLM_PROVIDER,LLM_BASE_URL,LLM_MODEL,LLM_API_KEY
from langchain_openai import ChatOpenAI
from rosa import ROSA, RobotSystemPrompts

# GPIO tools
from rc_hw_tools_lr import (
    drive_lr,
    forward,
    backward,
    turn_left,
    turn_right,
    drive_wheel,
    stop_all,
    get_encoder_counts,
)

# Camera tool (MUST NOT return base64)
from rc_cam_tools import capture_image


# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="RC ROSA + Camera (Pi)", layout="wide")


# -------------------------
# Session state (NO history)
# -------------------------
defaults = {
    "last_request": "",
    "last_answer": None,          # dict or str
    "last_image_path": "",        # latest captured image path
    "camera_dev": 0,
    "camera_w": 1280,
    "camera_h": 720,
    "camera_save_path": "/tmp/cap_streamlit.jpg",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# -------------------------
# Build ROSA once
# -------------------------
@st.cache_resource
def build_rosa():
    #llm = ChatOpenAI(
    #    model="gpt-4o",
    #    temperature=0,
    #    openai_api_key=OPENAI_API_KEY,
    #)
    if LLM_PROVIDER == "vllm":
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            openai_api_key="DUMMY",  # vLLMはダミーでOK
            base_url=LLM_BASE_URL,                # 例: http://<EC2-IP>:8000/v1
        )
    else:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
        )
    prompts = RobotSystemPrompts(
        embodiment_and_persona="You are an RC car controlled via Raspberry Pi GPIO with a USB camera.",
        about_your_capabilities=(
            "You can drive (forward/backward/turn/drive_wheel), stop, read encoder counts, "
            "and capture an image using capture_image tool. "
            "When user requests a photo, call capture_image."
        ),
        constraints_and_guardrails=(
                "If you will use a tool, output ONLY ONE JSON object on a single line.\n"
                "Do NOT use markdown. Do NOT add explanations.\n"
                "Format: {{\"name\":\"TOOL_NAME\",\"arguments\":{{...}}}}\n"
                "If no tool is needed, reply in Japanese briefly.\n"
                "Always prefer calling a tool for motion commands.\n"
            #"Safety rules:\n"
            #"1) Never exceed power and duration limits.\n"
            #"2) Always stop motors at the end of every action.\n"
            #"3) If user requests long motion, split it.\n"
            #"4) Prefer high-level motion tools.\n"
            #"5) If user asks for photo/camera, call capture_image.\n"
            #"6) Do NOT output huge data; keep replies concise.\n"
        ),
    )

    return ROSA(
        ros_version=2,  # 区分用（実際はGPIO直）
        llm=llm,
        prompts=prompts,
        tools=[
            drive_lr,
            forward,
            backward,
            turn_left,
            turn_right,
            drive_wheel,
            stop_all,
            get_encoder_counts,
            capture_image,
        ],
    )


rosa = build_rosa()


# -------------------------
# Helpers
# -------------------------
# -------------------------
# Tool execution fallback (A)
# Qwenが tool_calls を返さず、テキストに {"name":...} を書いてくる場合に実行する
# -------------------------
TOOL_MAP = {
    "drive_lr": drive_lr,
    "forward": forward,
    "backward": backward,
    "turn_left": turn_left,
    "turn_right": turn_right,
    "drive_wheel": drive_wheel,
    "stop_all": stop_all,
    "get_encoder_counts": get_encoder_counts,
    "capture_image": capture_image,
}

def _extract_first_json_object(text: str) -> dict | None:
    """テキスト中の最初の { ... } をJSONとして読み取る。なければNone。"""
    if not isinstance(text, str):
        return None
    s = text.strip()

    # ```json ... ``` を剥がす
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    # 最初に見つかった {...} を拾う（改行含む）
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None

    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def try_execute_tool_from_llm_text(ans_text: str) -> dict | str | None:
    """
    LLMが返したテキストから {"name": "...", "arguments": {...}} を抽出してツール実行。
    成功したら tool result を返す。失敗なら None。
    """
    obj = _extract_first_json_object(ans_text)
    if not obj:
        return None

    name = obj.get("name")
    args = obj.get("arguments", {})

    if not name or name not in TOOL_MAP:
        return {"ok": False, "error": f"Unknown tool name in LLM JSON: {name}", "raw": obj}

    try:
        # LangChain tool: invoke(dict)
        return TOOL_MAP[name].invoke(args if isinstance(args, dict) else {})
    except Exception as e:
        return {"ok": False, "error": f"Tool execution failed: {e}", "raw": obj}


PHOTO_KEYWORDS = ["写真", "撮影", "カメラ", "snapshot", "picture", "photo", "capture", "撮って", "撮れ"]

def is_photo_command(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(k.lower() in t for k in PHOTO_KEYWORDS)


def do_capture_via_tool(reason: str = ""):
    """Capture image by calling capture_image tool directly (no LLM)."""
    dev = int(st.session_state.camera_dev)
    w = int(st.session_state.camera_w)
    h = int(st.session_state.camera_h)
    save_path = str(st.session_state.camera_save_path)

    res = capture_image.invoke({"dev": dev, "width": w, "height": h, "save_path": save_path})
    st.session_state.last_request = reason or "CAPTURE (direct tool)"
    st.session_state.last_answer = res
    if isinstance(res, dict) and res.get("ok") and res.get("path"):
        st.session_state.last_image_path = res["path"]


# -------------------------
# UI
# -------------------------
st.title("RC ROSA + Camera (Raspberry Pi)")

left, right = st.columns([2, 3], gap="small")

with left:
    st.subheader("Camera")
    st.caption("⚠️ Pi側でフレームを保存")

    # Camera settings (persist)
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.camera_dev = st.number_input("device index (/dev/videoN)", 0, 99, int(st.session_state.camera_dev))
        st.session_state.camera_save_path = st.text_input("save path", st.session_state.camera_save_path)
    with c2:
        st.session_state.camera_w = st.selectbox("width", [640, 1280, 1920], index=[640, 1280, 1920].index(int(st.session_state.camera_w)))
        st.session_state.camera_h = st.selectbox("height", [480, 720, 1080], index=[480, 720, 1080].index(int(st.session_state.camera_h)))

    if st.button("Capture Now (direct tool)", use_container_width=True):
        do_capture_via_tool("CAPTURE (direct tool)")
        st.rerun()

    st.divider()

    st.subheader("Command")
    st.caption("ロボカーへ指令を！")

    with st.form("cmdform", clear_on_submit=True):
        user_text = st.text_input(
            "例：前進して / バックして / 左回転して / 左のタイヤを後ろに回して / 写真撮って",
            value="",
        )
        submitted = st.form_submit_button("Send")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Stop (Emergency)", use_container_width=True):
            try:
                res = stop_all.invoke({})
                st.session_state.last_request = "STOP (Emergency)"
                st.session_state.last_answer = res
            except Exception as e:
                st.session_state.last_request = "STOP (Emergency)"
                st.session_state.last_answer = {"ok": False, "error": str(e)}
            st.rerun()

    with b2:
        if st.button("Clear", use_container_width=True):
            st.session_state.last_request = ""
            st.session_state.last_answer = None
            st.session_state.last_image_path = ""
            st.rerun()

with right:
    st.subheader("Last Execution (overwritten each time)")

    # Execute on submit
    if submitted and user_text.strip():
        cmd = user_text.strip()

        # Rule-based: photo command => direct tool call (NO LLM)
        if is_photo_command(cmd):
            do_capture_via_tool(reason=cmd)
        else:
            st.session_state.last_request = cmd
            try:
                ans = rosa.invoke(cmd)
                #st.session_state.last_answer = ans
                # 1) もしROSAが既に tool を実行できていれば、dictが返ることが多い（そのまま採用）
                # 2) もし文字列なら、Qwenが書いた {"name":...} を拾ってツール実行して結果を採用
                if isinstance(ans, str):
                    tool_res = try_execute_tool_from_llm_text(ans)
                    if tool_res is not None:
                        ans = tool_res
                st.session_state.last_answer = ans

                # If LLM chose capture_image, we still only get small dict with path
                if isinstance(ans, dict) and ans.get("ok") and ans.get("path"):
                    st.session_state.last_image_path = ans["path"]

            except Exception as e:
                st.session_state.last_answer = {"ok": False, "error": str(e)}

    # Display last request/answer
    if not st.session_state.last_request:
        st.caption("No command sent yet.")
    else:
        st.write("**Request:**", st.session_state.last_request)
        st.write("**ROSA Answer / Tool Result:**")
        st.code(str(st.session_state.last_answer))

    # Display last captured image (by file path)
    if st.session_state.last_image_path:
        try:
            st.image(
                st.session_state.last_image_path,
                caption=f"Captured: {st.session_state.last_image_path}",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Image display failed: {e}")
    else:
        st.info("No image yet. Click 'Capture Now' or type '写真撮って'.")

    with st.expander("Encoder (optional)"):
        try:
            enc = get_encoder_counts.invoke({})
            st.code(str(enc), language="json")
        except Exception as e:
            st.write(f"Encoder error: {e}")

st.divider()
st.caption("Note: This app shows only the last request/result (no chat history).")
