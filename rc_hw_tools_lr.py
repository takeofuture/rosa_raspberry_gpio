# rc_hw_tools_lr.py
# ROSA/LangChain toolset for RC car on Raspberry Pi
# - drive_lr(left_power, right_power, duration_s, sample_encoder)
# - forward(duration_s, power)
# - backward(duration_s, power)
# - turn_left(duration_s, power)   # right forward, left backward
# - turn_right(duration_s, power)  # left forward, right backward
# - drive_wheel(side, power, duration_s)
# - stop_all()
# - get_encoder_counts()

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Literal

from gpiozero import PWMOutputDevice, Button
from langchain_core.tools import tool

# -------------------------
# GPIO pin assignment (EDIT HERE)
# -------------------------
# Right motor (H-bridge)
MOT_R_1 = PWMOutputDevice(pin=22, frequency=60)  # forward
MOT_R_2 = PWMOutputDevice(pin=23, frequency=60)  # backward

# Left motor (H-bridge)
MOT_L_1 = PWMOutputDevice(pin=18, frequency=60)  # forward
MOT_L_2 = PWMOutputDevice(pin=17, frequency=60)  # backward

# Encoders (EDIT HERE)
ENC_R = Button(pin=10, pull_up=True, bounce_time=0.0)
ENC_L = Button(pin=2,  pull_up=True, bounce_time=0.0)
# -------------------------
# Safety / limits
# -------------------------
MAX_POWER = 0.60        # clamp |power| <= 0.60 (tune)
MAX_DURATION = 10.0     # seconds (tune)
SAMPLE_PERIOD = 1.0     # encoder sampling period (seconds)

# For wrappers default
DEFAULT_POWER = 0.40


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _stop_all() -> None:
    for m in (MOT_R_1, MOT_R_2, MOT_L_1, MOT_L_2):
        m.value = 0.0


def _apply_signed_power(m_fwd: PWMOutputDevice, m_bwd: PWMOutputDevice, signed_power: float) -> None:
    """
    signed_power: -1..+1
      + : forward  -> (fwd=p, bwd=0)
      - : backward -> (fwd=0, bwd=p)
    """
    p = _clamp(abs(float(signed_power)), 0.0, MAX_POWER)
    if signed_power >= 0:
        m_fwd.value = p
        m_bwd.value = 0.0
    else:
        m_fwd.value = 0.0
        m_bwd.value = p


class EncoderCounter:
    def __init__(self, enc_button: Button):
        self._enc = enc_button
        self._lock = threading.Lock()
        self._count = 0
        self._enc.when_pressed = self._on_pulse

    def _on_pulse(self):
        with self._lock:
            self._count += 1

    def get(self) -> int:
        with self._lock:
            return int(self._count)

    def reset(self):
        with self._lock:
            self._count = 0


ENCODER_RIGHT = EncoderCounter(ENC_R)
ENCODER_LEFT = EncoderCounter(ENC_L)


@dataclass
class Sample:
    t_s: float
    left_count: int
    left_delta: int
    right_count: int
    right_delta: int


def _sample_both(duration_s: float) -> List[Dict[str, Any]]:
    start = time.time()
    prev_l = ENCODER_LEFT.get()
    prev_r = ENCODER_RIGHT.get()

    samples: List[Sample] = []
    samples.append(
        Sample(
            t_s=0.0,
            left_count=prev_l,
            left_delta=0,
            right_count=prev_r,
            right_delta=0,
        )
    )

    next_t = start + SAMPLE_PERIOD
    while True:
        now = time.time()
        if now - start >= duration_s:
            break

        time.sleep(max(0.0, next_t - now))
        now2 = time.time()

        cl = ENCODER_LEFT.get()
        cr = ENCODER_RIGHT.get()

        dl = cl - prev_l
        dr = cr - prev_r
        prev_l, prev_r = cl, cr

        samples.append(
            Sample(
                t_s=now2 - start,
                left_count=cl,
                left_delta=dl,
                right_count=cr,
                right_delta=dr,
            )
        )
        next_t += SAMPLE_PERIOD

    return [asdict(s) for s in samples]


# ============================================================
# ROSA-callable tools
# ============================================================

@tool
def stop_all() -> str:
    """Stop all motors immediately."""
    _stop_all()
    return "Stopped all motors (set all PWM to 0)."


@tool
def get_encoder_counts() -> Dict[str, Any]:
    """Return current encoder counts (no reset)."""
    return {
        "ok": True,
        "left_count": ENCODER_LEFT.get(),
        "right_count": ENCODER_RIGHT.get(),
    }


@tool
def drive_lr(left_power: float, right_power: float, duration_s: float, sample_encoder: bool = True) -> Dict[str, Any]:
    """
    Drive left and right wheels simultaneously.

    left_power/right_power: -1.0..+1.0 (negative=backward, positive=forward)
    duration_s: seconds

    Safety:
      - power is clamped to +/- MAX_POWER
      - duration is clamped to MAX_DURATION
      - always stops at the end

    Returns:
      - encoder 1Hz samples (counts + deltas) while driving
    """
    lp = _clamp(float(left_power), -MAX_POWER, MAX_POWER)
    rp = _clamp(float(right_power), -MAX_POWER, MAX_POWER)
    dur = _clamp(float(duration_s), 0.0, MAX_DURATION)

    # safety: start from stop
    _stop_all()

    holder: Dict[str, Any] = {"samples": None, "error": None}
    th: Optional[threading.Thread] = None

    if sample_encoder:
        def _worker():
            try:
                holder["samples"] = _sample_both(dur)
            except Exception as e:
                holder["error"] = str(e)

        th = threading.Thread(target=_worker, daemon=True)
        th.start()

    try:
        _apply_signed_power(MOT_L_1, MOT_L_2, lp)
        _apply_signed_power(MOT_R_1, MOT_R_2, rp)
        time.sleep(dur)
    finally:
        _stop_all()

    if th is not None:
        th.join(timeout=2.0)

    return {
        "ok": True,
        "command": {"left_power": lp, "right_power": rp, "duration_s": dur},
        "encoder": {
            "enabled": sample_encoder,
            "sample_period_s": SAMPLE_PERIOD,
            "samples": holder["samples"],
            "error": holder["error"],
        },
    }


# -------------------------
# Human-friendly wrappers (better tool selection by LLM)
# -------------------------

@tool
def forward(duration_s: float, power: float = DEFAULT_POWER, sample_encoder: bool = True) -> Dict[str, Any]:
    """Move forward for duration_s seconds."""
    p = _clamp(float(power), 0.0, MAX_POWER)
    return drive_lr.invoke({"left_power": p, "right_power": p, "duration_s": duration_s, "sample_encoder": sample_encoder})


@tool
def backward(duration_s: float, power: float = DEFAULT_POWER, sample_encoder: bool = True) -> Dict[str, Any]:
    """Move backward for duration_s seconds."""
    p = _clamp(float(power), 0.0, MAX_POWER)
    return drive_lr.invoke({"left_power": -p, "right_power": -p, "duration_s": duration_s, "sample_encoder": sample_encoder})


@tool
def turn_left(duration_s: float, power: float = DEFAULT_POWER, sample_encoder: bool = True) -> Dict[str, Any]:
    """
    Turn left in place for duration_s seconds.
    Implementation: right forward, left backward.
    """
    p = _clamp(float(power), 0.0, MAX_POWER)
    return drive_lr.invoke({"left_power": -p, "right_power": p, "duration_s": duration_s, "sample_encoder": sample_encoder})


@tool
def turn_right(duration_s: float, power: float = DEFAULT_POWER, sample_encoder: bool = True) -> Dict[str, Any]:
    """
    Turn right in place for duration_s seconds.
    Implementation: left forward, right backward.
    """
    p = _clamp(float(power), 0.0, MAX_POWER)
    return drive_lr.invoke({"left_power": p, "right_power": -p, "duration_s": duration_s, "sample_encoder": sample_encoder})


@tool
def drive_wheel(side: str, power: float, duration_s: float, sample_encoder: bool = True) -> Dict[str, Any]:
    """
    Drive a single wheel (left or right) for duration_s seconds.
    side: 'left' or 'right'
    power: -1.0..+1.0 (negative=backward, positive=forward)

    Useful for commands like:
      - "左タイヤだけ前進"
      - "右タイヤを0.3で2秒回して"
    """
    s = (side or "").strip().lower()
    p = _clamp(float(power), -MAX_POWER, MAX_POWER)
    dur = _clamp(float(duration_s), 0.0, MAX_DURATION)

    _stop_all()

    holder: Dict[str, Any] = {"samples": None, "error": None}
    th: Optional[threading.Thread] = None
    if sample_encoder:
        def _worker():
            try:
                holder["samples"] = _sample_both(dur)
            except Exception as e:
                holder["error"] = str(e)
        th = threading.Thread(target=_worker, daemon=True)
        th.start()

    try:
        if s == "left":
            _apply_signed_power(MOT_L_1, MOT_L_2, p)
            _apply_signed_power(MOT_R_1, MOT_R_2, 0.0)
        elif s == "right":
            _apply_signed_power(MOT_L_1, MOT_L_2, 0.0)
            _apply_signed_power(MOT_R_1, MOT_R_2, p)
        else:
            _stop_all()
            return {"ok": False, "error": "side must be 'left' or 'right'."}

        time.sleep(dur)
    finally:
        _stop_all()

    if th is not None:
        th.join(timeout=2.0)

    return {
        "ok": True,
        "command": {"side": s, "power": p, "duration_s": dur},
        "encoder": {
            "enabled": sample_encoder,
            "sample_period_s": SAMPLE_PERIOD,
            "samples": holder["samples"],
            "error": holder["error"],
        },
    }

