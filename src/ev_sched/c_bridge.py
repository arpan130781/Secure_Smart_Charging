"""ctypes Python Bridge to the Embedded C99 Core Library (libseccharge).

Provides zero-overhead direct invocation of the production-grade ANSI C
algorithms from Python.
"""

import sys
from ctypes import (
    CDLL,
    POINTER,
    Structure,
    byref,
    c_bool,
    c_double,
    c_int,
    c_size_t,
    c_uint32,
)
from pathlib import Path

# Locate compiled library in core_c/build
_LIB_NAME = "libseccharge.dylib" if sys.platform == "darwin" else "libseccharge.so"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_LIB_PATH = _REPO_ROOT / "core_c" / "build" / _LIB_NAME

_LIB: CDLL | None
try:
    _LIB = CDLL(str(_LIB_PATH))
    LIB_AVAILABLE = True
except OSError:
    _LIB = None
    LIB_AVAILABLE = False


class CSecDetectorAlert(Structure):
    _fields_ = [
        ("is_alarm", c_bool),
        ("residual_kw", c_double),
        ("test_statistic", c_double),
    ]


class CSecCUSUM(Structure):
    _fields_ = [
        ("threshold_kw", c_double),
        ("slack_kw", c_double),
        ("instant_threshold_kw", c_double),
        ("statistic", c_double),
    ]


class CSecKalmanChiSq(Structure):
    _fields_ = [
        ("x_hat", c_double),
        ("P_cov", c_double),
        ("Q_process_noise", c_double),
        ("R_meas_noise", c_double),
        ("chisq_threshold", c_double),
        ("ewma_chisq", c_double),
        ("ewma_alpha", c_double),
        ("initialized", c_bool),
    ]


class CSecEVSession(Structure):
    _fields_ = [
        ("session_id", c_uint32),
        ("max_rate_kw", c_double),
        ("energy_requested_kwh", c_double),
        ("energy_delivered_kwh", c_double),
        ("remaining_energy_kwh", c_double),
        ("dwell_hours", c_double),
        ("allocated_power_kw", c_double),
    ]


class CSecFeederState(Structure):
    _fields_ = [
        ("transformer_limit_kw", c_double),
        ("base_load_kw", c_double),
        ("safety_margin_kw", c_double),
        ("time_step_hours", c_double),
    ]


if LIB_AVAILABLE and _LIB is not None:
    # sec_cusum_init
    _LIB.sec_cusum_init.argtypes = [POINTER(CSecCUSUM), c_double, c_double, c_double]
    _LIB.sec_cusum_init.restype = c_int

    # sec_cusum_reset
    _LIB.sec_cusum_reset.argtypes = [POINTER(CSecCUSUM)]
    _LIB.sec_cusum_reset.restype = c_int

    # sec_cusum_update
    _LIB.sec_cusum_update.argtypes = [POINTER(CSecCUSUM), c_double, c_double, POINTER(CSecDetectorAlert)]
    _LIB.sec_cusum_update.restype = c_int

    # sec_kalman_chisq_init
    _LIB.sec_kalman_chisq_init.argtypes = [POINTER(CSecKalmanChiSq), c_double, c_double, c_double, c_double]
    _LIB.sec_kalman_chisq_init.restype = c_int

    # sec_kalman_chisq_reset
    _LIB.sec_kalman_chisq_reset.argtypes = [POINTER(CSecKalmanChiSq)]
    _LIB.sec_kalman_chisq_reset.restype = c_int

    # sec_kalman_chisq_update
    _LIB.sec_kalman_chisq_update.argtypes = [POINTER(CSecKalmanChiSq), c_double, c_double, POINTER(CSecDetectorAlert)]
    _LIB.sec_kalman_chisq_update.restype = c_int

    # sec_schedule_water_filling
    _LIB.sec_schedule_water_filling.argtypes = [
        POINTER(CSecFeederState),
        POINTER(CSecEVSession),
        c_size_t,
        c_bool,
        c_double,
    ]
    _LIB.sec_schedule_water_filling.restype = c_double


def c_cusum_update(
    det_struct: CSecCUSUM,
    reported_feeder_kw: float,
    expected_feeder_kw: float,
) -> tuple[bool, float, float]:
    """Invoke C core CUSUM update."""
    if not LIB_AVAILABLE or _LIB is None:
        raise RuntimeError("libseccharge C library is not available.")
    alert = CSecDetectorAlert()
    _LIB.sec_cusum_update(byref(det_struct), c_double(reported_feeder_kw), c_double(expected_feeder_kw), byref(alert))
    return bool(alert.is_alarm), float(alert.residual_kw), float(alert.test_statistic)


def c_schedule_water_filling(
    transformer_limit_kw: float,
    base_load_kw: float,
    safety_margin_kw: float,
    time_step_hours: float,
    sessions_data: list[dict],
    fail_safe_active: bool = False,
    safe_cap_kw: float = 0.0,
) -> tuple[float, list[float]]:
    """Invoke C core deterministic water-filling scheduler."""
    if not LIB_AVAILABLE or _LIB is None:
        raise RuntimeError("libseccharge C library is not available.")

    feeder = CSecFeederState(
        transformer_limit_kw=c_double(transformer_limit_kw),
        base_load_kw=c_double(base_load_kw),
        safety_margin_kw=c_double(safety_margin_kw),
        time_step_hours=c_double(time_step_hours),
    )

    n = len(sessions_data)
    array_type = CSecEVSession * n
    c_sessions = array_type()

    for i, s in enumerate(sessions_data):
        c_sessions[i].session_id = c_uint32(i + 1)
        c_sessions[i].max_rate_kw = c_double(s.get("max_rate_kw", 0.0))
        c_sessions[i].energy_requested_kwh = c_double(s.get("energy_requested_kwh", 0.0))
        c_sessions[i].energy_delivered_kwh = c_double(s.get("energy_delivered_kwh", 0.0))
        c_sessions[i].remaining_energy_kwh = c_double(s.get("remaining_energy_kwh", 0.0))
        c_sessions[i].dwell_hours = c_double(s.get("dwell_hours", 0.0))
        c_sessions[i].allocated_power_kw = c_double(0.0)

    total_kw = _LIB.sec_schedule_water_filling(
        byref(feeder),
        c_sessions,
        c_size_t(n),
        c_bool(fail_safe_active),
        c_double(safe_cap_kw),
    )

    allocated = [float(c_sessions[i].allocated_power_kw) for i in range(n)]
    return float(total_kw), allocated
