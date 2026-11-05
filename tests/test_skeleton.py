"""Skeleton test verifying that all modules exist and import cleanly."""

import importlib

import pytest

MODULE_NAMES = [
    "ev_sched",
    "ev_sched.models",
    "ev_sched.hal",
    "ev_sched.simulator",
    "ev_sched.scheduler",
    "ev_sched.attacks",
    "ev_sched.detector",
    "ev_sched.metrics",
]


@pytest.mark.parametrize("mod_name", MODULE_NAMES)
def test_skeleton_modules_importable(mod_name: str) -> None:
    mod = importlib.import_module(mod_name)
    assert mod is not None
