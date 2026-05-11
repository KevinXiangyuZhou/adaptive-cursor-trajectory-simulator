"""
Human-like Cursor Simulator - A Python package for human-like cursor movement simulation.

This package provides a Playwright-compatible interface for generating realistic
human cursor trajectories using models based on motor control and human behavior research.
"""

__version__ = "0.1.0"

from .cursor_simulator import CursorSimulator
from .model import model
from .mpcc_model import reset_warm_start
from .speed_model import SpeedModel, GAMSpeedModel
from .noise import motor_and_device_noise, single_step_motor_and_device_noise

__all__ = [
    "CursorSimulator",
    "model",
    "reset_warm_start",
    "SpeedModel",
    "GAMSpeedModel",
    "motor_and_device_noise",
    "single_step_motor_and_device_noise",
]
