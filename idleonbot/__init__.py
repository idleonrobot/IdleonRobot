"""Screen-reading bots for Idleon minigames.

Shared core lives here (window, capture, input, hotkeys); each minigame gets
its own module (e.g. idleonbot.choppin).
"""
import ctypes

# Work in physical pixels so window rects and captures line up under any
# Windows display scaling. Must happen before any window/DC calls.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except (AttributeError, OSError):
    ctypes.windll.user32.SetProcessDPIAware()
