"""Locate the Idleon game window."""
import win32gui

GAME_TITLE = "Legends Of Idleon"


class GameWindow:
    def __init__(self, title: str = GAME_TITLE):
        self.hwnd = win32gui.FindWindow(None, title)
        if not self.hwnd:
            raise RuntimeError(f"Couldn't find a window titled {title!r} - is the game running?")

    def client_rect(self) -> tuple[int, int, int, int]:
        """(left, top, width, height) of the drawable area, in screen pixels."""
        _, _, w, h = win32gui.GetClientRect(self.hwnd)
        left, top = win32gui.ClientToScreen(self.hwnd, (0, 0))
        return left, top, w, h

    def is_foreground(self) -> bool:
        return win32gui.GetForegroundWindow() == self.hwnd

    def is_minimized(self) -> bool:
        return bool(win32gui.IsIconic(self.hwnd))
