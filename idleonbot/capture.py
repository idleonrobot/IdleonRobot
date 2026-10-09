"""Fast capture of small screen regions via GDI BitBlt.

Reuses the memory DC/bitmap between grabs of the same size, which keeps a
small-region grab around a millisecond.
"""
import numpy as np
import win32con
import win32gui
import win32ui


class ScreenGrabber:
    def __init__(self):
        self._screen_hdc = win32gui.GetDC(0)
        self._src = win32ui.CreateDCFromHandle(self._screen_hdc)
        self._mem = self._src.CreateCompatibleDC()
        self._bmp = None
        self._size = None

    def grab(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        """Return the region as an (height, width, 3) uint8 RGB array."""
        if self._size != (width, height):
            if self._bmp is not None:
                win32gui.DeleteObject(self._bmp.GetHandle())
            self._bmp = win32ui.CreateBitmap()
            self._bmp.CreateCompatibleBitmap(self._src, width, height)
            self._mem.SelectObject(self._bmp)
            self._size = (width, height)
        self._mem.BitBlt((0, 0), (width, height), self._src, (left, top), win32con.SRCCOPY)
        bgra = np.frombuffer(self._bmp.GetBitmapBits(True), np.uint8).reshape(height, width, 4)
        return bgra[:, :, 2::-1]  # BGRA -> RGB (a view; copy if you keep it)

    def close(self):
        if self._bmp is not None:
            win32gui.DeleteObject(self._bmp.GetHandle())
        self._mem.DeleteDC()
        self._src.DeleteDC()
        win32gui.ReleaseDC(0, self._screen_hdc)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
