"""Global hotkeys by polling GetAsyncKeyState (no hooks, no extra packages)."""
import ctypes

user32 = ctypes.windll.user32

VK_F10 = 0x79


class HotkeyPoller:
    def __init__(self, *vks: int):
        self._down = {vk: False for vk in vks}

    def pressed(self, vk: int) -> bool:
        """True once per physical press of `vk` (edge-triggered)."""
        now = bool(user32.GetAsyncKeyState(vk) & 0x8000)
        was = self._down[vk]
        self._down[vk] = now
        return now and not was
