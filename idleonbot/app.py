"""A plain window for starting the bots.

    python -m idleonbot.app            (or IdleonRobot.exe)

Pick a bot and Start: it plays as soon as its minigame is open (F10 in the game stops it).
The bot runs as its own process (this program started with --run NAME), so Stop can
always end it; its output shows in the window.
"""
import importlib
import os
import queue
import subprocess
import sys
import threading

GROUPS = [
    ("Trophy minigames", [("Throwy Darts", "darts"), ("Swishy Hoops", "hoops"), ("Fishing", "fishing")]),
    ("Skills", [("Choppin'", "choppin"), ("Mining", "mining")]),
]
READY = {"darts"}       # the others are greyed out until they're polished


def run_bot(name, args):
    """Run one bot's normal command line in this process."""
    sys.argv = [f"idleonbot.{name}"] + args
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    try:
        importlib.import_module(f"idleonbot.{name}").main()
    except RuntimeError as e:          # e.g. the game isn't running
        print(e)


def command(name):
    if getattr(sys, "frozen", False):          # the .exe
        return [sys.executable, "--run", name]
    return [sys.executable, "-m", "idleonbot.app", "--run", name]


def window():
    import tkinter as tk
    from tkinter import scrolledtext

    root = tk.Tk()
    root.title("IdleonRobot")
    choice = tk.StringVar(value=GROUPS[0][1][0][1])
    procs = []
    lines = queue.Queue()

    top = tk.Frame(root)
    top.pack(fill="x", padx=8, pady=6)
    for title, bots in GROUPS:
        group = tk.LabelFrame(top, text=title, padx=6)
        group.pack(fill="x", pady=(0, 4))
        for label, name in bots:
            ready = name in READY
            tk.Radiobutton(group, text=label if ready else f"{label} (coming soon)", variable=choice, value=name,
                           state="normal" if ready else "disabled").pack(anchor="w")
    tk.Label(top, text="The bot plays as soon as the minigame is open. F10 in the game stops it.").pack(anchor="w", pady=(4, 0))
    button = tk.Button(top, text="Start", width=10)
    button.pack(anchor="w", pady=4)
    out = scrolledtext.ScrolledText(root, width=80, height=20, state="disabled")
    out.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def show(text):
        out.configure(state="normal")
        out.insert("end", text)
        out.see("end")
        out.configure(state="disabled")

    def reader(p, name):
        for line in p.stdout:
            lines.put(line)
        lines.put((p, name))

    def start(name):
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        p = subprocess.Popen(command(name), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1, creationflags=flags,
                             env={**os.environ, "PYTHONUNBUFFERED": "1"})
        procs.append(p)
        threading.Thread(target=reader, args=(p, name), daemon=True).start()
        show(f"--- started {name}\n")

    def start_stop():
        if not procs:
            start(choice.get())
            button.configure(text="Stop")
        else:
            for p in procs:
                p.terminate()

    def poll():
        try:
            while True:
                line = lines.get_nowait()
                if isinstance(line, tuple):        # a process ended
                    p, name = line
                    show(f"--- {name} stopped\n")
                    if p in procs:
                        procs.remove(p)
                    if not procs:
                        button.configure(text="Start")
                else:
                    show(line)
        except queue.Empty:
            pass
        root.after(100, poll)

    def close():
        for p in procs:
            p.terminate()
        root.destroy()

    button.configure(command=start_stop)
    root.protocol("WM_DELETE_WINDOW", close)
    poll()
    root.mainloop()


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--run":
        run_bot(sys.argv[2], sys.argv[3:])
    else:
        window()


if __name__ == "__main__":
    main()
