If this trophy is worth as much as a supporter pack to you, please consider tipping here! [Ko-fi.com/idlerobot](https://ko-fi.com/idlerobot)

# IdleonRobot

Bots that play the minigames in Legends of Idleon (Steam) by watching the game window
and clicking for you.

- **Throwy Darts**: plays until 9 bullseyes in a row (the Nine Dart Finish trophy), then
  stops

Coming soon (greyed out in the app while they're being polished):

- **Swishy Hoops**: plays to a score of 40 (the Baller trophy)
- **Fishing**: goes for the Megalodon and stops once it's caught it (the trophy)
- **Choppin'**: plays to 150 points
- **Mining**: jumps the cart over the gaps and collects ore for as long as it can

## How to use it

1. Download `IdleonRobot.exe` from the Releases page and run it. Windows may warn about an
   unknown app; choose "More info" then "Run anyway".
2. Start the game and open the minigame you want (a normal window size works best).
3. In IdleonRobot, pick the minigame and press **Start**.
4. Click back into the game. The bot starts playing straight away, and only while the
   game is the active window. It stops by itself when it's done (or at game over);
   **F10** in the game, or Stop in IdleonRobot, stops it any time.

Don't click in the game while the bot is playing. The window shows each throw's result
("miss" or how many bullseyes in a row).

**Fishing**: the bot leaves whales alone after catching two in a row. That's how the
Megalodon appears, so it isn't a bug.

## Privacy

The bot only looks at the game window, and only while the game is the active window.
It never captures the rest of your screen or saves any pictures. The only file it
writes is `darts_calibration.json` (what the Darts bot has learned about aiming), in
the folder it was run from.

## Running from source

The exe is built from exactly the code in this repo (`idleonbot/`), so you can read
it to check what it does, or skip the exe and run it with Python. Needs Windows and
Python 3.10+.

```
pip install -r requirements.txt
python IdleonRobot.py
```

To build the exe yourself:

```
pip install pyinstaller
python -m PyInstaller --onefile --noconsole --name IdleonRobot --collect-submodules idleonbot --exclude-module PIL IdleonRobot.py
```

Each bot can also be run on its own, for example `python -m idleonbot.fishing`. Add
`--help` to see its options, such as `--stop-at 0` (Darts, Hoops) or `--keep-going`
(Fishing) to keep playing after the trophy.

## Licence

MIT - see [LICENSE](LICENSE).
