"""The game's sound, for runs played on nes-py.

nes-py, the emulator behind gym-super-mario-bros, draws the picture but has
no sound chip, so a run played on it is silent. stable-retro's NES emulator
(FCEUmm) does emulate the sound chip, but simply playing the same buttons on
it doesn't reproduce the run: the two emulators differ in small timing
details, and within a few seconds Mario is somewhere else.

So instead ``GameSound`` runs stable-retro *in lockstep* with nes-py:

    before every frame, copy nes-py's RAM (the game state: positions,
    enemies, timers, ...) into stable-retro, press the same buttons, run
    one frame there, and keep the audio it produced.

Two parts of the RAM are treated differently:

* The sound engine's own state (where each channel is in the tune, note
  lengths) isn't copied. nes-py sometimes runs 0 or 2 of the game's frames
  in one step, so copying it would make the music skip or stutter;
  stable-retro's sound engine keeps its own and plays smoothly.
* The sound requests (one bit per jump, stomp, coin, tune, ...) are merged
  in, so every sound the nes-py run asks for is played.

nes-py stays in charge of the game, the picture and the score; stable-retro
only makes the sound. The stack page (0x100-0x1FF) isn't copied either: it
belongs to each emulator's CPU, not to the game.

Needs the optional ``sound`` extra: ``uv sync --extra sound``.
"""

from __future__ import annotations

import numpy as np

# stable-retro's NES button order, and each button's bit in the controller
# byte (as nes-py and marlio.controller encode it).
_RETRO_BUTTON_BITS = (0b10, 0, 0b100, 0b1000, 0b10000, 0b100000, 0b1000000, 0b10000000, 0b1)
_START = 0b1000
_RAM_SIZE = 2048
_STACK = range(0x100, 0x200)
# The sound engine's own state: where each channel is in the music, note
# lengths, sound-effect counters (from the SMB disassembly). Not copied, so
# stable-retro's sound engine keeps running on its own state and the music
# plays smoothly; it still receives the run's sound requests (the queues at
# 0xFA-0xFF) and everything else about the game.
_SOUND_ENGINE = [*range(0xF0, 0xFA), *range(0x7B0, 0x7CB)]
# The sound requests: one bit per sound effect or tune, set by the game and
# cleared by the sound engine once started. These are merged (bitwise or)
# rather than copied; see GameSound.
_SOUND_QUEUES = list(range(0xFA, 0x100))
_GAME_RAM = np.array([a for a in range(_RAM_SIZE)
                      if a not in _STACK and a not in _SOUND_ENGINE and a not in _SOUND_QUEUES])
_TIME_DIGITS = 0x07F8
_GAME_MODE = 0x0770  # 1 = playing
_PRELEVEL_TIMER = 0x07A0


class GameSound:
    """Produces the audio of a nes-py run of Super Mario Bros., one frame at a time.

    Typical use::

        sound = GameSound()
        sound.start(ram)                      # nes-py's RAM after reset
        for each frame:
            env.step(buttons)                 # nes-py plays the frame
            audio = sound.step(byte, env.unwrapped.ram)

    ``audio`` is int16 stereo, about 533 samples per frame at ``rate``
    (32040.5 samples per second, the NES's own rate).

    stable-retro allows one emulator per process, so make one ``GameSound``.
    """

    def __init__(self):
        try:
            from stable_retro import _retro
        except ImportError as error:
            raise ImportError(
                "sound needs stable-retro: install it with `uv sync --extra sound`"
            ) from error
        from gym_super_mario_bros._roms import smb1_rom_path

        self.emulator = _retro.RetroEmulator(smb1_rom_path())
        self._data = _retro.GameDataGlue()
        self.emulator.configure_data(self._data)
        self.rate = self.emulator.get_audio_rate()
        self._ram = np.zeros(_RAM_SIZE, np.uint8)
        self._next_ram: np.ndarray | None = None
        self._boot()
        self._in_game = self.emulator.get_state()

    def _frame(self, byte: int) -> np.ndarray:
        mask = np.array([(byte & bit) != 0 for bit in _RETRO_BUTTON_BITS], np.uint8)
        self.emulator.set_button_mask(mask, 0)
        self.emulator.step()
        self._read_ram()
        return self.emulator.get_audio()

    def _read_ram(self) -> None:
        self._data.update_ram()
        blocks = self._data.memory.blocks
        self._ram = np.frombuffer(blocks[0] + blocks[1024], np.uint8)

    def _write_ram(self, ram: np.ndarray) -> None:
        for address in _GAME_RAM[self._ram[_GAME_RAM] != ram[_GAME_RAM]]:
            self._data.memory.assign(int(address), "|u1", int(ram[address]))
        for address in _SOUND_QUEUES:
            merged = int(self._ram[address]) | int(ram[address])
            if merged != self._ram[address]:
                self._data.memory.assign(address, "|u1", merged)
        self._read_ram()

    def _boot(self) -> None:
        """Press start until a level is being played, so the sound chip is set up for play."""
        def time_left():
            return int("".join(str(self._ram[_TIME_DIGITS + i]) for i in range(3)))

        self._frame(_START)
        self._frame(0)
        while time_left() == 0 or self._ram[_GAME_MODE] != 1:
            self._frame(_START)
            self._frame(0)
            self._data.memory.assign(_PRELEVEL_TIMER, "|u1", 0)
            self._read_ram()

    def start(self, ram: np.ndarray) -> None:
        """Begin a run from this game state (nes-py's RAM after ``reset``)."""
        self.emulator.set_state(self._in_game)
        self._read_ram()
        self._next_ram = np.array(ram, np.uint8)

    def step(self, byte: int, ram_after: np.ndarray) -> np.ndarray:
        """The audio of the frame nes-py just played with controller ``byte``.

        ``ram_after`` is nes-py's RAM after that frame; it's the starting
        point of the next one.
        """
        if self._next_ram is None:
            raise RuntimeError("call start() first")
        self._write_ram(self._next_ram)
        audio = self._frame(byte)
        self._next_ram = np.array(ram_after, np.uint8)
        return audio
