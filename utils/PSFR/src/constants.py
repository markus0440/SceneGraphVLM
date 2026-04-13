import pathlib

BASE_DIR = pathlib.Path(__file__).parents[1]


# Defaults (overridable by config.json via run.py)
FRAMES_DIR = BASE_DIR / "data" / "frames"
OUTPUT_DIR = BASE_DIR / "data" / "output"
KEYFRAMES_DIR = OUTPUT_DIR / "keyframes"


# Image I/O defaults
DEFAULT_EXT = "png"
PAD_LEN = 6