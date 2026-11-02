# conftest.py — thêm root project vào sys.path để test import được `vad` và `scripts`
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
