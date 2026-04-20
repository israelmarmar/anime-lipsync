import math
from typing import List, Tuple

import soundfile as sf

from .constants import ipa_to_vowel, vowel_to_mouth_type, _DEFAULT_LANG, _INVALID_LANG_IDS


def sanitize_lang(lang_id) -> str:
    s = str(lang_id).strip() if lang_id is not None else ""
    return _DEFAULT_LANG if (not s or s in _INVALID_LANG_IDS) else s


class AllosaurusDetector:
    def __init__(self, audio_path: str, fps: int, lang_id: str):
        self.audio_path = audio_path
        self.fps        = fps
        self.lang_id    = lang_id
        self._duration  = sf.info(audio_path).duration

    @property
    def total_frames(self) -> int:
        return int(math.ceil(self._duration * self.fps))

    def detect(self) -> Tuple[List[str], List[int]]:
        try:
            from allosaurus.app import read_recognizer
        except ImportError as e:
            raise ImportError("pip install allosaurus") from e

        model = read_recognizer()
        print(f"[LipSync] Allosaurus lang='{self.lang_id}' {self._duration:.1f}s...")
        raw = model.recognize(self.audio_path, lang_id=self.lang_id, timestamp=True)

        segments = []
        for line in raw.strip().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                t0, dur, phone = float(parts[0]), float(parts[1]), parts[2]
                segments.append((t0, t0 + dur, ipa_to_vowel(phone)))
            except ValueError:
                continue

        n, fd    = self.total_frames, 1.0 / self.fps
        phonemes = ["_"] * n
        for t0, t1, vowel in segments:
            if vowel == "_":
                continue
            for f in range(max(0, int(t0 * self.fps)),
                           min(n, int(math.ceil(t1 * self.fps)))):
                if min(f * fd + fd, t1) - max(f * fd, t0) > 0:
                    phonemes[f] = vowel

        return phonemes, [vowel_to_mouth_type(p) for p in phonemes]
