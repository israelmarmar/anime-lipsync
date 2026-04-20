import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Pipeline
_QUEUE_SENTINEL       = None
_DEFAULT_LANG         = "uni"
_INVALID_LANG_IDS     = {"ipa", "0", "1", "", "None", "none"}
_VRAM_PER_WORKER_MB   = 1_600
_VRAM_PER_KSAMPLER_MB = 2_500
_MIN_WORKERS          = 1
_MAX_WORKERS          = 16
_OOM_MAX_RETRIES      = 3
_OOM_BACKOFF_BASE     = 2.0

# Imagem
_IMG_EXT      = "webp"
_WEBP_QUALITY = 80

# Tipos de boca
MOUTH_TYPE_NAMES = {0: "neutral_closed", 1: "half_open", 2: "fully_open"}

MOUTH_TYPE_PROMPTS = {
    0: ("Anime screencap style, a traditional-style anime character's "
        "closed mouth, only by lines, happy"),
    1: ("Anime screencap style, no teeth, a traditional-style anime, "
        "a traditional-style anime character's half open mouth, "
        "pink or reddish roof of the mouth without, happy face"),
    2: ("Anime screencap style, no teeth, a traditional-style anime "
        "character's mouth wide open, pink or reddish roof of the mouth, "
        "happy face"),
}

# Mapeamento fonema → tipo de boca
_VOWEL_TO_MOUTH_TYPE = {"_": 0, "e": 1, "i": 1, "u": 1, "a": 2, "o": 2}

IPA_VOWEL_MAP = {
    "a": "a", "aː": "a", "ã": "a", "ă": "a",
    "ɑ": "a", "ɑː": "a", "ɐ": "a", "ɐː": "a", "ä": "a", "ɶ": "a",
    "e": "e", "eː": "e", "e̞": "e",
    "ɛ": "e", "ɛː": "e", "ɜ": "e", "ɜː": "e",
    "æ": "e", "ə": "e", "əː": "e", "ɘ": "e",
    "ø": "e", "øː": "e", "œ": "e", "ɵ": "e", "ɵː": "e",
    "i": "i", "iː": "i", "ij": "i", "i̞": "i", "i̥": "i", "i̯": "i",
    "ɪ": "i", "ɪ̯": "i", "y": "i", "yː": "i", "ʏ": "i", "ɨ": "i", "ɨː": "i",
    "o": "o", "oː": "o",
    "ɔ": "o", "ɔː": "o", "ɒ": "o", "ɒː": "o", "ʌ": "o", "ɤ": "o",
    "u": "u", "uː": "u", "ʉ": "u", "ʉː": "u", "ʊ": "u", "ɯ": "u",
}

SILENCE_TOKENS = {"sil", "<sil>", "sp", "<sp>", "spn", "<noise>", "SIL", "SPN"}


def vowel_to_mouth_type(v: str) -> int:
    return _VOWEL_TO_MOUTH_TYPE.get(v, 0)


def ipa_to_vowel(phone: str) -> str:
    if phone in SILENCE_TOKENS:
        return "_"
    v = IPA_VOWEL_MAP.get(phone)
    if v:
        return v
    return IPA_VOWEL_MAP.get(
        phone.rstrip("ː").rstrip("̯").rstrip("̥").rstrip("̤"), "_")
