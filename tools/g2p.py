"""Grapheme-to-phoneme helper: turn an English word/phrase into a phoneme string
in the model's eSpeak-IPA symbols, ready to pass to `TAVER_set_target` /
`TAVER_cli --target`.

Two backends:
  * If `phonemizer` (+ espeak-ng) is installed, any word is supported.
  * Otherwise a small built-in lexicon covers the demo vocabulary. The C++ CLI
    embeds the same lexicon, so `--word tree` works without this tool.

Usage:
    python tools/g2p.py three
    python tools/g2p.py "tree three light"
"""
import sys

# Canonical pronunciations in the model's symbol set (see models/phoneme_vocab.txt).
LEXICON = {
    "tree": "t \u0279 i\u02d0",
    "three": "\u03b8 \u0279 i\u02d0",
    "think": "\u03b8 \u026a \u014b k",
    "sink": "s \u026a \u014b k",
    "ship": "\u0283 \u026a p",
    "sip": "s \u026a p",
    "right": "\u0279 a\u026a t",
    "light": "l a\u026a t",
    "thanks": "\u03b8 \u00e6 \u014b k s",
    "water": "w \u0254 t \u025a",
    "hello": "h \u0259 l o\u028a",
}


def via_phonemizer(text):
    try:
        from phonemizer import phonemize
    except Exception:
        return None
    ipa = phonemize(text, language="en-us", backend="espeak",
                    strip=True, with_stress=False)
    # espeak concatenates phonemes; space them out coarsely by character runs.
    return " ".join(ch for ch in ipa.replace(" ", " ") if ch.strip())


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    text = " ".join(sys.argv[1:]).lower()
    out = []
    for w in text.split():
        if w in LEXICON:
            out.append(LEXICON[w])
        else:
            ph = via_phonemizer(w)
            if ph:
                out.append(ph)
            else:
                sys.stderr.write(
                    f"[g2p] '{w}' not in lexicon and phonemizer/espeak not installed.\n"
                    f"      Install with: pip install phonemizer  (and espeak-ng)\n")
                return 2
    # Word delimiter for this model's vocab is a space; join words with it too.
    print(" ".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
