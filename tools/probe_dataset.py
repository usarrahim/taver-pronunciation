"""Probe the speechocean762 test split: download (cached) + show structure."""
import io, sys
import soundfile as sf
from datasets import load_dataset, Audio

ds = load_dataset("mispeech/speechocean762", split="test")
ds = ds.cast_column("audio", Audio(decode=False))  # keep raw bytes; decode via soundfile
print("num examples:", len(ds))
ex = ds[0]
a = ex["audio"]
wav, sr = sf.read(io.BytesIO(a["bytes"]))
print("decoded audio:", wav.shape, "sr=", sr, "dtype=", wav.dtype)
print("keys:", list(ex.keys()))
for k, v in ex.items():
    if k == "audio":
        print("  audio keys:", list(v.keys()))
    elif k == "words":
        print("  words[0]:", v[0] if v else None)
        print("  num words:", len(v))
    else:
        print(f"  {k}: {v}")
sys.stdout.flush()
