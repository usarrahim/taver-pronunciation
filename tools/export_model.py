"""Export a pretrained wav2vec2 phoneme-recognition model to ONNX for Taver.

The default model, ``facebook/wav2vec2-lv-60-espeak-cv-ft``, is a CTC acoustic
model that emits eSpeak IPA phonemes directly from the raw 16 kHz waveform. That
is exactly what the GOP scorer needs to tell, e.g., /t/ (tree) apart from /theta/
(three) instead of letting a language model paper over the mistake.

Outputs (into ``models/``):
  * phoneme.onnx          - FP32 model, input_values[B,T] -> logits[B,F,V]
  * phoneme.int8.onnx     - dynamically quantized model (smaller/faster on CPU)
  * phoneme_vocab.txt     - "<id>\\t<symbol>" per line (+ blank id header)

Run inside the project venv:
    tools\\.venv\\Scripts\\python.exe tools\\export_model.py
"""

import argparse
import os
import sys

import torch

DEFAULT_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "models"))
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    fp32_path = os.path.join(outdir, "phoneme.onnx")
    int8_path = os.path.join(outdir, "phoneme.int8.onnx")
    vocab_path = os.path.join(outdir, "phoneme_vocab.txt")

    import json
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2ForCTC

    print(f"[export] loading {args.model} ...", flush=True)
    # NOTE: we deliberately avoid AutoProcessor here. The phoneme tokenizer for
    # this checkpoint hard-requires the `phonemizer`/espeak-ng backend just to
    # construct, which is an awkward native dependency on Windows. For export we
    # only need (a) the model weights and (b) the symbol<->id vocabulary, so we
    # pull vocab.json straight from the hub.
    model = Wav2Vec2ForCTC.from_pretrained(args.model)
    model.eval()

    vocab_file = hf_hub_download(args.model, "vocab.json")
    with open(vocab_file, "r", encoding="utf-8") as vf:
        vocab = json.load(vf)  # symbol -> id
    id_to_sym = {int(i): s for s, i in vocab.items()}
    vocab_size = max(id_to_sym) + 1
    blank_id = model.config.pad_token_id
    if blank_id is None:
        blank_id = vocab.get("<pad>", 0)
    word_delim = "|" if "|" in vocab else " "

    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write(f"# vocab_size={vocab_size} blank_id={blank_id} "
                f"word_delimiter={word_delim}\n")
        for i in range(vocab_size):
            f.write(f"{i}\t{id_to_sym.get(i, '<unk>')}\n")
    print(f"[export] wrote vocab ({vocab_size} symbols, blank={blank_id}) -> {vocab_path}",
          flush=True)

    # Dummy input: 1 second of audio. Time axis is dynamic.
    dummy = torch.randn(1, 16000, dtype=torch.float32)

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_values):
            return self.m(input_values).logits

    wrapper = Wrapper(model)
    print(f"[export] tracing -> {fp32_path}", flush=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        fp32_path,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={"input_values": {0: "batch", 1: "samples"},
                      "logits": {0: "batch", 1: "frames"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # stable TorchScript exporter: predictable I/O + dynamic axes
    )
    print(f"[export] FP32 model written ({os.path.getsize(fp32_path)/1e6:.1f} MB)", flush=True)

    if not args.no_quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            print(f"[export] quantizing (int8) -> {int8_path}", flush=True)
            quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
            print(f"[export] INT8 model written ({os.path.getsize(int8_path)/1e6:.1f} MB)",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[export] quantization skipped: {e}", flush=True)

    print("EXPORT-DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
