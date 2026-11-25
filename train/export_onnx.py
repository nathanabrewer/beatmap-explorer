"""Export a trained model for the browser (or any ONNX runtime).

    python -I train/export_onnx.py --ckpt runs/NAME/best.pt --out web-model/ [--int8]

Writes a bundle a web page can load with onnxruntime-web:
    encoder.onnx       mel window (1, 800, 96) float32 -> memory (1, 400, d)
    decoder.onnx       memory + tokens (1, L) int64 -> logits (1, L, vocab)
    *.int8.onnx        8-bit weights (much smaller download), with --int8
    mel_filters.bin    float32 (96, 1025): power spectrum -> mel bands, exactly
                       as training made them (24 kHz, 2048-point FFT, hop 240)
    model.json         token names, window and time settings, how to compute
                       the input (log(mel + 1e-6)), checks run at export
The exported graphs are compared with PyTorch on a real window before saving.
Licensing: a model trained on BeatSaver maps (or on extracted official maps)
must not be shipped; publish only models trained on maps you have rights to.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prepare  # noqa: E402
import tokens as T  # noqa: E402
from model import MapModel  # noqa: E402


class Encoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, mel):
        return self.m.encode(mel)


class Decoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, memory, tokens):
        return self.m.decode(memory, tokens)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--check-mel", help="a prepared song (path without extension) to test on; default: noise")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu")
    model = MapModel(ck["config"]["size"], dropout=0.0)
    model.load_state_dict(ck["model"])
    model.eval()
    # PyTorch's fused inference path for transformer layers has no ONNX form.
    torch.backends.mha.set_fastpath_enabled(False)
    if a.check_mel:
        mel = prepare.mel_float(prepare.read_mel(a.check_mel + ".mel.npy"))[: T.N_TIME]
    else:
        mel = np.random.default_rng(0).normal(-6, 3, (T.N_TIME, prepare.N_MELS)).astype(np.float32)
    x = torch.tensor(mel).unsqueeze(0)
    toks = torch.tensor([T.header(4, "Standard", 16) + [T.SEP, T.TIME0 + 250, T.ID["NOTE_010"], T.ID["DIR_1"]]])
    enc_path, dec_path = os.path.join(a.out, "encoder.onnx"), os.path.join(a.out, "decoder.onnx")
    with torch.no_grad():
        memory = model.encode(x)
        logits = model.decode(memory, toks)
        torch.onnx.export(Encoder(model), (x,), enc_path, input_names=["mel"], output_names=["memory"], opset_version=17, dynamo=False)
        torch.onnx.export(Decoder(model), (memory, toks), dec_path, input_names=["memory", "tokens"], output_names=["logits"],
                          dynamic_axes={"tokens": {1: "length"}, "logits": {1: "length"}}, opset_version=17, dynamo=False)
    import onnxruntime as ort
    checks = {}
    paths = [("fp32", enc_path, dec_path)]
    if a.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        for p in (enc_path, dec_path):
            quantize_dynamic(p, p.replace(".onnx", ".int8.onnx"), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
        paths.append(("int8", enc_path.replace(".onnx", ".int8.onnx"), dec_path.replace(".onnx", ".int8.onnx")))
    for label, ep, dp in paths:
        e = ort.InferenceSession(ep, providers=["CPUExecutionProvider"])
        d = ort.InferenceSession(dp, providers=["CPUExecutionProvider"])
        mem = e.run(None, {"mel": mel[None]})[0]
        out = d.run(None, {"memory": mem, "tokens": toks.numpy()})[0]
        same_top = float((out.argmax(-1) == logits.numpy().argmax(-1)).mean())
        checks[label] = {"maxLogitDiff": float(np.abs(out - logits.numpy()).max()), "sameTopToken": same_top,
                         "encoderMB": round(os.path.getsize(ep) / 1e6, 1), "decoderMB": round(os.path.getsize(dp) / 1e6, 1)}
        print(f"{label}: logits differ by at most {checks[label]['maxLogitDiff']:.4f}, same top token {same_top:.0%}, "
              f"encoder {checks[label]['encoderMB']} MB, decoder {checks[label]['decoderMB']} MB")
    import librosa
    fb = librosa.filters.mel(sr=prepare.SAMPLE_RATE, n_fft=prepare.N_FFT, n_mels=prepare.N_MELS, fmin=prepare.FMIN, fmax=prepare.FMAX).astype(np.float32)
    fb.tofile(os.path.join(a.out, "mel_filters.bin"))
    info = {
        "size": ck["config"]["size"], "step": ck.get("step"), "trainedOn": ck.get("args", {}).get("data"),
        "audio": {"sampleRate": prepare.SAMPLE_RATE, "mono": True, "nFft": prepare.N_FFT, "hop": prepare.HOP, "window": "hann, centered (reflect padding)",
                  "power": 2, "nMels": prepare.N_MELS, "fmin": prepare.FMIN, "fmax": prepare.FMAX, "melFilters": "mel_filters.bin float32 [nMels][nFft/2+1]",
                  "input": "log(mel + 1e-6), frames x bands; silence outside the song is log(1e-6)"},
        "window": {"seconds": T.WINDOW, "contextSeconds": T.CONTEXT, "ticksPerSecond": T.TICKS, "frames": T.N_TIME},
        "tokens": T.NAMES, "specials": {"PAD": T.PAD, "BOS": T.BOS, "EOS": T.EOS, "SEP": T.SEP, "TIME0": T.TIME0},
        "wallDurations": T.WALL_DURATIONS, "checks": checks,
        "decoding": "see train/generate.py: header BOS DIFF CHAR NJS ROT, context events, SEP, then sample with the grammar mask until EOS",
    }
    json.dump(info, open(os.path.join(a.out, "model.json"), "w"), indent=1)
    print(f"bundle written to {a.out}")


if __name__ == "__main__":
    main()
