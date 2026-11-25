#!/usr/bin/env python3
"""The part of tts that runs inside a model's virtualenv. You normally don't call this directly, use tts.py.

  engine.py prepare --ref FILE --voice-dir DIR [--ref-text T] [--lang de]
      Turns any audio file into a clean 24 kHz mono reference (<= 15 s) in DIR/ref.wav and writes the
      transcript DIR/ref.txt (given by --ref-text, or heard by Whisper).
  engine.py speak --model REPO --voice-dir DIR --text-file F --out OUT.wav [--lang de] [--seed N]
      Reads DIR/ref.wav and speaks the text in that cloned voice.
  engine.py speak --model REPO --speaker NAME [--style "..."] --text-file F --out OUT.wav
      Speaks the text with one of the built-in speakers of the Qwen3-TTS CustomVoice model (no reference needed).

Settings that come from a lot of trial and error with Qwen3-TTS: repetition_penalty 1.05, never greedy decoding (it babbles
forever), a hard cap on generated tokens (a missing end-of-speech token otherwise runs for minutes), no post-processing of
the output, and a reference of about 5 to 15 seconds.
"""
import argparse
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import time

REF_SR = 24000
REF_MIN = 3.0          # seconds; shorter references clone badly
REF_KEEP = 15.0        # a longer reference is cut to a sentence end at or before this
REF_TRIM_ABOVE = 16.0
GAP = 0.25             # seconds of silence between generated chunks
CHUNK_CHARS = 250      # text is spoken in sentence-sized pieces; long single passes drift
CJK = ("zh", "ja", "ko")
QWEN_LANGS = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
              "ru": "Russian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean"}


def log(*a):
    print(*a, flush=True)


def die(msg):
    print(f"error: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


# ---------- device ----------

def pick_device(want):
    import torch
    if want == "auto":
        if torch.cuda.is_available():
            want = "cuda"
        elif torch.backends.mps.is_available():
            want = "mps"
        else:
            want = "cpu"
    return "cuda:0" if want == "cuda" else want


# ---------- reference audio ----------

def load_audio(path):
    """Any audio file -> mono float32 at 24 kHz. librosa reads wav/flac/ogg/mp3; ffmpeg (if installed) covers the rest."""
    import librosa
    try:
        y, _ = librosa.load(path, sr=REF_SR, mono=True)
        return y
    except Exception as first:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            die(f"cannot read {path} ({first}). Install ffmpeg or convert the file to wav.")
        tmp = pathlib.Path(path).with_suffix(".tmp-convert.wav")
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(REF_SR), str(tmp)], check=True)
        try:
            y, _ = librosa.load(str(tmp), sr=REF_SR, mono=True)
        finally:
            tmp.unlink(missing_ok=True)
        return y


def transcribe(y, lang):
    """-> (text, words) with word times in seconds (faster-whisper, CPU, so it never competes with the TTS model for VRAM)."""
    import librosa
    from faster_whisper import WhisperModel
    log("Listening to the reference (Whisper)...")
    model = WhisperModel(os.environ.get("VOICE_WHISPER", "small"), device="cpu", compute_type="int8")
    y16 = librosa.resample(y, orig_sr=REF_SR, target_sr=16000)
    segments, _ = model.transcribe(y16, language=None if lang in ("", "auto") else lang, beam_size=5, word_timestamps=True)
    words = [w for s in segments for w in (s.words or [])]
    return "".join(w.word for w in words).strip(), words


def pick_cut(words, lo=8.0, hi=REF_KEEP):
    """Where to cut a long reference: the last sentence end between lo and hi, else the last real pause, else hi."""
    ends = [w.end for w in words if lo <= w.end <= hi and w.word.strip()[-1:] in ".!?…。！？"]
    if ends:
        return ends[-1]
    pause = None
    for a, b in zip(words, words[1:]):
        if lo * 0.6 <= a.end <= hi and b.start - a.end >= 0.3:
            pause = a.end
    return pause or hi


def prepare(a):
    import librosa
    import soundfile as sf
    voice_dir = pathlib.Path(a.voice_dir)
    voice_dir.mkdir(parents=True, exist_ok=True)
    y = load_audio(a.ref)
    y, _ = librosa.effects.trim(y, top_db=35)
    seconds = len(y) / REF_SR
    if seconds < REF_MIN:
        die(f"the reference is only {seconds:.1f} s long after trimming silence. Use 5 to 15 s of clean speech.")
    text = (a.ref_text or "").strip()
    if seconds > REF_TRIM_ABOVE:
        if text:
            cut = REF_KEEP
            log(f"Reference is {seconds:.0f} s, using the first {cut:.0f} s. (--ref-text should match that part.)")
        else:
            text, words = transcribe(y, a.lang)
            cut = pick_cut(words)
            text = "".join(w.word for w in words if w.end <= cut + 0.05).strip()
            log(f"Reference is {seconds:.0f} s, cut to {cut:.1f} s at a sentence end.")
        y = y[: int(cut * REF_SR)]
    elif not text:
        text, _ = transcribe(y, a.lang)
    sf.write(str(voice_dir / "ref.wav"), y, REF_SR, subtype="PCM_16")
    if text:
        (voice_dir / "ref.txt").write_text(text + "\n", encoding="utf-8")
    elif (voice_dir / "ref.txt").exists():
        (voice_dir / "ref.txt").unlink()   # stale transcript of an older reference
    log(f"Reference ready: {len(y) / REF_SR:.1f} s" + (f"\nTranscript: {text}" if text else ""))
    if len(y) / REF_SR < 5:
        log("Note: under 5 s of reference works, but 8 to 15 s clones noticeably better.")


# ---------- text ----------

def split_text(text, maxc):
    """Sentence-sized pieces of at most maxc characters (a longer sentence is split at commas, then at spaces)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…。！？])\s*", " ".join(text.split())) if s.strip()]
    pieces = []
    for s in sentences:
        while len(s) > maxc:
            cut = max(s.rfind(sep, 0, maxc) for sep in (", ", "; ", "、", "，", " "))
            cut = cut + 1 if cut > maxc // 3 else maxc
            pieces.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            pieces.append(s)
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + 1 + len(p) > maxc:
            chunks.append(cur)
            cur = p
        else:
            joiner = "" if re.search(r"[　-鿿＀-￯]$", cur) else " "   # no spaces between Chinese/Japanese sentences
            cur = f"{cur}{joiner}{p}".strip()
    if cur:
        chunks.append(cur)
    return chunks


# ---------- models ----------

class Qwen:
    """Qwen3-TTS. With `speaker`: a built-in speaker (CustomVoice model). Otherwise: the cloned voice in `voice_dir` (Base model)."""

    def __init__(self, repo, device, lang, voice_dir=None, speaker="", style=""):
        import torch
        from qwen_tts import Qwen3TTSModel
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32   # bf16 is only safe on CUDA
        log(f"Loading {repo} on {device} (the first run downloads it)...")
        t = time.time()
        self.model = Qwen3TTSModel.from_pretrained(repo, device_map=device, dtype=dtype)
        log(f"Model loaded in {time.time() - t:.0f} s")
        self.lang = QWEN_LANGS.get(lang, "Auto")
        self.per_char = 0.5 if lang in CJK else 0.2
        self.speaker, self.style = speaker.lower(), style.strip()
        if self.speaker:
            known = self.model.get_supported_speakers() or []
            if self.speaker not in known:
                die(f"unknown speaker '{speaker}'. Available: {', '.join(known)}")
            return
        ref = voice_dir / "ref.wav"
        txt = voice_dir / "ref.txt"
        if txt.exists() and txt.read_text(encoding="utf-8").strip():
            self.prompt = self.model.create_voice_clone_prompt(ref_audio=str(ref), ref_text=txt.read_text(encoding="utf-8").strip())
        else:
            # no transcript: ask Whisper now (best quality); if that is not possible fall back to the weaker speaker-only mode
            import librosa
            text, _ = transcribe(librosa.load(str(ref), sr=REF_SR, mono=True)[0], lang)
            if text:
                txt.write_text(text + "\n", encoding="utf-8")
                log(f"Transcript: {text}")
                self.prompt = self.model.create_voice_clone_prompt(ref_audio=str(ref), ref_text=text)
            else:
                log("No transcript available, using speaker-only mode (lower similarity).")
                self.prompt = self.model.create_voice_clone_prompt(ref_audio=str(ref), x_vector_only_mode=True)

    def speak(self, text, seed):
        import numpy as np
        import torch
        tokens = min(1024, int(max(8.0, len(text) * self.per_char + 5) * 12.5))   # 12.5 tokens per second of speech
        kw = {"repetition_penalty": 1.05, "max_new_tokens": tokens, "max_time": 180.0}
        torch.manual_seed(seed)
        if self.speaker:
            wavs, sr = self.model.generate_custom_voice(text=[text], speaker=[self.speaker], language=[self.lang],
                                                        instruct=[self.style or None], **kw)
        else:
            wavs, sr = self.model.generate_voice_clone(text=[text], language=[self.lang], voice_clone_prompt=self.prompt, **kw)
        w = np.asarray(wavs[0], dtype="float32")
        return w, sr, len(w) / sr >= (tokens / 12.5) * 0.97   # True = hit the cap, i.e. it ran away


def speak(a):
    import numpy as np
    import soundfile as sf
    voice_dir = pathlib.Path(a.voice_dir) if a.voice_dir else None
    if not a.speaker and not (voice_dir and (voice_dir / "ref.wav").exists()):
        die(f"{voice_dir}/ref.wav is missing. Run prepare first (tts.py does this for you).")
    text = pathlib.Path(a.text_file).read_text(encoding="utf-8").strip()
    if not text:
        die("the text is empty")
    lang = a.lang.lower()
    device = pick_device(a.device)
    engine = Qwen(a.model, device, lang, voice_dir, a.speaker, a.style)
    chunks = split_text(text, CHUNK_CHARS // 2 if lang in CJK else CHUNK_CHARS)
    seed = a.seed if a.seed is not None else random.randrange(1, 2**31)
    log(f"Speaking {len(text)} characters in {len(chunks)} piece(s), seed {seed}")
    parts, sr_out = [], None
    t0 = time.time()
    for i, chunk in enumerate(chunks, 1):
        for attempt in range(3):
            w, sr, ran_away = engine.speak(chunk, seed + attempt * 7919)
            peak = float(np.abs(w).max()) if len(w) else 0.0
            if not ran_away and peak >= 0.01:
                break
            log(f"  piece {i}: {'ran away' if ran_away else 'silent'}, trying again with another seed")
        else:
            die(f"piece {i} failed 3 times: \"{chunk[:60]}...\". Shorten or reword that sentence, or try --seed.")
        parts.append(w)
        sr_out = sr
        log(f"[{i}/{len(chunks)}] {len(w) / sr:.1f} s of audio ({time.time() - t0:.0f} s elapsed)")
    gap = np.zeros(int(GAP * sr_out), dtype="float32")
    audio = parts[0]
    for p in parts[1:]:
        audio = np.concatenate([audio, gap, p])
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, sr_out, subtype="PCM_16")   # the model's own sample rate (24 kHz), not resampled
    log(f"Done: {out}  ({len(audio) / sr_out:.1f} s, {sr_out // 1000} kHz, seed {seed})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--ref", required=True)
    p.add_argument("--voice-dir", required=True)
    p.add_argument("--ref-text", default="")
    p.add_argument("--lang", default="auto")
    s = sub.add_parser("speak")
    s.add_argument("--model", default="")
    s.add_argument("--voice-dir", default="")
    s.add_argument("--speaker", default="")
    s.add_argument("--style", default="")
    s.add_argument("--text-file", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--lang", default="auto")
    s.add_argument("--seed", type=int)
    s.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    a = ap.parse_args()
    prepare(a) if a.cmd == "prepare" else speak(a)


if __name__ == "__main__":
    main()
