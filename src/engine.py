#!/usr/bin/env python3
"""The part of tts that runs inside a model's virtualenv. You normally don't call this directly, use tts.py.

  engine.py prepare --ref FILE --voice-dir DIR [--ref-text T] [--lang de]
      Turns any audio file into a clean 24 kHz mono reference (<= 15 s) in DIR/ref.wav and writes the
      transcript DIR/ref.txt (given by --ref-text, or heard by Whisper).
  engine.py transcribe --file AUDIO --out-file TEXT.txt [--lang de]
      Writes what is said in an audio file (Whisper).
  engine.py speak --model REPO --voice-dir DIR --text-file F --out OUT.wav [--lang de] [--seed N]
      Reads DIR/ref.wav and speaks the text in that cloned voice.

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
BATCH = 4              # pieces generated in one go (about twice as fast as one after the other)
MIN_MATCH = 0.75       # how well the words of a piece must match the text when Whisper listens to it again (1.0 = identical)
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


_whisper = []


def whisper_model():
    if not _whisper:
        from faster_whisper import WhisperModel
        _whisper.append(WhisperModel(os.environ.get("VOICE_WHISPER", "large-v3-turbo"), device="cpu", compute_type="int8"))
    return _whisper[0]


def transcribe(y, lang, quiet=False):
    """-> (text, words) with word times in seconds (faster-whisper, CPU, so it never competes with the TTS model for VRAM)."""
    import librosa
    if not quiet:
        log("Listening to the reference (Whisper)...")
    y16 = librosa.resample(y, orig_sr=REF_SR, target_sr=16000)
    segments, _ = whisper_model().transcribe(y16, language=None if lang in ("", "auto") else lang, beam_size=5, word_timestamps=True)
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
    return pause or max((w.end for w in words if w.end <= hi), default=hi)   # never in the middle of a word


def prepare(a):
    import librosa
    import soundfile as sf
    voice_dir = pathlib.Path(a.voice_dir)
    voice_dir.mkdir(parents=True, exist_ok=True)
    y = load_audio(a.ref)
    full = len(y)
    y, idx = librosa.effects.trim(y, top_db=35)
    clipped = (full - idx[1]) < 0.1 * REF_SR      # speech runs right up to the end of the file: the recording was probably cut off
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
        text, words = transcribe(y, a.lang)
        if clipped and len(words) > 3 and words[-1].end >= len(y) / REF_SR - 0.2:
            # the last word is a fragment (the file ends in the middle of it): drop it, and the pieces of it, and end the recording before it
            words = words[:-1]
            while len(words) > 3 and (not any(c.isalnum() for c in words[-1].word) or words[-1].end - words[-1].start < 0.12 or words[-1].end >= len(y) / REF_SR - 0.2):
                words = words[:-1]
            y = y[: int((words[-1].end + 0.2) * REF_SR)]
            text = "".join(w.word for w in words).strip()
            log("The recording ended in the middle of a word, cut it off before that word.")
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
    """Qwen3-TTS Base model: speaks with the cloned voice in `voice_dir`."""

    def __init__(self, repo, device, lang, voice_dir):
        import torch
        from qwen_tts import Qwen3TTSModel
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32   # bf16 is only safe on CUDA
        log(f"Loading {repo} on {device} (the first run downloads it)...")
        t = time.time()
        self.model = Qwen3TTSModel.from_pretrained(repo, device_map=device, dtype=dtype)
        log(f"Model loaded in {time.time() - t:.0f} s")
        self.lang = QWEN_LANGS.get(lang, "Auto")
        self.per_char = 0.5 if lang in CJK else 0.2
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

    def speak_many(self, texts, seed):
        """Several pieces in one batch -> [(wave, sample rate, ran_away)]. A piece 'ran away' when it used up almost all the tokens it may have."""
        import numpy as np
        import torch
        caps = [min(1024, int(max(8.0, len(t) * self.per_char + 5) * 12.5)) for t in texts]   # 12.5 tokens per second of speech
        kw = {"repetition_penalty": 1.05, "max_new_tokens": max(caps), "max_time": 180.0}
        torch.manual_seed(seed)
        n = len(texts)
        wavs, sr = self.model.generate_voice_clone(text=list(texts), language=[self.lang] * n, voice_clone_prompt=self.prompt * n, **kw)
        out = []
        for w, cap in zip(wavs, caps):
            w = np.asarray(w, dtype="float32")
            out.append((w, sr, len(w) / sr >= (cap / 12.5) * 0.97))
        return out

    def speak(self, text, seed):
        return self.speak_many([text], seed)[0]


def change_speed(audio, sr, speed):
    """Speak faster (>1) or slower (<1) without changing the pitch: ffmpeg's atempo if ffmpeg is installed, otherwise librosa."""
    import numpy as np
    import soundfile as sf
    if abs(speed - 1.0) < 0.01:
        return audio
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            src, dst = pathlib.Path(d) / "in.wav", pathlib.Path(d) / "out.wav"
            sf.write(str(src), audio, sr, subtype="FLOAT")
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-filter:a", f"atempo={speed:.4f}", str(dst)], check=True)
            out, _ = sf.read(str(dst), dtype="float32")
        return np.asarray(out, dtype="float32")
    import librosa
    return librosa.effects.time_stretch(audio, rate=speed).astype("float32")


def many_safe(engine, texts, seed):
    """speak_many, one piece after the other if the batch does not fit (for example out of memory)."""
    try:
        return engine.speak_many(texts, seed)
    except Exception as e:  # noqa: BLE001
        if len(texts) == 1:
            raise
        log(f"  batch of {len(texts)} failed ({type(e).__name__}), going one by one")
        return [engine.speak(t, seed + i) for i, t in enumerate(texts)]


def similarity(text, heard, lang):
    """How well what Whisper heard matches the text, 0 to 1. Extra or repeated words count against it."""
    import difflib
    import re as _re
    if lang in CJK:
        x, y = list(_re.sub(r"[\s\W]", "", text)), list(_re.sub(r"[\s\W]", "", heard))
    else:
        x, y = _re.sub(r"[^\w ]", "", text.lower()).split(), _re.sub(r"[^\w ]", "", heard.lower()).split()
    if not x:
        return 1.0
    r = difflib.SequenceMatcher(None, x, y).ratio()
    return min(r, 0.5) if len(y) > len(x) * 1.3 + 1 else r


def broken(r):
    import numpy as np
    return r[2] or not len(r[0]) or float(np.abs(r[0]).max()) < 0.01


def listen_back(results, chunks, lang):
    """Let Whisper listen to each piece again: -> one score per piece (a broken piece scores 0)."""
    return [0.0 if broken(r) else similarity(c, transcribe(r[0], lang, quiet=True)[0], lang) for r, c in zip(results, chunks)]


def transcribe_file(a):
    text, _ = transcribe(load_audio(a.file), a.lang, quiet=True)
    pathlib.Path(a.out_file).write_text(text + "\n", encoding="utf-8")


def speak(a):
    import numpy as np
    import soundfile as sf
    voice_dir = pathlib.Path(a.voice_dir) if a.voice_dir else None
    if not (voice_dir and (voice_dir / "ref.wav").exists()):
        die(f"{voice_dir}/ref.wav is missing. Run prepare first (tts.py does this for you).")
    text = pathlib.Path(a.text_file).read_text(encoding="utf-8").strip()
    if not text:
        die("the text is empty")
    lang = a.lang.lower()
    device = pick_device(a.device)
    engine = Qwen(a.model, device, lang, voice_dir)
    chunks = split_text(text, CHUNK_CHARS // 2 if lang in CJK else CHUNK_CHARS)
    seed = a.seed if a.seed is not None else random.randrange(1, 2**31)
    log(f"Speaking {len(text)} characters in {len(chunks)} piece(s), seed {seed}")
    n = len(chunks)
    t0 = time.time()
    results = [None] * n
    for start in range(0, n, BATCH):                      # 1) all pieces, a few at a time
        idx = list(range(start, min(n, start + BATCH)))
        for i, r in zip(idx, many_safe(engine, [chunks[i] for i in idx], seed + start)):
            results[i] = r
        log(f"[{idx[-1] + 1}/{n}] {time.time() - t0:.0f} s elapsed")
    scores = listen_back(results, chunks, lang) if a.check else [0.0 if broken(r) else 1.0 for r in results]   # 2) does it say what was written?
    for attempt in range(1, 4):                           # 3) pieces that do not are made again, together, with another seed
        bad = [i for i in range(n) if scores[i] < MIN_MATCH]
        if not bad:
            break
        log(f"  {len(bad)} piece(s) do not match the text (best {max(scores[i] for i in bad):.2f}), making them again (try {attempt})")
        new = many_safe(engine, [chunks[i] for i in bad], seed + attempt * 7919)
        new_scores = listen_back(new, [chunks[i] for i in bad], lang) if a.check else [0.0 if broken(r) else 1.0 for r in new]
        improved = False
        for i, r, sc in zip(bad, new, new_scores):
            if sc > scores[i] + 0.02:
                results[i], scores[i], improved = r, sc, True   # keep the try that matches the text best
        if not improved:                                   # the same score again: more tries will not help (often Whisper's own slip)
            break
    for i, r in enumerate(results):
        if broken(r):
            die(f"piece {i + 1} failed 4 times: \"{chunks[i][:60]}...\". Shorten or reword that sentence, or try --seed.")
        if scores[i] < MIN_MATCH:
            log(f"  piece {i + 1} still differs from the text (match {scores[i]:.2f}), keeping the best try")
    parts = [r[0] for r in results]
    sr_out = results[0][1]
    gap = np.zeros(int(GAP * sr_out), dtype="float32")
    audio = parts[0]
    for p in parts[1:]:
        audio = np.concatenate([audio, gap, p])
    audio = change_speed(audio, sr_out, a.speed)
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
    t = sub.add_parser("transcribe")
    t.add_argument("--file", required=True)
    t.add_argument("--out-file", required=True)
    t.add_argument("--lang", default="auto")
    s = sub.add_parser("speak")
    s.add_argument("--model", default="")
    s.add_argument("--voice-dir", required=True)
    s.add_argument("--no-check", dest="check", action="store_false")
    s.add_argument("--speed", type=float, default=1.0)
    s.add_argument("--text-file", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--lang", default="auto")
    s.add_argument("--seed", type=int)
    s.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    a = ap.parse_args()
    {"prepare": prepare, "transcribe": transcribe_file, "speak": speak}[a.cmd](a)


if __name__ == "__main__":
    main()
