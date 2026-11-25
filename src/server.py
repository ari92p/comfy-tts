"""Voice service: Qwen3-TTS (voice clone + built-in speakers) and Whisper (transcript of a sample) as a small HTTP API.

Started by `python3 tts.py --serve` (inside the Qwen environment of this repo). Same HTTP API as the Studio's voice service.
Restrict who may reach it with a firewall rule; there is no login. One job at a time; a model is loaded on first
use and dropped from the GPU again after IDLE_SECONDS without a request, because the card is shared with ComfyUI.

    GET    /health                      state, loaded models
    GET    /speakers                    built-in speakers (CustomVoice)
    GET    /voices                      saved voices:  [{name, title, text, seconds, created}]
    GET    /voices/{name}/ref           the reference (all enabled samples joined) as WAV
    PATCH  /voices/{name}               json {title}: rename (the name used in URLs and personas stays)
    POST   /voices/{name}/samples       form: audio, text (optional): add a sample; PATCH / DELETE /voices/{name}/samples/{id}; GET .../audio
    PUT    /voices/{name}               form: audio (file), text (optional, else Whisper), title, overwrite=1
    DELETE /voices/{name}
    POST   /transcribe                  form: audio (file), language (optional)  ->  {text, language, seconds}
    POST   /generate                    json: {text, voice | speaker, language, instruct, seed, ...}  ->  audio/wav

Settings found by trial and error with Qwen3-TTS: repetition_penalty 1.05, never do_sample=False (endless
babble), voices are not post-processed. Against hangs every generation gets max_new_tokens (at most 1024) and max_time, and a
watchdog ends the process if a job is still stuck after 4 minutes (`tts.py --serve` then starts it again).
Samples over 16 s are cut to a sentence end at or before 15 s (Whisper word times; the whole recording stays as ref_full.wav).
The clone prompt of each voice is computed once and kept as prompt.pt.
"""

import argparse
import gc
import io
import json
import os
import pathlib
import random
import re
import shutil
import threading
import time
from typing import Optional

# Environment (all optional): VOICE_DIR = where saved voices live (default: voices/ in the repository root; point it at an existing
# folder to keep your voices), HF_HOME = Hugging Face cache, VOICE_OFFLINE=1 = never talk to the Hub (models must be cached).
if os.environ.get("VOICE_OFFLINE") == "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, Response  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from pydantic import BaseModel  # noqa: E402

VOICES = pathlib.Path(os.environ.get("VOICE_DIR") or pathlib.Path(__file__).resolve().parent.parent / "voices")
IDLE_SECONDS = int(os.environ.get("VOICE_IDLE_SECONDS", "300"))
MODELS = {"clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"}
WHISPER = os.environ.get("VOICE_WHISPER", "large-v3-turbo")   # e.g. "small" for a much smaller download
SPEAKERS = [
    {"name": "aiden", "note": "Englisch, Mann (US)"}, {"name": "ryan", "note": "Englisch, Mann"},
    {"name": "dylan", "note": "Chinesisch (Peking), Mann"}, {"name": "eric", "note": "Chinesisch (Sichuan), Mann"},
    {"name": "uncle_fu", "note": "Chinesisch, älterer Mann"}, {"name": "vivian", "note": "Chinesisch, Frau"},
    {"name": "serena", "note": "Chinesisch, Frau"}, {"name": "ono_anna", "note": "Japanisch, Frau"},
    {"name": "sohee", "note": "Koreanisch, Frau"},
]
LANGUAGES = ("Auto", "English", "German", "French", "Spanish", "Italian", "Portuguese", "Russian", "Chinese", "Japanese", "Korean")
WHISPER_LANGS = {"English": "en", "German": "de", "French": "fr", "Spanish": "es", "Italian": "it", "Portuguese": "pt",
                 "Russian": "ru", "Chinese": "zh", "Japanese": "ja", "Korean": "ko"}
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
MIN_REF, MAX_REF = 2.0, 40.0   # seconds of sample audio that are accepted
KEEP_SECONDS, TRIM_ABOVE = 15.0, 16.0   # a longer sample is cut to a sentence end at or before KEEP_SECONDS (5 to 15 s clones best)
MAX_TOKENS = 1024              # at 12.5 tokens per second: without it a missing end-of-speech token runs forever (known Qwen3-TTS problem)
JOB_SECONDS, HARD_SECONDS = 180, 240   # generation stops after JOB_SECONDS (max_time); a job still stuck after HARD_SECONDS ends the process
TEXT_MAX = 1000                # characters; MAX_TOKENS is about 80 s of speech

app = FastAPI(title="Voice")
JOB = threading.Lock()         # one job at a time (GPU shared with ComfyUI)
_models = {}                   # "clone" | "custom" | "whisper" -> loaded model
_used = [time.time()]
_started = [0.0]               # start of the running job (0 = none), watched by the watchdog


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------- models ----------

def device_and_dtype():
    """CUDA exactly as before (bf16); Apple GPU or CPU in float32 so the same server also runs on a Mac."""
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def fetch(repo):
    """Local snapshot path of a model: from the cache if it is there, otherwise downloaded once (unless VOICE_OFFLINE=1)."""
    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception:
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            raise
        log(f"downloading {repo} (first use)")
        return snapshot_download(repo)


def prefetch():
    """Make sure all models are on disk before the service starts listening. A download inside the first request would give
    no feedback and the watchdog (HARD_SECONDS) would end a slow download half way. Already cached models cost nothing."""
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return
    log("checking models (the first start downloads about 9 GB, once; the progress bars below show it)")
    for kind in ("clone", "custom"):
        try:
            fetch(MODELS[kind])
        except Exception as e:  # noqa: BLE001
            log(f"could not download {MODELS[kind]}: {e!r}. It will be tried again on the first request.")
    try:
        from faster_whisper.utils import download_model
        download_model(WHISPER)
    except Exception as e:  # noqa: BLE001
        log(f"could not download the Whisper model {WHISPER}: {e!r}")
    log("models ready")


def model(kind):
    if kind not in _models:
        t = time.time()
        if kind == "whisper":
            from faster_whisper import WhisperModel
            # CPU: CTranslate2 wants CUDA-12 cuBLAS, the venv only has CUDA 13 (torch). A 30 s clip takes seconds anyway.
            _models[kind] = WhisperModel(WHISPER, device="cpu", compute_type="int8")
        else:
            from qwen_tts import Qwen3TTSModel
            # local snapshot path: qwen-tts otherwise queries the Hub API on every load
            device, dtype = device_and_dtype()
            _models[kind] = Qwen3TTSModel.from_pretrained(fetch(MODELS[kind]), device_map=device, dtype=dtype)
        log(f"model {kind} loaded in {time.time() - t:.1f}s")
    return _models[kind]


def unload_all():
    if _models:
        log("unloading", ", ".join(_models))
        _models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def janitor():
    while True:
        time.sleep(20)
        if _models and time.time() - _used[0] > IDLE_SECONDS and JOB.acquire(blocking=False):
            try:
                if time.time() - _used[0] > IDLE_SECONDS:
                    unload_all()
            finally:
                JOB.release()


def watchdog():
    """A job that outlives HARD_SECONDS is stuck (max_time did not stop it): end the process. `tts.py --serve` starts it again."""
    while True:
        time.sleep(5)
        if _started[0] and time.time() - _started[0] > HARD_SECONDS:
            log(f"job stuck for more than {HARD_SECONDS}s, exiting")
            os._exit(3)


class job:
    """`with job():` runs one job; a second request meanwhile gets 503 'busy' instead of queueing up."""

    def __enter__(self):
        if not JOB.acquire(blocking=False):
            raise HTTPException(503, "Der Voice-Dienst ist gerade beschäftigt")
        _started[0] = time.time()
        return self

    def __exit__(self, *exc):
        _started[0] = 0.0
        _used[0] = time.time()
        JOB.release()


# ---------- audio helpers ----------

def read_audio(raw):
    """Upload -> mono float32 + sample rate. soundfile reads wav, flac, ogg and mp3."""
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception:
        raise HTTPException(400, "Die Audiodatei konnte nicht gelesen werden (WAV, MP3, FLAC oder OGG)")
    return data.mean(axis=1), sr


def check_ref(audio, sr):
    sec = len(audio) / sr
    if sec < MIN_REF:
        raise HTTPException(400, f"Das Beispiel ist zu kurz ({sec:.1f} s), mindestens {MIN_REF:.0f} s")
    if sec > MAX_REF:
        raise HTTPException(400, f"Das Beispiel ist zu lang ({sec:.0f} s), höchstens {MAX_REF:.0f} s")
    if float(np.abs(audio).max()) < 0.01:
        raise HTTPException(400, "Das Beispiel ist stumm")
    return sec


def transcribe_audio(audio, sr, language=None):
    """-> (text, detected language, words [{word, start, end}])"""
    from torchaudio.functional import resample
    wav = resample(torch.from_numpy(audio), sr, 16000).numpy()
    segments, info = model("whisper").transcribe(wav, language=WHISPER_LANGS.get(language or ""), beam_size=5, word_timestamps=True)
    words = [{"word": w.word, "start": w.start, "end": w.end} for s in segments for w in s.words]
    return "".join(w["word"] for w in words).strip(), info.language, words


def fit_sample(audio, sr, words):
    """A sample over TRIM_ABOVE seconds is cut at a sentence end (Whisper word times) between 8 s and KEEP_SECONDS, else at the latest
    pause before KEEP_SECONDS; the text is exactly what is said in the excerpt. -> (audio, text, cut seconds or None when nothing was cut)"""
    dur = len(audio) / sr
    if dur <= TRIM_ABOVE:
        return audio, None, None
    cand = [i for i, w in enumerate(words) if 5.0 <= w["end"] <= KEEP_SECONDS]
    ends = [i for i in cand if re.search(r"[.!?…][\"')\]]*$", words[i]["word"].strip())]
    gap = lambda i: (words[i + 1]["start"] if i + 1 < len(words) else dur) - words[i]["end"]
    ends = [i for i in ends if words[i]["end"] >= 8.0]
    if ends:
        k = ends[-1]
    elif cand:   # no sentence end worth cutting at (spoken without punctuation): the latest real pause before KEEP_SECONDS
        pauses = [i for i in cand if gap(i) >= 0.25]
        k = (pauses or cand)[-1]
    else:
        raise HTTPException(422, "Im Beispiel wurden zu wenige Wörter erkannt, um es zu kürzen")
    cut = words[k]["end"] + 0.25
    if k + 1 < len(words):
        cut = min(cut, (words[k]["end"] + words[k + 1]["start"]) / 2)
    cut = min(cut, dur)
    return audio[: int(cut * sr)], "".join(w["word"] for w in words[: k + 1]).strip(), round(cut, 2)


def to_wav(audio, sr):
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------- voices ----------

def voice_dir(name):
    if not NAME_RE.match(name):
        raise HTTPException(400, "Name: Kleinbuchstaben, Ziffern, - und _ (höchstens 40 Zeichen)")
    return VOICES / name


def voice_info(d):
    ref, txt = d / "ref.wav", d / "ref.txt"
    if not (ref.is_file() and txt.is_file()):
        return None
    meta = {}
    if (d / "meta.json").is_file():
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    info = sf.info(str(ref))
    return {"name": d.name, "title": meta.get("title") or d.name, "text": txt.read_text(encoding="utf-8").strip(),
            "seconds": round(info.frames / info.samplerate, 1), "created": meta.get("created") or ref.stat().st_mtime,
            "trimmed": (d / "ref_full.wav").is_file(), "prompt": (d / "prompt.pt").is_file(), "samples": load_samples(d)}


# A voice can keep several samples, but exactly one is the template that is cloned from (ref.wav + ref.txt, which the pipeline scripts
# read as well): the sample with "on": true in samples.json. The others are alternatives (and material for later fine-tuning).
SR_REF = 24000


def load_samples(d):
    f = d / "samples.json"
    if f.is_file():
        return json.loads(f.read_text(encoding="utf-8"))
    ref = d / "ref.wav"   # voice from before samples existed: its reference is its only sample
    return [{"id": "s1", "text": (d / "ref.txt").read_text(encoding="utf-8").strip(), "seconds": round(sf.info(str(ref)).duration, 1), "on": True}]


def ensure_samples(d):
    if not (d / "samples.json").is_file():
        items = load_samples(d)
        (d / "samples").mkdir(exist_ok=True)
        shutil.copyfile(d / "ref.wav", d / "samples" / "s1.wav")
        save_samples(d, items)


def save_samples(d, items):
    (d / "samples.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")


def to_ref_rate(audio, sr):
    if sr == SR_REF:
        return audio
    from torchaudio.functional import resample
    return resample(torch.from_numpy(audio), sr, SR_REF).numpy()


def fix_text(t):
    """UTF-8 text that was read as Latin-1 somewhere ("fÃ¼r") back to normal."""
    if "Ã" in t or "Â" in t:
        try:
            return t.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return t


def main_sample(items):
    on = [x for x in items if x["on"]]
    if len(on) != 1:
        raise HTTPException(400, "Genau ein Beispiel muss als Vorlage gewählt sein")
    return on[0]


def set_main(items, sid):
    for x in items:
        x["on"] = x["id"] == sid


def write_ref(d, items):
    """ref.wav / ref.txt = the one sample chosen as the template. Several speakers or languages in one template make the voice
    wobble in our tests, so samples are never joined; the others are kept as alternatives."""
    x = main_sample(items)
    a, sr = sf.read(str(d / "samples" / f"{x['id']}.wav"), dtype="float32", always_2d=True)
    (d / "ref.wav").write_bytes(to_wav(to_ref_rate(a.mean(axis=1), sr), SR_REF))
    (d / "ref.txt").write_text(x["text"].strip(), encoding="utf-8")
    (d / "prompt.pt").unlink(missing_ok=True)


def rebuild_ref(d, items):
    """write_ref plus the clone prompt (needs the clone model: call inside a job)."""
    write_ref(d, items)
    voice_prompt(d)


def tidy_voices():
    """At start: repair texts with broken umlauts and make sure every voice has exactly one template sample."""
    for d in VOICES.glob("*"):
        if not (d / "ref.wav").is_file() or not (d / "ref.txt").is_file():
            continue
        try:
            txt = (d / "ref.txt").read_text(encoding="utf-8")
            if fix_text(txt) != txt:
                (d / "ref.txt").write_text(fix_text(txt), encoding="utf-8")
                (d / "prompt.pt").unlink(missing_ok=True)
                log(f"{d.name}: ref.txt umlauts repaired")
            if (d / "samples.json").is_file():
                items = load_samples(d)
                changed = False
                for x in items:
                    if fix_text(x["text"]) != x["text"]:
                        x["text"], changed = fix_text(x["text"]), True
                on = [x for x in items if x["on"]]
                if len(on) != 1:   # older voices joined several samples: keep the newest one that was on
                    set_main(items, (on or items)[-1]["id"])
                    changed = True
                if changed:
                    save_samples(d, items)
                    write_ref(d, items)
                    log(f"{d.name}: template sample set to {main_sample(items)['id']}")
        except Exception as e:  # noqa: BLE001
            log(f"{d.name}: tidy failed: {e!r}")


def voice_prompt(d):
    """The clone prompt of a voice (speaker embedding + reference codes), computed once and kept in prompt.pt.
    Needs the clone model loaded, so call it inside a job."""
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
    m, pf = model("clone"), d / "prompt.pt"
    if not (pf.is_file() and pf.stat().st_mtime >= max((d / "ref.wav").stat().st_mtime, (d / "ref.txt").stat().st_mtime)):
        items = m.create_voice_clone_prompt(ref_audio=str(d / "ref.wav"), ref_text=(d / "ref.txt").read_text(encoding="utf-8").strip())
        torch.save([{"ref_code": None if it.ref_code is None else it.ref_code.cpu(), "ref_spk_embedding": it.ref_spk_embedding.cpu(),
                     "x_vector_only_mode": it.x_vector_only_mode, "icl_mode": it.icl_mode, "ref_text": it.ref_text} for it in items], pf)
        log(f"prompt computed: {d.name}")
    raw = torch.load(pf, map_location="cpu", weights_only=True)
    to = lambda t: None if t is None else t.to(m.device)
    return [VoiceClonePromptItem(ref_code=to(r["ref_code"]), ref_spk_embedding=to(r["ref_spk_embedding"]), x_vector_only_mode=r["x_vector_only_mode"],
                                 icl_mode=r["icl_mode"], ref_text=r["ref_text"]) for r in raw]


@app.get("/health")
def health():
    return {"ok": True, "busy": JOB.locked(), "loaded": sorted(_models), "idle_seconds": IDLE_SECONDS,
            "voices": sum(1 for d in VOICES.glob("*") if voice_info(d)) if VOICES.is_dir() else 0,
            "gpu": torch.cuda.is_available() or torch.backends.mps.is_available()}


@app.get("/speakers")
def speakers():
    return {"speakers": SPEAKERS, "languages": list(LANGUAGES)}


@app.get("/voices")
def voices():
    VOICES.mkdir(parents=True, exist_ok=True)
    out = [v for v in (voice_info(d) for d in sorted(VOICES.iterdir()) if d.is_dir() and not d.name.startswith("_")) if v]
    return {"voices": out}


@app.get("/voices/{name}/ref")
def voice_ref(name: str):
    p = voice_dir(name) / "ref.wav"
    if not p.is_file():
        raise HTTPException(404, "Voice nicht gefunden")
    return FileResponse(p, media_type="audio/wav")


@app.put("/voices/{name}")
def voice_save(name: str, audio: UploadFile = File(...), text: str = Form(""), title: str = Form(""), overwrite: str = Form("")):
    d = voice_dir(name)
    if d.exists() and not overwrite:
        raise HTTPException(409, "Diesen Namen gibt es schon")
    samples, sr = read_audio(audio.file.read())
    sec = check_ref(samples, sr)
    text = fix_text(text.strip())
    with job():
        full = None
        if d.exists():   # overwrite: start over with the new recording
            (d / "samples.json").unlink(missing_ok=True)
            shutil.rmtree(d / "samples", ignore_errors=True)
            (d / "ref_full.wav").unlink(missing_ok=True)
        if sec > TRIM_ABOVE:   # too long to clone well: keep one clean excerpt, remember the whole recording
            _, _, words = transcribe_audio(samples, sr)
            full = samples
            samples, text, cut = fit_sample(samples, sr, words)
            sec = len(samples) / sr
        elif not text:
            text, _, _ = transcribe_audio(samples, sr)
        if not text:
            raise HTTPException(422, "Es wurde kein Text erkannt, bitte die Abschrift selbst eintragen")
        d.mkdir(parents=True, exist_ok=True)
        (d / "ref.wav").write_bytes(to_wav(samples, sr))
        (d / "ref.txt").write_text(text, encoding="utf-8")
        if full is not None:
            (d / "ref_full.wav").write_bytes(to_wav(full, sr))
        (d / "meta.json").write_text(json.dumps({"title": title.strip() or name, "created": time.time()}, ensure_ascii=False), encoding="utf-8")
        ensure_samples(d)
        voice_prompt(d)
    log(f"voice saved: {name} ({sec:.1f}s)")
    return voice_info(d)


def own_voice(name):
    d = voice_dir(name)
    if not voice_info(d):
        raise HTTPException(404, "Voice nicht gefunden")
    return d


class Rename(BaseModel):
    title: str


@app.patch("/voices/{name}")
def voice_rename(name: str, r: Rename):
    d = own_voice(name)
    title = r.title.strip()
    if not title:
        raise HTTPException(400, "Bitte einen Namen eingeben")
    f = d / "meta.json"
    meta = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    meta["title"] = title
    f.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return voice_info(d)


@app.post("/voices/{name}/samples")
def sample_add(name: str, audio: UploadFile = File(...), text: str = Form("")):
    d = own_voice(name)
    samples, sr = read_audio(audio.file.read())
    sec = check_ref(samples, sr)
    text = fix_text(text.strip())
    with job():
        if sec > TRIM_ABOVE:
            _, _, words = transcribe_audio(samples, sr)
            samples, text, _ = fit_sample(samples, sr, words)
        elif not text:
            text, _, _ = transcribe_audio(samples, sr)
        if not text:
            raise HTTPException(422, "Es wurde kein Text erkannt, bitte die Abschrift selbst eintragen")
        ensure_samples(d)
        items = load_samples(d)
        sid = "s%d" % (max(int(x["id"][1:]) for x in items) + 1)
        audio24 = to_ref_rate(samples, sr)
        (d / "samples" / f"{sid}.wav").write_bytes(to_wav(audio24, SR_REF))
        items.append({"id": sid, "text": fix_text(text), "seconds": round(len(audio24) / SR_REF, 1), "on": False})   # an alternative until chosen as template
        save_samples(d, items)
    log(f"sample added: {name}/{sid}")
    return voice_info(d)


class SampleEdit(BaseModel):
    on: Optional[bool] = None
    text: Optional[str] = None


@app.patch("/voices/{name}/samples/{sid}")
def sample_edit(name: str, sid: str, e: SampleEdit):
    d = own_voice(name)
    with job():
        ensure_samples(d)
        items = load_samples(d)
        x = next((i for i in items if i["id"] == sid), None)
        if not x:
            raise HTTPException(404, "Beispiel nicht gefunden")
        was_main = x["on"]
        if e.on is False and was_main:
            raise HTTPException(400, "Wähle stattdessen ein anderes Beispiel als Vorlage")
        if e.on:
            set_main(items, sid)
        if e.text is not None:
            if not e.text.strip():
                raise HTTPException(400, "Die Abschrift darf nicht leer sein")
            x["text"] = fix_text(e.text.strip())
        save_samples(d, items)
        if x["on"]:   # the template changed (another sample or its text)
            rebuild_ref(d, items)
    return voice_info(d)


@app.delete("/voices/{name}/samples/{sid}")
def sample_delete(name: str, sid: str):
    d = own_voice(name)
    with job():
        ensure_samples(d)
        items = load_samples(d)
        keep = [i for i in items if i["id"] != sid]
        if len(keep) == len(items):
            raise HTTPException(404, "Beispiel nicht gefunden")
        if not keep:
            raise HTTPException(400, "Das letzte Beispiel kann nicht gelöscht werden. Lösche die Voice")
        was_main = any(i["id"] == sid and i["on"] for i in items)
        if was_main:
            set_main(keep, keep[0]["id"])
        (d / "samples" / f"{sid}.wav").unlink(missing_ok=True)
        save_samples(d, keep)
        if was_main:
            rebuild_ref(d, keep)
    return voice_info(d)


@app.get("/voices/{name}/samples/{sid}/audio")
def sample_audio(name: str, sid: str):
    d = own_voice(name)
    p = d / "samples" / f"{sid}.wav"
    if not re.fullmatch(r"s\d+", sid) or not (p.is_file() or (sid == "s1" and not (d / "samples.json").is_file())):
        raise HTTPException(404, "Beispiel nicht gefunden")
    return FileResponse(p if p.is_file() else d / "ref.wav", media_type="audio/wav")


@app.delete("/voices/{name}")
def voice_delete(name: str):
    d = voice_dir(name)
    if not d.is_dir():
        raise HTTPException(404, "Voice nicht gefunden")
    shutil.rmtree(d)
    log(f"voice deleted: {name}")
    return {"ok": True}


@app.post("/transcribe")
def transcribe(audio: UploadFile = File(...), language: str = Form("")):
    samples, sr = read_audio(audio.file.read())
    sec = check_ref(samples, sr)
    with job():
        text, lang, words = transcribe_audio(samples, sr, language)
        _, cut_text, cut = fit_sample(samples, sr, words)
    return {"text": cut_text or text, "language": lang, "seconds": round(sec, 1), "keep": cut}


# ---------- generate ----------

class Gen(BaseModel):
    text: str
    voice: str = ""              # saved voice (clone) ...
    speaker: str = ""            # ... or built-in speaker
    language: str = "English"
    instruct: str = ""           # speaking style for built-in speakers
    seed: int | None = None
    repetition_penalty: float = 1.05
    temperature: float | None = None
    subtalker_temperature: float | None = None


@app.post("/generate")
def generate(g: Gen):
    text = g.text.strip()
    if not text:
        raise HTTPException(400, "Kein Text")
    if len(text) > TEXT_MAX:
        raise HTTPException(400, f"Der Text ist zu lang (höchstens {TEXT_MAX} Zeichen)")
    if g.language not in LANGUAGES:
        raise HTTPException(400, "Unbekannte Sprache")
    if bool(g.voice) == bool(g.speaker):
        raise HTTPException(400, "Entweder eine Voice oder einen Sprecher angeben")
    if g.speaker and g.speaker not in {s["name"] for s in SPEAKERS}:
        raise HTTPException(400, "Unbekannter Sprecher")
    d = None
    if g.voice:
        d = voice_dir(g.voice)
        if not voice_info(d):
            raise HTTPException(404, "Voice nicht gefunden")

    # cap against endless babble: ~0.2 s per character is far more than any real reading speed, never more than MAX_TOKENS
    tokens = min(MAX_TOKENS, int(max(8.0, len(text) * 0.2 + 5) * 12.5))
    cap_s = tokens / 12.5
    kw = {"repetition_penalty": g.repetition_penalty, "max_new_tokens": tokens, "max_time": float(JOB_SECONDS)}
    for k in ("temperature", "subtalker_temperature"):
        if getattr(g, k) is not None:
            kw[k] = getattr(g, k)
    seed = g.seed if g.seed is not None else random.randrange(1, 2**31)
    lang = g.language
    t0 = time.time()
    with job():
        torch.manual_seed(seed)
        if d:
            m = model("clone")
            wavs, sr = m.generate_voice_clone(text=[text], language=[lang], voice_clone_prompt=voice_prompt(d), **kw)
        else:
            wavs, sr = model("custom").generate_custom_voice(text=[text], speaker=[g.speaker], language=[lang],
                                                             instruct=[g.instruct.strip() or None], **kw)
    w = np.asarray(wavs[0], dtype="float32")
    seconds = len(w) / sr
    if time.time() - t0 >= JOB_SECONDS - 1:
        raise HTTPException(422, f"Die Erzeugung wurde nach {JOB_SECONDS} Sekunden abgebrochen. Text kürzen oder neu versuchen")
    if seconds >= cap_s * 0.97:
        raise HTTPException(422, "Die Erzeugung ist aus dem Ruder gelaufen (zu lang). Text kürzen oder neu versuchen")
    peak = float(np.abs(w).max()) if len(w) else 0.0
    if peak < 0.01:
        raise HTTPException(422, "Es kam kein Ton heraus. Bitte neu versuchen")
    log(f"generate {'voice ' + g.voice if d else 'speaker ' + g.speaker}: {len(text)} chars -> {seconds:.1f}s in {time.time() - t0:.1f}s (seed {seed})")
    return Response(to_wav(w, sr), media_type="audio/wav",
                    headers={"X-Seconds": f"{seconds:.2f}", "X-Seed": str(seed), "X-Peak": f"{peak:.3f}",
                             "X-Elapsed": f"{time.time() - t0:.1f}"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8190)
    a = ap.parse_args()
    VOICES.mkdir(parents=True, exist_ok=True)
    tidy_voices()
    prefetch()
    threading.Thread(target=janitor, daemon=True).start()
    threading.Thread(target=watchdog, daemon=True).start()
    log(f"voice service on {a.host}:{a.port}, voices in {VOICES}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
