#!/usr/bin/env python3
"""tts: text to speech on your own computer, with built-in speakers or a voice cloned from a short recording.

  python3 tts.py                                    guided mode: asks a few questions, helps you pick a model
  python3 tts.py --speaker ryan --text "Hello!"     built-in speaker, one command, no questions
  python3 tts.py --clone me.wav --text "Hello!"     your own cloned voice (writes out/me-<time>.wav)

Run with --help for all options. This file only needs a plain Python 3.9+; the models run in private virtualenvs
(.venvs/) that are created for you on first use, see README.md.
"""
import argparse
import datetime
import os
import pathlib
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "src"
VENVS = ROOT / ".venvs"
VOICES = ROOT / "voices"
OUT = ROOT / "out"
IS_WIN = os.name == "nt"

# Everything the launcher knows about the models. `langs` are ISO codes. `python` is the interpreter uv fetches for the venv.
QWEN_LANGS = ["en", "de", "fr", "es", "it", "pt", "ru", "zh", "ja", "ko"]
MODELS = {
    "qwen3-tts-1.7b": {
        "backend": "qwen", "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "speaker_repo": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "langs": QWEN_LANGS, "python": "3.12",
        "title": "Qwen3-TTS 1.7B",
        "note": "the largest Qwen3-TTS model: built-in speakers and cloning, 10 languages; needs a GPU (NVIDIA 8+ GB or an Apple Silicon Mac)",
    },
}
# Built-in speakers of the Qwen3-TTS CustomVoice models. Each is native in one language and speaks the others with an accent.
SPEAKERS = [
    ("aiden", "English (US), man"), ("ryan", "English, man"), ("vivian", "Chinese, woman"), ("serena", "Chinese, woman"),
    ("uncle_fu", "Chinese, older man"), ("dylan", "Chinese (Beijing), man"), ("eric", "Chinese (Sichuan), man"),
    ("ono_anna", "Japanese, woman"), ("sohee", "Korean, woman"),
]
DEFAULT_SPEAKER = {"en": "aiden", "zh": "vivian", "ja": "ono_anna", "ko": "sohee"}
LANG_NAMES = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
              "ru": "Russian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean"}


def say(*a):
    print(*a, flush=True)


def die(msg):
    print(f"\nerror: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


# ---------- hardware ----------

def detect_hardware():
    """-> {kind: cuda|mps|cpu, name, mem_gb, cc}. Looks, never installs anything."""
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=15)
            name, mem, cc = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
            return {"kind": "cuda", "name": name, "mem_gb": round(float(mem) / 1024), "cc": float(cc)}
        except Exception:
            pass
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip())
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        except Exception:
            mem, chip = 0, "Apple Silicon"
        return {"kind": "mps", "name": chip or "Apple Silicon", "mem_gb": round(mem / 2**30), "cc": 0.0}
    return {"kind": "cpu", "name": platform.processor() or platform.machine(), "mem_gb": 0, "cc": 0.0}


def describe(hw):
    if hw["kind"] == "cuda":
        return f"{hw['name']}, {hw['mem_gb']} GB VRAM, runs on the GPU (CUDA)"
    if hw["kind"] == "mps":
        return f"{hw['name']}, {hw['mem_gb']} GB memory, runs on the Apple GPU (MPS)"
    return "no supported GPU found, runs on the CPU (works, but expect minutes per sentence)"


def recommend(lang, hw):
    """The model to use for this language and hardware. `lang` may be 'auto'."""
    return "qwen3-tts-1.7b"


def supports(model, lang):
    return lang in ("auto", "") or lang in MODELS[model]["langs"]


# ---------- virtualenvs ----------

def env_python(backend):
    return VENVS / backend / ("Scripts/python.exe" if IS_WIN else "bin/python")


def env_ready(backend):
    return (VENVS / backend / ".tts-ready").exists()


def run(cmd, **kw):
    return subprocess.run(cmd, **kw)


def ensure_env(model, hw, assume_yes):
    """Create the private virtualenv for this model's backend on first use."""
    backend = MODELS[model]["backend"]
    if env_ready(backend):
        return
    uv = shutil.which("uv")
    if not uv:
        die("`uv` is needed once to set up the Python environment (it also fetches the right Python version).\n"
            "  macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh      (or: brew install uv)\n"
            "  Windows:     powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"\n"
            "Then run this command again.")
    say(f"\nFirst use of {MODELS[model]['title']}: setting up a private Python environment in .venvs/{backend}")
    say("This downloads PyTorch and the model libraries (several GB) and takes a few minutes. It happens once.")
    if not assume_yes and sys.stdin.isatty():
        if ask("Go ahead?", "y").lower() not in ("y", "yes", "j", "ja"):
            die("cancelled")
    venv = VENVS / backend
    if venv.exists():
        shutil.rmtree(venv)     # a half-finished earlier attempt
    py = str(env_python(backend))
    run([uv, "venv", "--python", MODELS[model]["python"], str(venv)], check=True)
    reqs = ["-r", str(ROOT / "requirements" / f"{backend}.txt")]
    nvidia = hw["kind"] == "cuda"
    if backend == "qwen" and nvidia and IS_WIN:
        # on Windows the default PyPI torch is CPU-only
        run([uv, "pip", "install", "--python", py, "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu128"], check=True)
    run([uv, "pip", "install", "--python", py] + reqs, check=True)
    (venv / ".tts-ready").write_text("ok\n")
    say("Environment ready.\n")


def engine(model, args):
    """Run engine.py in the model's virtualenv and stream its output."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTORCH_ENABLE_MPS_FALLBACK="1",
               TOKENIZERS_PARALLELISM="false")
    cmd = [str(env_python(MODELS[model]["backend"])), str(SRC / "engine.py")] + args
    r = run(cmd, env=env)
    if r.returncode:
        sys.exit(r.returncode)


# ---------- small helpers ----------

def ask(prompt, default=""):
    try:
        s = input(f"{prompt} " + (f"[{default}] " if default else "")).strip()
    except (EOFError, KeyboardInterrupt):
        say("")
        sys.exit(130)
    return s or default


def clean_path(s):
    """Accept a path as typed, quoted, or dragged into the terminal window."""
    s = s.strip()
    if len(s) > 1 and s[0] == s[-1] and s[0] in "'\"":
        s = s[1:-1]
    elif not IS_WIN:
        s = re.sub(r"\\(.)", r"\1", s)
    return str(pathlib.Path(os.path.expanduser(s)))


def voice_name(s):
    return re.sub(r"[^a-z0-9_-]+", "-", s.lower()).strip("-")[:40]


def saved_voices():
    if not VOICES.exists():
        return []
    return sorted(d.name for d in VOICES.iterdir() if (d / "ref.wav").exists() and not d.name.startswith("_"))


def shell_join(argv):
    return subprocess.list2cmdline(argv) if IS_WIN else " ".join(shlex.quote(a) for a in argv)


def launcher():
    return "python" if IS_WIN else "python3"


def rel(p):
    """Show paths relative to the current directory where that is shorter."""
    try:
        return os.path.relpath(p)
    except ValueError:
        return str(p)


# ---------- the actual work ----------

def prepare_voice(model, voice_dir, ref, ref_text, lang):
    args = ["prepare", "--ref", ref, "--voice-dir", str(voice_dir), "--lang", lang]
    if ref_text:
        args += ["--ref-text", ref_text]
    engine(model, args)


def speak(model, voice_dir, text, out, lang, seed, device, speaker="", style=""):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        tmp = f.name
    try:
        args = ["speak", "--text-file", tmp, "--out", str(out), "--lang", lang, "--device", device]
        if speaker:
            args += ["--model", MODELS[model]["speaker_repo"], "--speaker", speaker, "--style", style]
        else:
            args += ["--model", MODELS[model]["repo"], "--voice-dir", str(voice_dir)]
        if seed is not None:
            args += ["--seed", str(seed)]
        engine(model, args)
    finally:
        os.unlink(tmp)


def final_output(out, model):
    """-> (wav to generate, final file). engine.py always writes a WAV; other extensions are converted with ffmpeg afterwards."""
    out = pathlib.Path(out)
    if out.suffix.lower() == ".wav":
        return out, out
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        die(f"{out.suffix} output needs ffmpeg (not found). Use a .wav name, or install ffmpeg.")
    return out.with_suffix(".tmp.wav"), out


def finish_output(wav, out):
    if wav != out:
        run([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-i", str(wav), str(out)], check=True)
        wav.unlink()


# ---------- headless ----------

def check_speaker(model, lang, speaker):
    if speaker.lower() not in [n for n, _ in SPEAKERS]:
        die(f"unknown speaker '{speaker}'. Available: {', '.join(n for n, _ in SPEAKERS)}")


def headless(a, hw):
    lang = a.lang.lower()
    model = a.model if a.model != "auto" else recommend(lang, hw)
    if not supports(model, lang):
        die(f"{MODELS[model]['title']} does not support '{lang}'. Try --model auto, or see: python3 tts.py --list-models")
    text = a.text if a.text is not None else pathlib.Path(clean_path(a.text_file)).read_text(encoding="utf-8")
    if not text.strip():
        die("the text is empty")
    ref, voice_dir = "", None
    if a.speaker:
        check_speaker(model, lang, a.speaker)
        label = a.speaker.lower()
    elif a.voice:
        voice_dir = VOICES / voice_name(a.voice)
        if not (voice_dir / "ref.wav").exists():
            die(f"no saved voice called '{a.voice}'. Saved voices: {', '.join(saved_voices()) or 'none yet'}")
        label = voice_name(a.voice)
    else:
        ref = clean_path(a.clone)
        if not os.path.isfile(ref):
            die(f"reference file not found: {ref}")
        label = voice_name(a.save_voice) if a.save_voice else "_last"
        voice_dir = VOICES / label
    out = pathlib.Path(a.out) if a.out else OUT / f"{label.strip('_') or 'voice'}-{datetime.datetime.now():%Y%m%d-%H%M%S}.wav"
    wav, out = final_output(out, model)
    say(f"Model: {MODELS[model]['title']}   Language: {LANG_NAMES.get(lang, lang)}   Device: {describe(hw)}")
    ensure_env(model, hw, a.yes or not sys.stdin.isatty())
    if ref:
        prepare_voice(model, voice_dir, ref, a.clone_text, lang)
    speak(model, voice_dir, text, wav, lang, a.seed, a.device, a.speaker, a.style)
    finish_output(wav, out)
    say(f"\nSaved: {out}")


# ---------- guided mode ----------

def pick_model(lang, hw, speaker_mode):
    model = recommend(lang, hw)
    say(f"\nModel: {MODELS[model]['title']} ({MODELS[model]['note']})")
    return model


def guided(hw):
    say("\n=== tts: guided mode ===")
    say("I'll ask a few questions, set things up and generate your first clip.")
    say("At the end I print the one-line command that does the same without questions.\n")
    say(f"Your computer: {describe(hw)}\n")

    say("1) Which voice?")
    say("   1. A built-in speaker (no recording needed)")
    say("   2. My own voice, or any voice I have a recording of (cloning)")
    speaker_mode = ask("   Number", "1") != "2"

    lang = ask("\n2) Which language will the spoken text be in? (code, e.g. en, de, fr, es)", "en").lower()
    if lang not in QWEN_LANGS:
        say(f"   '{lang}' is not supported here. Supported: {', '.join(QWEN_LANGS)}")
        lang = "en"
    if speaker_mode and lang not in DEFAULT_SPEAKER:
        say(f"   Note: every built-in speaker is native in English, Chinese, Japanese or Korean and speaks {LANG_NAMES.get(lang, lang)} with an accent.")
        say("   For a natural voice in that language, choose cloning with a recording of a native speaker.")
    model = pick_model(lang, hw, speaker_mode)

    speaker, style, voice, ref, label = "", "", "", "", ""
    if speaker_mode:
        say("\n3) Pick a speaker:")
        for i, (n, note) in enumerate(SPEAKERS, 1):
            say(f"   {i}. {n}: {note}")
        d = DEFAULT_SPEAKER.get(lang, "ryan")
        pick = ask("   Number or name", d)
        names = [n for n, _ in SPEAKERS]
        speaker = names[int(pick) - 1] if pick.isdigit() and 1 <= int(pick) <= len(names) else (pick.lower() if pick.lower() in names else d)
        style = ask("   Optional: how should it speak? (e.g. \"slowly and sadly\", \"excited\", or Enter to skip)", "")
        voice_dir = None
    else:
        voices = saved_voices()
        if voices:
            say(f"\n3) Saved voices: {', '.join(voices)}")
            voice = ask("   Type one of these names to reuse it, or press Enter to use a new recording", "")
            voice = voice_name(voice) if voice_name(voice) in voices else ""
        if not voice:
            say("\n3) Your reference recording: 5 to 15 seconds of one person speaking clearly, no music or background noise.")
            say("   Any audio file works (wav, mp3, m4a, ...). Drag the file into this window or type its path.")
            while True:
                ref = clean_path(ask("   File:"))
                if os.path.isfile(ref):
                    break
                say("   I can't find that file, try again.")
            label = voice_name(ask("   Save this voice under a name so you can reuse it? (name, or Enter to skip)", ""))
        voice_dir = VOICES / (voice or label or "_last")

    ensure_env(model, hw, False)

    if ref:
        say("\nPreparing the reference...")
        prepare_voice(model, voice_dir, ref, "", lang)
        txt = voice_dir / "ref.txt"
        if txt.exists():
            heard = txt.read_text(encoding="utf-8").strip()
            say(f"\nI heard this in your recording:\n   \"{heard}\"")
            fix = ask("   Correct? Press Enter, or type the right text (it clones better when the transcript is exact):", "")
            if fix:
                txt.write_text(fix + "\n", encoding="utf-8")

    say("\n4) What should it say? Type a sentence, or give the path of a .txt file.")
    t = ask("   Text:")
    while not t:
        t = ask("   Text:")
    cand = clean_path(t)
    text = pathlib.Path(cand).read_text(encoding="utf-8") if t.lower().endswith(".txt") and os.path.isfile(cand) else t
    base = speaker or voice or label or "voice"
    default_out = rel(OUT / f"{base}-{datetime.datetime.now():%Y%m%d-%H%M%S}.wav")
    out = pathlib.Path(clean_path(ask("5) Where should the audio go?", default_out)))

    argv = [launcher(), "tts.py"]
    if speaker:
        argv += ["--speaker", speaker] + (["--style", style] if style else [])
    else:
        argv += ["--voice", voice or label] if (voice or label) else ["--clone", ref]
    argv += ["--text", text] if len(text) <= 200 else ["--text-file", "my-text.txt"]
    argv += ["--lang", lang, "--model", model, "-o", rel(out)]
    say("\nNext time, this one command does everything above without questions:")
    say(f"\n   {shell_join(argv)}\n")

    wav, final = final_output(out, model)
    speak(model, voice_dir, text, wav, lang, None, "auto", speaker, style)
    finish_output(wav, final)
    say(f"\nSaved: {final}")
    say("If it doesn't sound right: try another --seed" + ("." if speaker else ", a cleaner or slightly longer reference, or the other model."))


# ---------- server ----------

def serve(a, hw):
    """Run server.py (the HTTP voice service the Studio talks to) in the Qwen environment; start it again if it ends."""
    import time
    ensure_env("qwen3-tts-1.7b", hw, a.yes or not sys.stdin.isatty())
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTORCH_ENABLE_MPS_FALLBACK="1", TOKENIZERS_PARALLELISM="false")
    if a.voices_dir:
        env["VOICE_DIR"] = str(pathlib.Path(a.voices_dir).resolve())
    if a.offline:
        env["VOICE_OFFLINE"] = "1"
    if a.whisper:
        env["VOICE_WHISPER"] = a.whisper
    cmd = [str(env_python("qwen")), str(SRC / "server.py"), "--host", a.host, "--port", str(a.port)]
    say(f"Voice service on http://{a.host}:{a.port}  (Ctrl+C stops it)")
    while True:
        try:
            code = run(cmd, env=env).returncode
        except KeyboardInterrupt:
            return
        # exit code 3 = the watchdog ended a stuck job; any other exit is restarted as well, after a short pause
        say(f"Voice service ended (exit {code}), starting it again in 5 s...")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            return


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Text to speech with built-in speakers or a cloned voice. No arguments = guided mode.",
        epilog="Examples:  python3 tts.py --speaker ryan --text \"Hello world\" --lang en\n"
               "          python3 tts.py --clone me.wav --text \"Hello world\" --lang en",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speaker", help="a built-in speaker (see --list-speakers), no recording needed")
    ap.add_argument("--style", default="", help="with --speaker: how to speak, e.g. \"slowly and sadly\" (plain English works best)")
    ap.add_argument("--clone", metavar="FILE", help="clone a voice from this recording: 5-15 s of one speaker (wav, mp3, m4a, ...)")
    ap.add_argument("--voice", help="use a voice saved earlier with --save-voice")
    ap.add_argument("--save-voice", metavar="NAME", help="keep the prepared reference under this name for later (--voice NAME)")
    ap.add_argument("--text", help="what to say")
    ap.add_argument("--text-file", help="read what to say from a text file")
    ap.add_argument("-o", "--out", help="output file (default: out/<voice>-<time>.wav; .mp3 works if ffmpeg is installed)")
    ap.add_argument("--lang", default="auto", help="language of the text, e.g. en, de, fr (default: auto = let the model decide)")
    ap.add_argument("--model", default="auto", choices=["auto"] + list(MODELS), help="default auto = best for your language and computer")
    ap.add_argument("--clone-text", help="with --clone: the exact words spoken in the recording (default: Whisper listens for you)")
    ap.add_argument("--seed", type=int, help="same seed + same text = same result; change it for a different take")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("-y", "--yes", action="store_true", help="set up the Python environment without asking")
    ap.add_argument("--setup", nargs="?", const="auto", metavar="MODEL", help="only install the environment for MODEL, then exit")
    ap.add_argument("--serve", action="store_true", help="run the HTTP voice service (what the Studio talks to) instead of generating once")
    ap.add_argument("--host", default="127.0.0.1", help="with --serve: address to listen on (0.0.0.0 = reachable from other computers)")
    ap.add_argument("--port", type=int, default=8190, help="with --serve: port (default 8190)")
    ap.add_argument("--voices-dir", help="with --serve: folder with the saved voices (default: voices/); use an existing folder to keep your voices")
    ap.add_argument("--whisper", help="with --serve: Whisper model for transcripts (default large-v3-turbo; \"small\" is a much smaller download)")
    ap.add_argument("--offline", action="store_true", help="with --serve: never contact Hugging Face (models must already be downloaded)")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--list-voices", action="store_true")
    ap.add_argument("--list-speakers", action="store_true")
    a = ap.parse_args()

    hw = detect_hardware()
    if a.list_models:
        say(f"Your computer: {describe(hw)}\n")
        for m, info in MODELS.items():
            say(f"{m}\n   {info['title']}: {info['note']}\n   languages: {', '.join(info['langs'])}")
        return
    if a.list_speakers:
        say("\n".join(f"{n:10} {note}" for n, note in SPEAKERS))
        return
    if a.list_voices:
        say("\n".join(saved_voices()) or "No saved voices yet. Create one with: --clone FILE --save-voice NAME")
        return
    if a.serve:
        return serve(a, hw)
    if a.setup:
        m = a.setup if a.setup != "auto" else recommend("auto", hw)
        if m not in MODELS:
            die(f"unknown model '{m}'. Choose from: {', '.join(MODELS)}")
        ensure_env(m, hw, True)
        return
    if a.speaker or a.clone or a.voice or a.text or a.text_file:
        if sum(bool(x) for x in (a.speaker, a.clone, a.voice)) != 1:
            die("give exactly one of --speaker NAME (built-in), --clone FILE (clone a new recording) or --voice NAME (a saved clone)")
        if (a.text is None) == (a.text_file is None):
            die("give exactly one of --text \"...\" or --text-file FILE")
        return headless(a, hw)
    if not sys.stdin.isatty():
        ap.print_help()
        return
    guided(hw)


if __name__ == "__main__":
    main()
