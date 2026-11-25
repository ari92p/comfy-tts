#!/usr/bin/env python3
"""easy-tts: text to speech on your own computer, with voice cloning from a short recording.

  easy-tts                                    guided: asks for the voice, the language (new voices) and the text
  easy-tts --voice me "Hello!"                one command with a saved voice (it remembers its language; also: echo "Hello!" | easy-tts --voice me)
  easy-tts --clone me.wav --save me --lang en "Hello!"    clone a voice from a recording and keep it (with its language)

Run with --help for all options. This file only needs a plain Python 3.9+; the models run in private virtualenvs
(.venvs/) that are created for you on first use, see README.md.
"""
import argparse
import datetime
import json
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
WHISPER_REPO = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"   # the default Whisper model (engine.py and server.py use the same)
MODELS = {
    "qwen3-tts-1.7b": {
        "backend": "qwen", "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "langs": QWEN_LANGS, "python": "3.12",
        "title": "Qwen3-TTS 1.7B",
        "note": "the largest Qwen3-TTS model, clones a voice, 10 languages; needs a GPU (NVIDIA 8+ GB or an Apple Silicon Mac)",
    },
}
LANG_NAMES = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
              "ru": "Russian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean"}


def banner():
    bold = sys.stdout.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"
    say("\033[1m# easy-tts\033[0m" if bold else "# easy-tts")


def say(*a):
    print(*a, flush=True)


def status(msg):
    say("#easy-tts: " + msg)


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


def quiet_run(cmd):
    """Run a setup command without its chatter; on failure show what it printed."""
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        die("setup failed:\n" + "\n".join(r.stdout.strip().splitlines()[-25:]))


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
    status("Setting up (first time only, a few minutes)...")
    venv = VENVS / backend
    if venv.exists():
        shutil.rmtree(venv)     # a half-finished earlier attempt
    py = str(env_python(backend))
    quiet_run([uv, "venv", "--python", MODELS[model]["python"], str(venv)])
    reqs = ["-r", str(SRC / "requirements.txt")]
    nvidia = hw["kind"] == "cuda"
    if backend == "qwen" and nvidia and IS_WIN:
        # on Windows the default PyPI torch is CPU-only
        quiet_run([uv, "pip", "install", "--python", py, "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu128"])
    quiet_run([uv, "pip", "install", "--python", py] + reqs)
    (venv / ".tts-ready").write_text("ok\n")


def engine(model, args, quiet=True):
    """Run engine.py in the model's virtualenv. Quiet unless a model has to be downloaded (then its progress bars show)."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTORCH_ENABLE_MPS_FALLBACK="1",
               TOKENIZERS_PARALLELISM="false")
    cmd = [str(env_python(MODELS[model]["backend"])), str(SRC / "engine.py")] + args
    if not quiet:
        r = run(cmd, env=env)
        if r.returncode:
            sys.exit(r.returncode)
        return
    w = Working("#easy-tts: Working...")
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    w.done()
    if r.returncode:
        die("the voice engine failed:\n" + "\n".join(r.stdout.strip().splitlines()[-25:]))


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


def voice_dirs():
    """All saved voices as {name: folder} (voices/<name>/ with ref.wav and ref.txt)."""
    if not VOICES.is_dir():
        return {}
    return {d.name: d for d in sorted(VOICES.iterdir()) if d.is_dir() and not d.name.startswith("_") and (d / "ref.wav").exists()}


def saved_voices():
    return list(voice_dirs())


def voice_lang(d):
    """The language saved with a voice (voices/<name>/meta.json), or ''."""
    try:
        return json.loads((d / "meta.json").read_text(encoding="utf-8")).get("lang", "")
    except (OSError, ValueError):
        return ""


def set_voice_lang(d, lang):
    """Keep the language with the voice; other entries of meta.json (written by the voice service) stay."""
    if lang and lang != "auto":
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        meta["lang"] = lang
        (d / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")


def find_voice(name):
    """A saved voice by name -> (name, folder), or None."""
    name = voice_name(name)
    d = voice_dirs().get(name)
    return (name, d) if d else None


def shell_join(argv):
    return subprocess.list2cmdline(argv) if IS_WIN else " ".join(shlex.quote(a) for a in argv)


def launcher():
    return "python" if IS_WIN else "python3"


def rel(p):
    """A path relative to the current folder when it lies inside it, otherwise the full path."""
    p = pathlib.Path(p).resolve()
    try:
        return str(p.relative_to(pathlib.Path.cwd()))
    except ValueError:
        return str(p)


def hf_cache():
    return pathlib.Path(os.environ.get("HF_HUB_CACHE") or (pathlib.Path(os.environ.get("HF_HOME") or "~/.cache/huggingface").expanduser() / "hub"))


def cached(repo, key_file):
    """Is this Hugging Face model already on disk? (decides whether a download progress bar is worth showing)"""
    snaps = hf_cache() / ("models--" + repo.replace("/", "--")) / "snapshots"
    return snaps.is_dir() and any((d / key_file).exists() for d in snaps.iterdir())


class Working:
    """A one-line 'working...' note that disappears when done (only on a terminal)."""

    def __init__(self, text):
        self.on = sys.stdout.isatty()
        if self.on:
            sys.stdout.write(text + "\r")
            sys.stdout.flush()
        self.n = len(text)

    def done(self):
        if self.on:
            sys.stdout.write(" " * self.n + "\r")
            sys.stdout.flush()


# ---------- the actual work ----------

def prepare_voice(model, voice_dir, ref, lang):
    args = ["prepare", "--ref", ref, "--voice-dir", str(voice_dir), "--lang", lang]
    quiet = cached(WHISPER_REPO, "model.bin")
    if not quiet:
        status("Downloading the speech recognition model (about 1.5 GB, first time only)...")
    engine(model, args, quiet)


def speak(model, voice_dir, text, out, lang, seed, device, fast=False, speed=1.0):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        tmp = f.name
    try:
        args = ["speak", "--text-file", tmp, "--out", str(out), "--lang", lang, "--device", device]
        args += ["--model", MODELS[model]["repo"], "--voice-dir", str(voice_dir)]
        if seed is not None:
            args += ["--seed", str(seed)]
        if fast:
            args += ["--no-check"]
        if speed != 1.0:
            args += ["--speed", str(speed)]
        quiet = cached(MODELS[model]["repo"], "model.safetensors") and (fast or cached(WHISPER_REPO, "model.bin"))
        if not quiet:
            status("Downloading the voice model (about 4 GB, first time only)...")
        engine(model, args, quiet)
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


# ---------- one clip ----------

def read_text_file(path):
    return pathlib.Path(clean_path(path)).read_text(encoding="utf-8")


def transcribe_audio(a, hw):
    """Voice to text: write what is said in an audio file as a text file (default: out/<name>-<time>.txt)."""
    src = clean_path(a.transcribe)
    if not os.path.isfile(src):
        die(f"audio file not found: {src}")
    model = recommend("auto", hw)
    ensure_env(model, hw, True)
    out = pathlib.Path(a.out) if a.out else OUT / f"{pathlib.Path(src).stem}-{datetime.datetime.now():%Y%m%d-%H%M%S}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    quiet = cached(WHISPER_REPO, "model.bin")
    if not quiet:
        status("Downloading the speech recognition model (about 1.5 GB, first time only)...")
    engine(model, ["transcribe", "--file", src, "--out-file", str(out), "--lang", (a.lang or "auto").lower()], quiet)
    status(f"Saved: {rel(out)}")


def save_voice_only(a, hw, lang):
    """--clone FILE --save NAME without a text: prepare and keep the voice, make no audio."""
    ref = clean_path(a.clone)
    if not os.path.isfile(ref):
        die(f"recording not found: {ref}")
    name = voice_name(a.save)
    if not name:
        die("--save needs a name")
    model = recommend(lang, hw)
    ensure_env(model, hw, True)
    prepare_voice(model, VOICES / name, ref, lang)
    set_voice_lang(VOICES / name, lang)
    status(f"Voice saved: {name} ({lang}). Use it: easy-tts --voice {name} \"your text\"")


def make_clip(a, hw, text, lang, voice="", clone=""):
    """Speak `text` with a saved voice or a recording, write the file, print where it is."""
    model = a.model if a.model != "auto" else recommend(lang, hw)
    if not supports(model, lang):
        die(f"'{lang}' is not supported. Supported languages: {', '.join(QWEN_LANGS)}")
    if not text.strip():
        die("the text is empty")
    ref = ""
    if voice:
        hit = find_voice(voice)
        if not hit:
            die(f"no saved voice called '{voice}'. Saved voices: {', '.join(saved_voices()) or 'none yet'}")
        label, voice_dir = hit
    else:
        ref = clean_path(clone)
        if not os.path.isfile(ref):
            die(f"recording not found: {ref}")
        label = voice_name(a.save) if a.save else "_last"
        voice_dir = VOICES / label
    out = pathlib.Path(a.out) if a.out else OUT / f"{label.strip('_') or 'voice'}-{datetime.datetime.now():%Y%m%d-%H%M%S}.wav"
    wav, out = final_output(out, model)
    ensure_env(model, hw, True)
    if ref:
        prepare_voice(model, voice_dir, ref, lang)
        if a.save:
            set_voice_lang(voice_dir, lang)
    speak(model, voice_dir, text, wav, lang, a.seed, a.device, a.fast, a.speed)
    finish_output(wav, out)
    status(f"Saved: {rel(out)}")


# ---------- first-start setup ----------

def new_voice(model, lang):
    """A new voice from a recording (Whisper writes down what is said into ref.txt without asking). -> name"""
    say("\nRecording: 5 to 15 seconds of one person speaking clearly (drag the file here)")
    while True:
        ref = clean_path(ask("File:"))
        if os.path.isfile(ref):
            break
        say("File not found, try again.")
    default = voice_name(pathlib.Path(ref).stem) or "voice"
    voice = voice_name(ask("Name", default)) or default
    voice_dir = VOICES / voice
    prepare_voice(model, voice_dir, ref, lang)
    set_voice_lang(voice_dir, lang)
    return voice


def ask_lang(default="en"):
    lang = ask(f"Language ({', '.join(QWEN_LANGS)})", default).lower()
    while lang not in QWEN_LANGS:
        lang = ask("Please pick one of the list", default).lower()
    return lang


def dialog(hw):
    """The guided run: voice (saved or a new recording), language, text. -> (lang, voice, text)"""
    banner()
    voice, lang = "", ""
    dirs = voice_dirs()
    voices = list(dirs)
    if voices:
        say("\nVoices")
        for i, v in enumerate(voices, 1):
            say(f"  {i:>2}  {v:<22}{voice_lang(dirs[v])}")
        say("   n  new recording")
        pick = ask("Choose", "n").strip().lower()
        hit = (voices[int(pick) - 1], None) if pick.isdigit() and 1 <= int(pick) <= len(voices) else find_voice(pick) if pick != "n" else None
        voice = hit[0] if hit else ""
        lang = voice_lang(dirs[voice]) if voice else ""
    if not lang:                       # a new recording, or an older voice without a language
        say("")
        lang = ask_lang()
    model = recommend(lang, hw)
    ensure_env(model, hw, True)
    new = not voice
    if new:
        voice = new_voice(model, lang)
    elif not voice_lang(dirs[voice]):
        set_voice_lang(dirs[voice], lang)
    say("")
    t = ask("Text (Enter = only save the voice):" if new else "Text:")
    while not t and not new:
        t = ask("Text:")
    if not t:
        return lang, voice, ""
    return lang, voice, read_text_file(t) if t.lower().endswith(".txt") and os.path.isfile(clean_path(t)) else t


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
    cmd = [str(env_python("qwen")), str(SRC / "server.py"), "--host", a.host, "--port", str(a.port)]
    say(f"easy-tts service on http://{a.host}:{a.port}  (Ctrl+C stops it)")
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
    ap = argparse.ArgumentParser(prog="easy-tts",
        description="easy-tts: text to speech on your own computer, with voice cloning from a short recording. "
                    "Run just `easy-tts` to be guided; with arguments give the voice and the language.",
        epilog="Examples:  easy-tts                                     guided\n"
               "          easy-tts --voice me \"Hello world\"               one command\n"
               "          echo \"Hello\" | easy-tts --voice me             text from stdin\n"
               "          easy-tts --clone me.wav --save me --lang en \"Hello\"",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="*", metavar="TEXT", help="what to say (or use --text, --text-file, or pipe it in)")
    ap.add_argument("--clone", metavar="FILE", help="clone a voice from this recording: 5-15 s of one speaker (wav, mp3, m4a, ...)")
    ap.add_argument("--voice", help="use a voice saved earlier (see --save)")
    ap.add_argument("--save", metavar="NAME", help="with --clone: keep the voice under this name (--voice NAME). Without a text, it only saves the voice")
    ap.add_argument("--text", help="what to say")
    ap.add_argument("--text-file", help="read what to say from a text file")
    ap.add_argument("-o", "--out", help="output file (default: out/<voice>-<time>.wav; .mp3 works if ffmpeg is installed)")
    ap.add_argument("--lang", help="language of the text: en de fr es it pt ru zh ja ko (required with --clone; a saved voice remembers it)")
    ap.add_argument("--model", default="auto", choices=["auto"] + list(MODELS), help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, help="same seed + same text = same result; change it for a different take")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--speed", type=float, default=1.0, metavar="X", help="speaking speed: 1.0 = as the voice speaks, 1.2 = faster, 0.8 = slower (0.5 to 2)")
    ap.add_argument("--fast", action="store_true", help="skip the check that listens to the result and redoes pieces that do not match the text")
    ap.add_argument("-y", "--yes", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--setup", nargs="?", const="auto", metavar="MODEL", help="only install the environment, then exit")
    ap.add_argument("--serve", action="store_true", help="run the HTTP voice service instead of generating once")
    ap.add_argument("--host", default="127.0.0.1", help="with --serve: address to listen on (0.0.0.0 = reachable from other computers)")
    ap.add_argument("--port", type=int, default=8190, help="with --serve: port (default 8190)")
    ap.add_argument("--voices-dir", help="with --serve: folder with the saved voices (default: voices/); use an existing folder to keep your voices")
    ap.add_argument("--offline", action="store_true", help="with --serve: never contact Hugging Face (models must already be downloaded)")
    ap.add_argument("--list-models", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--transcribe", metavar="AUDIO", help="voice to text: write what is said in an audio file to out/<name>-<time>.txt (add --lang if you know it, -o to choose the file)")
    ap.add_argument("--list-voices", action="store_true", help="show the voices you saved")
    # Forgive a single dash on a long option: "-voice klara" means "--voice klara" (otherwise argparse reads it as -v -o ice ...)
    longs = {o for o in ap._option_string_actions if o.startswith("--")}

    def fix(arg):
        name, eq, rest = arg.partition("=")
        return "-" + name + eq + rest if len(name) > 2 and not name.startswith("--") and "-" + name in longs else arg

    a = ap.parse_args([fix(x) for x in sys.argv[1:]])

    hw = detect_hardware()
    if a.list_models:
        say(f"Your computer: {describe(hw)}\n")
        for m, info in MODELS.items():
            say(f"{m}\n   {info['title']}: {info['note']}\n   languages: {', '.join(info['langs'])}")
        return
    if a.list_voices:
        say("\n".join(f"{v:<22}{voice_lang(d)}" for v, d in voice_dirs().items()) or "No saved voices yet. Run easy-tts to record one.")
        return
    if a.transcribe:
        return transcribe_audio(a, hw)
    if a.serve:
        return serve(a, hw)
    if a.setup:
        m = a.setup if a.setup != "auto" else recommend("auto", hw)
        if m not in MODELS:
            die(f"unknown model '{m}'. Choose from: {', '.join(MODELS)}")
        ensure_env(m, hw, True)
        return

    if a.clone and a.voice:
        die("give either --clone FILE (a new recording) or --voice NAME (a voice saved earlier), not both")
    if not 0.5 <= a.speed <= 2.0:
        die("--speed must be between 0.5 and 2 (1.0 = unchanged)")
    if sum(x is not None and x != [] for x in (a.text, a.words or None, a.text_file)) > 1:
        die("give the text once: as an argument, with --text, or with --text-file")

    if not (a.words or a.text is not None or a.text_file or a.voice or a.clone or a.lang):   # plain `easy-tts`: be guided
        if not sys.stdin.isatty():
            ap.print_help()
            return
        lang, voice, text = dialog(hw)
        if not text:
            return status(f"Voice saved: {voice} ({lang}). Use it: easy-tts --voice {voice} \"your text\"")
        return make_clip(a, hw, text, lang, voice, "")

    missing = []
    lang = (a.lang or "").lower()
    if not (a.voice or a.clone):
        missing.append("--voice NAME (saved: " + (", ".join(saved_voices()) or "none yet") + ") or --clone FILE")
    elif a.voice and not lang:
        hit = find_voice(a.voice)
        lang = voice_lang(hit[1]) if hit else ""     # a saved voice brings its language
        if hit and not lang:
            missing.append(f"--lang ({', '.join(QWEN_LANGS)}); the voice '{hit[0]}' has none saved, it is kept with the voice once you give it")
    if (a.clone or not (a.voice or a.clone)) and not lang:
        missing.append(f"--lang ({', '.join(QWEN_LANGS)})")
    if missing:
        die("missing " + " and ".join(missing) + ".\nOr just run `easy-tts` and answer the questions.")
    if a.clone and a.save and a.text is None and not a.words and not a.text_file:
        return save_voice_only(a, hw, lang)                # no text: only keep the voice
    if a.text is not None:
        text = a.text
    elif a.words:
        text = " ".join(a.words)
    elif a.text_file:
        text = read_text_file(a.text_file)
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        t = ""
        while not t:
            t = ask("Text:")
        text = read_text_file(t) if t.lower().endswith(".txt") and os.path.isfile(clean_path(t)) else t
    if a.voice and a.lang:
        hit = find_voice(a.voice)
        if hit and not voice_lang(hit[1]):
            set_voice_lang(hit[1], lang)                 # first time a language is given: keep it with the voice
    make_clip(a, hw, text, lang, a.voice or "", a.clone or "")


if __name__ == "__main__":
    main()
