#!/usr/bin/env python3
"""easy-tts: text to speech on your own computer, with voice cloning from a short recording.

  tts                                  first start: a short setup (language, your voice); afterwards it asks for a text and speaks it
  tts "Hello!"                         speak a text with your voice        (also: echo "Hello!" | tts)
  tts --configure                      change the language or the voice
  tts --clone me.wav --text "Hello!"   clone a voice from a recording (add --save-voice NAME to keep it)

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
CONFIG = ROOT / "config.json"   # your language and voice, written by the first-start setup
VERBOSE = False
IS_WIN = os.name == "nt"

# Everything the launcher knows about the models. `langs` are ISO codes. `python` is the interpreter uv fetches for the venv.
QWEN_LANGS = ["en", "de", "fr", "es", "it", "pt", "ru", "zh", "ja", "ko"]
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
    say("easy-tts - text to speech on your own computer, with voice cloning")


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


def quiet_run(cmd):
    """Run a setup command without its chatter (everything with -v); on failure show what it printed."""
    if VERBOSE:
        run(cmd, check=True)
        return
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
    say("Setting up (first time only, a few minutes)...")
    venv = VENVS / backend
    if venv.exists():
        shutil.rmtree(venv)     # a half-finished earlier attempt
    py = str(env_python(backend))
    quiet_run([uv, "venv", "--python", MODELS[model]["python"], str(venv)])
    reqs = ["-r", str(ROOT / "requirements" / f"{backend}.txt")]
    nvidia = hw["kind"] == "cuda"
    if backend == "qwen" and nvidia and IS_WIN:
        # on Windows the default PyPI torch is CPU-only
        quiet_run([uv, "pip", "install", "--python", py, "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu128"])
    quiet_run([uv, "pip", "install", "--python", py] + reqs)
    (venv / ".tts-ready").write_text("ok\n")


def engine(model, args, quiet=True):
    """Run engine.py in the model's virtualenv. Quiet unless -v or a model has to be downloaded (then its progress bars show)."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTORCH_ENABLE_MPS_FALLBACK="1",
               TOKENIZERS_PARALLELISM="false")
    cmd = [str(env_python(MODELS[model]["backend"])), str(SRC / "engine.py")] + args
    if VERBOSE or not quiet:
        r = run(cmd, env=env)
        if r.returncode:
            sys.exit(r.returncode)
        return
    w = Working("Working...")
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


def lang_folder(lang):
    return lang if lang in QWEN_LANGS else "other"


def voice_dirs():
    """All saved voices as {name: folder}: 'de/klara' for voices/de/klara (sorted by language), 'klara' for an older voices/klara."""
    found = {}
    if VOICES.is_dir():
        for d in sorted(VOICES.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            if (d / "ref.wav").exists():
                found[d.name] = d
            else:
                for e in sorted(d.iterdir()):
                    if e.is_dir() and not e.name.startswith("_") and (e / "ref.wav").exists():
                        found[f"{d.name}/{e.name}"] = e
    return found


def saved_voices():
    return list(voice_dirs())


def find_voice(name, lang=""):
    """A saved voice by 'de/klara' or just 'klara' -> (name, folder), or None. If 'klara' exists in several languages the language decides."""
    found = voice_dirs()
    name = "/".join(voice_name(x) for x in name.split("/"))
    if name in found:
        return name, found[name]
    hits = [k for k in found if k.split("/")[-1] == name]
    if len(hits) > 1 and lang:
        hits = [k for k in hits if k.startswith(lang + "/")] or hits
    if len(hits) > 1:
        die(f"'{name}' exists in several languages: {', '.join(hits)}. Say which one, for example --voice {hits[0]}")
    return (hits[0], found[hits[0]]) if hits else None


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


def load_config():
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_config(cfg):
    CONFIG.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def hf_cache():
    return pathlib.Path(os.environ.get("HF_HUB_CACHE") or (pathlib.Path(os.environ.get("HF_HOME") or "~/.cache/huggingface").expanduser() / "hub"))


def cached(repo, key_file):
    """Is this Hugging Face model already on disk? (decides whether a download progress bar is worth showing)"""
    snaps = hf_cache() / ("models--" + repo.replace("/", "--")) / "snapshots"
    return snaps.is_dir() and any((d / key_file).exists() for d in snaps.iterdir())


class Working:
    """A one-line 'working...' note that disappears when done (only on a terminal)."""

    def __init__(self, text):
        self.on = sys.stdout.isatty() and not VERBOSE
        if self.on:
            sys.stdout.write(text + "\r")
            sys.stdout.flush()
        self.n = len(text)

    def done(self):
        if self.on:
            sys.stdout.write(" " * self.n + "\r")
            sys.stdout.flush()


# ---------- the actual work ----------

def prepare_voice(model, voice_dir, ref, ref_text, lang):
    args = ["prepare", "--ref", ref, "--voice-dir", str(voice_dir), "--lang", lang]
    if ref_text:
        args += ["--ref-text", ref_text]
    quiet = cached("Systran/faster-whisper-small", "model.bin") or bool(ref_text)
    if not quiet:
        say("Downloading the speech recognition model (about 0.5 GB, first time only)...")
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
        quiet = cached(MODELS[model]["repo"], "model.safetensors") and (fast or cached("Systran/faster-whisper-small", "model.bin"))
        if not quiet:
            say("Downloading the voice model (about 4 GB, first time only)...")
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


def make_clip(a, hw, text, lang, voice="", clone=""):
    """Speak `text` with a saved voice or a recording, write the file, print where it is."""
    model = a.model if a.model != "auto" else recommend(lang, hw)
    if not supports(model, lang):
        die(f"'{lang}' is not supported. Supported languages: {', '.join(QWEN_LANGS)}")
    if not text.strip():
        die("the text is empty")
    ref = ""
    if voice:
        hit = find_voice(voice, lang)
        if not hit:
            die(f"no saved voice called '{voice}'. Saved voices: {', '.join(saved_voices()) or 'none yet'}")
        label, voice_dir = hit[0].split("/")[-1], hit[1]
    else:
        ref = clean_path(clone)
        if not os.path.isfile(ref):
            die(f"recording not found: {ref}")
        label = voice_name(a.save_voice) if a.save_voice else "_last"
        voice_dir = VOICES / lang_folder(lang) / label if a.save_voice else VOICES / label
    out = pathlib.Path(a.out) if a.out else OUT / f"{label.strip('_') or 'voice'}-{datetime.datetime.now():%Y%m%d-%H%M%S}.wav"
    wav, out = final_output(out, model)
    ensure_env(model, hw, True)
    if ref:
        prepare_voice(model, voice_dir, ref, a.clone_text, lang)
    speak(model, voice_dir, text, wav, lang, a.seed, a.device, a.fast, a.speed)
    finish_output(wav, out)
    say(f"Saved: {rel(out)}")


# ---------- first-start setup ----------

def configure(hw, first):
    """Ask for the language and the voice, prepare the voice, remember both in config.json."""
    cfg = load_config()
    say("\nInitial configuration" if first else "\nConfiguration")
    lang = ask(f"Language of your texts ({', '.join(QWEN_LANGS)})", cfg.get("lang", "en")).lower()
    while lang not in QWEN_LANGS:
        lang = ask("Not supported. Please pick one of the list", "en").lower()
    voice, ref = "", ""
    voices = saved_voices()
    if voices:
        hit = find_voice(ask(f"Use a saved voice? ({', '.join(voices)}) Name, or Enter for a new recording", "") or "-", lang)
        voice = hit[0] if hit else ""
    if not voice:
        say("Recording of the voice: 5 to 15 seconds of one person speaking clearly. Drag the file here.")
        while True:
            ref = clean_path(ask("File:"))
            if os.path.isfile(ref):
                break
            say("File not found, try again.")
        default = voice_name(pathlib.Path(ref).stem) or "voice"
        voice = f"{lang_folder(lang)}/{voice_name(ask('Name for this voice', default)) or default}"
    model = recommend(lang, hw)
    ensure_env(model, hw, True)
    if ref:
        prepare_voice(model, VOICES / voice, ref, "", lang)
        txt = VOICES / voice / "ref.txt"
        if txt.exists():
            say(f"I heard: \"{txt.read_text(encoding='utf-8').strip()}\"")
            fix = ask("Correct? Press Enter, or type the right text (an exact transcript clones better):", "")
            if fix:
                txt.write_text(fix + "\n", encoding="utf-8")
    cfg.update(lang=lang, voice=voice)
    save_config(cfg)
    return cfg


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
    global VERBOSE
    ap = argparse.ArgumentParser(
        description="easy-tts: text to speech on your own computer, with voice cloning from a short recording. "
                    "First start: a short setup. Afterwards: tts \"your text\".",
        epilog="Examples:  tts \"Hello world\"                      speak with your configured voice\n"
               "          echo \"Hello world\" | tts                text from stdin\n"
               "          tts --clone me.wav --save-voice me --text \"Hello world\" --lang en",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="*", metavar="TEXT", help="what to say (or use --text, --text-file, or pipe it in)")
    ap.add_argument("--configure", action="store_true", help="change the language or the voice you use")
    ap.add_argument("--clone", metavar="FILE", help="clone a voice from this recording: 5-15 s of one speaker (wav, mp3, m4a, ...)")
    ap.add_argument("--voice", help="use a voice saved earlier with --save-voice")
    ap.add_argument("--save-voice", metavar="NAME", help="keep the prepared recording under this name for later (--voice NAME)")
    ap.add_argument("--text", help="what to say")
    ap.add_argument("--text-file", help="read what to say from a text file")
    ap.add_argument("-o", "--out", help="output file (default: out/<voice>-<time>.wav; .mp3 works if ffmpeg is installed)")
    ap.add_argument("--lang", help="language of the text, e.g. en, de, fr (default: the one you configured)")
    ap.add_argument("--model", default="auto", choices=["auto"] + list(MODELS), help=argparse.SUPPRESS)
    ap.add_argument("--clone-text", help="with --clone: the exact words spoken in the recording (default: Whisper listens for you)")
    ap.add_argument("--seed", type=int, help="same seed + same text = same result; change it for a different take")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--speed", type=float, default=1.0, metavar="X", help="speaking speed: 1.0 = as the voice speaks, 1.2 = faster, 0.8 = slower (0.5 to 2)")
    ap.add_argument("--fast", action="store_true", help="skip the check that listens to the result and redoes pieces that do not match the text")
    ap.add_argument("-v", "--verbose", action="store_true", help="show everything the setup and the voice engine print")
    ap.add_argument("-y", "--yes", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--setup", nargs="?", const="auto", metavar="MODEL", help="only install the environment, then exit")
    ap.add_argument("--serve", action="store_true", help="run the HTTP voice service instead of generating once")
    ap.add_argument("--host", default="127.0.0.1", help="with --serve: address to listen on (0.0.0.0 = reachable from other computers)")
    ap.add_argument("--port", type=int, default=8190, help="with --serve: port (default 8190)")
    ap.add_argument("--voices-dir", help="with --serve: folder with the saved voices (default: voices/); use an existing folder to keep your voices")
    ap.add_argument("--whisper", help="with --serve: Whisper model for transcripts (default large-v3-turbo; \"small\" is a much smaller download)")
    ap.add_argument("--offline", action="store_true", help="with --serve: never contact Hugging Face (models must already be downloaded)")
    ap.add_argument("--list-models", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--list-voices", action="store_true", help="show the voices you saved")
    # Forgive a single dash on a long option: "-voice klara" means "--voice klara" (otherwise argparse reads it as -v -o ice ...)
    longs = {o for o in ap._option_string_actions if o.startswith("--")}

    def fix(arg):
        name, eq, rest = arg.partition("=")
        return "-" + name + eq + rest if len(name) > 2 and not name.startswith("--") and "-" + name in longs else arg

    a = ap.parse_args([fix(x) for x in sys.argv[1:]])
    VERBOSE = a.verbose

    hw = detect_hardware()
    if a.list_models:
        say(f"Your computer: {describe(hw)}\n")
        for m, info in MODELS.items():
            say(f"{m}\n   {info['title']}: {info['note']}\n   languages: {', '.join(info['langs'])}")
        return
    if a.list_voices:
        say("\n".join(saved_voices()) or "No saved voices yet. Run tts once to set one up.")
        return
    banner()
    if a.serve:
        return serve(a, hw)
    if a.setup:
        m = a.setup if a.setup != "auto" else recommend("auto", hw)
        if m not in MODELS:
            die(f"unknown model '{m}'. Choose from: {', '.join(MODELS)}")
        ensure_env(m, hw, True)
        return

    cfg = load_config()
    if a.clone and a.voice:
        die("give either --clone FILE (a new recording) or --voice NAME (a voice saved earlier), not both")
    if not 0.5 <= a.speed <= 2.0:
        die("--speed must be between 0.5 and 2 (1.0 = unchanged)")
    if sum(x is not None and x != [] for x in (a.text, a.words or None, a.text_file)) > 1:
        die("give the text once: as an argument, with --text, or with --text-file")
    if a.configure:
        if not sys.stdin.isatty():
            die("--configure needs a terminal")
        configure(hw, first=not cfg.get("voice"))
        say("Saved. Now: tts \"your text\"")
        return

    voice, clone = a.voice or "", a.clone or ""
    lang = (a.lang or cfg.get("lang") or "auto").lower()
    if not voice and not clone:
        voice = cfg.get("voice", "")
        if not voice:                       # first start
            if not sys.stdin.isatty():
                die("not set up yet: run tts once in a terminal (or use --clone FILE)")
            cfg = configure(hw, first=True)
            voice, lang = cfg["voice"], (a.lang or cfg["lang"]).lower()
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
    make_clip(a, hw, text, lang, voice, clone)


if __name__ == "__main__":
    main()
