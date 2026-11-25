# easy-tts-voice-clone

Text to speech on your own computer: pick a built-in speaker, or clone a voice from a 10-second recording.
No account, no API key, everything runs locally. It can also run as a small HTTP voice service for other programs.

```bash
git clone https://github.com/ari92p/easy-tts-voice-clone && cd easy-tts-voice-clone
python3 tts.py
```

The guided mode asks a few questions, helps you pick a model that fits your computer, sets everything up and
generates your first clip. At the end it prints the one-line command that repeats it without questions.

## One command, no questions

Built-in speaker:

```bash
python3 tts.py --speaker ryan --text "Hello, this is a built-in voice." --lang en
python3 tts.py --speaker aiden --style "slowly and sadly" --text "I miss the summer." --lang en
```

Your own voice (clone it from a recording):

```bash
python3 tts.py --ref me.wav --text "Hello, this is my cloned voice." --lang en
python3 tts.py --ref me.mp3 --text-file chapter1.txt --lang de -o chapter1.wav
```

Keep a cloned voice so you don't prepare the recording every time:

```bash
python3 tts.py --ref me.wav --save-voice me --text "First line"     # saved as voices/me
python3 tts.py --voice me --text "Any later line" -o later.wav
```

Results land in `out/`. Long texts are spoken sentence by sentence and joined. On Windows use `python` instead of `python3`.

| Option | What it does |
| --- | --- |
| `--speaker NAME` | built-in speaker, see `--list-speakers`; `--style "..."` says how to speak |
| `--ref FILE` | clone from this recording (wav, mp3, m4a, ...) |
| `--voice NAME` / `--save-voice NAME` | use / keep a prepared cloned voice |
| `--text "..."` / `--text-file FILE` | what to say |
| `--lang de` | language of the text (`auto` lets Qwen decide; Chatterbox needs it, default English) |
| `--model` | `auto` (default), `qwen3-tts-1.7b`, `qwen3-tts-0.6b`, `chatterbox` (cloning only) |
| `--seed N` | same seed and text give the same result; change it for another take |
| `-o FILE` | output (`.wav`; `.mp3` and others work if ffmpeg is installed) |
| `--ref-text "..."` | exact words of your recording (Qwen); otherwise Whisper listens for you |
| `--exaggeration`, `--cfg` | Chatterbox only: emotion and pacing |
| `-y` | set up the Python environment without asking |
| `--list-models`, `--list-speakers`, `--list-voices`, `--setup [MODEL]` | look around, or only install |

## Which model, which speaker?

| Model | Languages | Does | Notes |
| --- | --- | --- | --- |
| `qwen3-tts-1.7b` | 10: en de fr es it pt ru zh ja ko | built-in speakers **and** cloning | the best quality; an NVIDIA GPU with 8+ GB is the tested setup |
| `qwen3-tts-0.6b` | same 10 | speakers and cloning | smaller and faster, a bit less similar |
| `chatterbox` | 23 (adds Arabic, Dutch, Polish, Turkish, Hindi, ...) | cloning only | no transcript needed; emotion control; adds an inaudible watermark to every file |

Built-in speakers: `aiden`, `ryan` (English), `vivian`, `serena`, `uncle_fu`, `dylan`, `eric` (Chinese), `ono_anna` (Japanese), `sohee` (Korean).
**Each speaker is native in one language and speaks the others with an accent.** For English, Chinese, Japanese and Korean the
speakers are the easy choice. For German, French and the rest, a clone of a recording by a native speaker sounds far more natural.

Licenses: Qwen3-TTS is Apache-2.0, Chatterbox is MIT. This repository is MIT.

## What you need

- Python 3.9 or newer to run `tts.py` (it uses nothing but the standard library).
- [`uv`](https://docs.astral.sh/uv/) once, to build the model environments. It also fetches the right Python version itself.
  `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`, on Windows `irm https://astral.sh/uv/install.ps1 | iex`.
- A few GB of disk space and a few minutes on first use. Models and libraries are downloaded once.
- Optional: `ffmpeg` (reading unusual audio formats, writing mp3).

Each model family gets its own environment in `.venvs/` because their libraries pin conflicting versions.
Nothing is installed globally. Delete `.venvs/` to reset.

## A good reference recording (for cloning)

The recording matters more than any setting.

- 5 to 15 seconds of **one** person, spoken normally. Longer files are cut for you at a sentence end.
- No music, no echo, no second voice. A phone recording in a quiet room is fine.
- Speak the language you want to generate, ideally in the tone you want back.
- If the voice drifts or sounds off, try another recording before blaming the model. Then try `--seed`.

## Run it as a voice service

```bash
python3 tts.py --serve                       # http://127.0.0.1:8190
python3 tts.py --serve --host 0.0.0.0        # reachable from other computers (there is no login, use a firewall)
```

This is the HTTP service the Studio uses for its Voice page: speakers, saved voices with several samples, Whisper transcripts,
`POST /generate`. It keeps one model loaded at a time, frees the GPU after 5 minutes without requests, and restarts itself
if a job hangs. The endpoints are listed at the top of [server.py](server.py). It uses Qwen3-TTS 1.7B (Base for cloning, CustomVoice for speakers).

Options: `--port`, `--voices-dir FOLDER` (where voices are kept, default `voices/`), `--offline` (never contact Hugging Face).
Voices cloned on the command line and voices saved through the service share the same folder format, so both see each other's voices.

### Replacing an existing voice service on the PC

Moving from the older `voice_server.py` setup (Windows, `C:\AI\tts`):

1. Test first, next to the running service: `python tts.py --serve --port 8191 --voices-dir C:\AI\tts\voices --offline`, then point a client at port 8191.
2. Keep your voices: `--voices-dir C:\AI\tts\voices` uses the existing folder as it is. Models already cached in `C:\AI\tts\hf` are used if `HF_HOME` points there.
3. Switch over: stop the old scheduled task, run `start_server.cmd` (adjust the `--voices-dir` line in it), and change the task to start `start_server.cmd` at logon.

## What has been tested

Tested on Windows with an NVIDIA RTX 5080 (16 GB), each time generating audio and transcribing it again with Whisper:

- cloning with `qwen3-tts-1.7b`: reference preparation with automatic transcript, a long text in several pieces, the text comes back word for word;
- built-in speakers with `--style` on the command line (`aiden`);
- `--serve`, set up from scratch, on a second port next to the old service and with a copy of its 16 voices: `/health`, `/speakers`, `/voices`,
  `/generate` with a speaker (`ryan`, with `instruct`) and with an existing cloned voice, `/transcribe`;
- the command line using a voice that was saved by the service (same folder format);
- cloning with `chatterbox`, including setting up its environment from scratch.

Tested on a Mac (Apple M5, 24 GB, Apple GPU): cloning with `qwen3-tts-0.6b`, set up from scratch with the system Python 3.9, automatic transcript,
German text. 7 seconds of audio took 11 seconds, and the text came back word for word when transcribed again.

**Not tested yet:** built-in speakers and `--serve` on the Mac, `qwen3-tts-1.7b` and `chatterbox` on the Mac, the other service endpoints
(uploading voices, samples, renaming, deleting), switching the real service over (the old one was left untouched), and Linux or CPU only.
The size hints in the guided mode are rules of thumb, not measurements. If something fails, please open an issue with the output.

In a German round-trip test (generate, then transcribe again) Qwen reproduced the text word for word, while Chatterbox once said "Kaffee-Karte" for "Grafikkarte".

## Please use it responsibly

Only clone voices you have the right to clone: your own, or someone who clearly agreed. Don't use cloned voices to deceive,
impersonate or harass anyone, and check the laws and platform rules where you publish. Chatterbox marks its output with a
watermark; Qwen does not, so say when audio is synthetic.

## Troubleshooting

- **A `SoX could not be found` warning appears:** harmless, nothing here needs it.
- **`error: ... ran away` / a piece fails 3 times:** the model sometimes babbles on a sentence. Reword that sentence, shorten it, or change `--seed`.
- **CUDA out of memory:** use `qwen3-tts-0.6b`, or close other programs using the GPU.
- **Windows with an NVIDIA card but it runs on the CPU:** delete `.venvs/` and run again; the setup installs the CUDA build of PyTorch.
- **Mac or CPU is slow:** the 0.6B model ran faster than real time on an Apple M5; CPU only is much slower. Use the 0.6B model there.
- **The first Mac run takes minutes before it speaks:** that is the model download (about 2.5 GB), later runs start in seconds.
- **The service answers "busy" (503):** it runs one job at a time; the client should wait and try again.

## How it works

`tts.py` detects your hardware, builds the right environment, and calls `engine.py` inside it.
For cloning, `engine.py` turns your recording into a clean 24 kHz reference (cut to at most 15 s, transcribed with Whisper for Qwen).
It then speaks the text in sentence-sized pieces and re-rolls a piece with another seed if it runs away or comes out silent.
Settings come from a lot of trial and error with Qwen3-TTS (repetition penalty 1.05, a hard cap on generated tokens, no post-processing).
`server.py` is the long-running version of the same idea behind an HTTP API.
