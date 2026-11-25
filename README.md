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
| `--lang de` | language of the text (default `auto`: the model decides) |
| `--seed N` | same seed and text give the same result; change it for another take |
| `-o FILE` | output (`.wav`; `.mp3` and others work if ffmpeg is installed) |
| `--ref-text "..."` | exact words of your recording; otherwise Whisper listens for you |
| `-y` | set up the Python environment without asking |
| `--list-models`, `--list-speakers`, `--list-voices`, `--setup [MODEL]` | look around, or only install |

## Which model, which speaker?

There is one model: Qwen3-TTS 1.7B, the largest model of the Qwen3-TTS family. It has two parts that are loaded as needed:
the Base model clones a voice, the CustomVoice model speaks with the built-in speakers. It supports 10 languages: en de fr es it pt ru zh ja ko.
It needs a GPU: an NVIDIA card (8+ GB is a rule of thumb) or an Apple Silicon Mac.

Built-in speakers: `aiden`, `ryan` (English), `vivian`, `serena`, `uncle_fu`, `dylan`, `eric` (Chinese), `ono_anna` (Japanese), `sohee` (Korean).
**Each speaker is native in one language and speaks the others with an accent.** For English, Chinese, Japanese and Korean the
speakers are the easy choice. For German, French and the rest, a clone of a recording by a native speaker sounds far more natural.

Licenses: Qwen3-TTS is Apache-2.0. This repository is MIT.

## What is in here

```
tts.py               the program: guided mode and one-command mode
start_server.cmd     keeps the voice service running (Windows)
start_server.sh      the same for Mac and Linux
src/engine.py        runs inside the private environment: prepares references, speaks
src/server.py        the HTTP voice service
requirements/        the Python packages of the private environment
examples/            two small texts to try
```

`voices/` (saved voices), `out/` (results) and `.venvs/` (the environment) are created on first use and are not part of the repository.

## What you need

- Python 3.9 or newer to run `tts.py` (it uses nothing but the standard library).
- [`uv`](https://docs.astral.sh/uv/) once, to build the model environments. It also fetches the right Python version itself.
  `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`, on Windows `irm https://astral.sh/uv/install.ps1 | iex`.
- A few GB of disk space and a few minutes on first use. Models and libraries are downloaded once.
- Optional: `ffmpeg` (reading unusual audio formats, writing mp3).

Everything runs in a private environment in `.venvs/`. Nothing is installed globally. Delete `.venvs/` to reset.

## Sound quality

The files are exactly what the model makes: 24 kHz, 16 bit WAV, nothing resampled or processed. The model has no higher resolution to give.
If a video or audio program insists on 48 kHz, convert it yourself, for example with
`ffmpeg -i in.wav -ar 48000 -c:a pcm_s24le out.wav`; that makes the file compatible but adds no detail.
What really decides how good it sounds: the model (there is only the largest one), a clean reference recording, and `--seed`
(try a few seeds and keep the best take). The HTTP service still answers with 24 kHz WAV files, as the Studio expects.

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
if a job hangs. The endpoints are listed at the top of [src/server.py](src/server.py). It uses Qwen3-TTS 1.7B by default (Base for cloning, CustomVoice for speakers).

Options: `--port`, `--voices-dir FOLDER` (where voices are kept, default `voices/`), `--whisper small` (a much smaller transcript model than the default `large-v3-turbo`), `--offline` (never contact Hugging Face).
To keep it running, use `start_server.cmd` (Windows) or `start_server.sh` (Mac/Linux); both restart the service if it ends.
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
- the same again with the final code layout (`src/`), set up from scratch with `tts.py`, plus the service endpoints for extra samples
  (add, listen, choose as template, delete), uploading, renaming and deleting a voice.

Tested on a Mac (Apple M5, 24 GB, Apple GPU), set up from scratch with the system Python 3.9: cloning with `qwen3-tts-1.7b` and a German
text of 256 characters took 26 seconds for 15 seconds of audio, and the text came back word for word when transcribed again. All nine
built-in speakers and 14 saved voices also ran there; the German clips came back word for word apart from one word in one clip.
`--serve --whisper small` ran there too: uploading a voice (with automatic transcript), `/generate` with a speaker and with the cloned voice,
`/transcribe`, renaming, extra samples and deleting.

**Not tested yet:** switching the real service over (the old one was left untouched), and Linux or CPU only.
The GPU memory hint (8+ GB) is a rule of thumb, not a measurement. If something fails, please open an issue with the output.

## Please use it responsibly

Only clone voices you have the right to clone: your own, or someone who clearly agreed. Don't use cloned voices to deceive,
impersonate or harass anyone, and check the laws and platform rules where you publish. The generated audio carries no watermark,
so say when it is synthetic.

## Troubleshooting

- **A `SoX could not be found` warning appears:** harmless, nothing here needs it.
- **`error: ... ran away` / a piece fails 3 times:** the model sometimes babbles on a sentence. Reword that sentence, shorten it, or change `--seed`.
- **CUDA out of memory:** close other programs that use the GPU (ComfyUI, games). The 1.7B model needs several GB on its own.
- **Windows with an NVIDIA card but it runs on the CPU:** delete `.venvs/` and run again; the setup installs the CUDA build of PyTorch.
- **Mac or CPU is slow:** on an Apple M5, 15 seconds of speech took about 26 seconds. CPU only is much slower.
- **The first run takes minutes before it speaks:** that is the model download (about 4.5 GB for the Base model, the same again for the CustomVoice model), later runs start in seconds.
- **The service answers "busy" (503):** it runs one job at a time; the client should wait and try again.

## How it works

`tts.py` detects your hardware, builds the right environment, and calls `src/engine.py` inside it.
For cloning, `engine.py` turns your recording into a clean 24 kHz reference (cut to at most 15 s, transcribed with Whisper for Qwen).
It then speaks the text in sentence-sized pieces and re-rolls a piece with another seed if it runs away or comes out silent.
Settings come from a lot of trial and error with Qwen3-TTS (repetition penalty 1.05, a hard cap on generated tokens, no post-processing).
`src/server.py` is the long-running version of the same idea behind an HTTP API.
