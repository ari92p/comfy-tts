# easy-tts

Text to speech on your own computer, with voice cloning: give it a 10-second recording of a voice and let it say anything.
No account, no API key, everything runs locally.

## Quick start

```bash
git clone https://github.com/ari92p/easy-tts && cd easy-tts
python3 tts.py
```

The guided mode asks a few questions, sets everything up, clones your voice for a first clip and prints the one-line command that repeats it.
On Windows use `python` instead of `python3`.

You need Python 3.9+, [`uv`](https://docs.astral.sh/uv/) (`brew install uv`, or `curl -LsSf https://astral.sh/uv/install.sh | sh`,
on Windows `irm https://astral.sh/uv/install.ps1 | iex`) and a GPU: an NVIDIA card or a Mac with Apple Silicon.
The first run downloads PyTorch and the model (about 4 GB for the model), once.

## Run it as command

Choose exactly one voice:

| You type | What it means | Recording needed? |
| --- | --- | --- |
| `--clone me.wav` | clone a new voice from your recording | yes, once |
| `--voice me` | a cloned voice you saved earlier | no |

```bash
# clone a voice from a recording and keep it for later
python3 tts.py --clone me.wav --save-voice me --text "Hello, this is my cloned voice." --lang en

# use the saved voice any time
python3 tts.py --voice me --text "Any later line" --lang en

# a long text from a file, to a file
python3 tts.py --voice me --text-file chapter1.txt --lang en -o chapter1.wav
```

Results go to `out/` (or where `-o` says). Long texts are spoken sentence by sentence and joined.

| Option | What it does |
| --- | --- |
| `--text "..."` / `--text-file FILE` | what to say |
| `--lang de` | language of the text: en de fr es it pt ru zh ja ko (default `auto`) |
| `--save-voice NAME` | with `--clone`: keep the voice for `--voice NAME` |
| `--clone-text "..."` | with `--clone`: the exact words of your recording (otherwise Whisper listens for you) |
| `--seed N` | same seed and text give the same result; change it for another take |
| `-o FILE` | output file (`.mp3` and others work if ffmpeg is installed) |
| `--list-voices` | show the voices you saved |
| `-y` | set up the Python environment without asking |

## Run it as service

```bash
python3 tts.py --serve                      # http://127.0.0.1:8190
python3 tts.py --serve --host 0.0.0.0       # reachable from other computers (no login, use a firewall)
```

A small HTTP service for other programs: saved voices with several samples, Whisper transcripts, `POST /generate`.
The endpoints are listed at the top of [src/server.py](src/server.py). On start it downloads missing models (about 4.7 GB, with progress bars),
then answers one job at a time, frees the GPU after 5 minutes without requests and restarts itself if a job hangs.
Options: `--port`, `--voices-dir FOLDER`, `--whisper small`, `--offline` (never contact Hugging Face).
`start_server.cmd` (Windows) and `start_server.sh` (Mac/Linux) start it and keep it running.

## Languages

en, de, fr, es, it, pt, ru, zh, ja, ko. Set the language of your text with `--lang`.

## A good recording for cloning

The recording matters more than any setting.

- 5 to 15 seconds of **one** person, spoken normally. Longer files are cut for you at a sentence end.
- No music, no echo, no second voice. A phone recording in a quiet room is fine.
- Speak the language you want to generate, ideally in the tone you want back.
- If the voice sounds off, try another recording first, then another `--seed`.

## Models, downloads and files

The model is Qwen3-TTS 1.7B Base (Apache-2.0), the largest of its family. It clones voices; there are no built-in speakers.
**You don't have to put any model anywhere.** On first use the program downloads what it needs from Hugging Face into its cache,
`~/.cache/huggingface/hub` (Windows: `C:\Users\<you>\.cache\huggingface\hub`, or the place `HF_HOME` points to):

| Download | When | Size |
| --- | --- | --- |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | always | about 4.2 GB |
| Whisper `small` | the transcript of your recording | about 0.5 GB |

Already have a model? Put its cache folder (e.g. `models--Qwen--Qwen3-TTS-12Hz-1.7B-Base`) into that cache and nothing is downloaded.

Inside the repository folder only this is created (none of it goes into git): `.venvs/` (the Python environment), `voices/` (saved voices) and `out/` (results).
Nothing is installed globally. Delete `.venvs/` to reset.

MIT license. Qwen3-TTS is Apache-2.0.
