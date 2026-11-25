# easy-tts-voice-clone

Text to speech on your own computer. Pick a built-in speaker, or clone a voice from a 10-second recording.
No account, no API key, everything runs locally.

## Quick start

```bash
git clone https://github.com/ari92p/easy-tts-voice-clone && cd easy-tts-voice-clone
python3 tts.py
```

The guided mode asks a few questions, sets everything up, makes your first clip and prints the one-line command that repeats it.
On Windows use `python` instead of `python3`.

You need Python 3.9+, [`uv`](https://docs.astral.sh/uv/) (`brew install uv`, or `curl -LsSf https://astral.sh/uv/install.sh | sh`,
on Windows `irm https://astral.sh/uv/install.ps1 | iex`) and a GPU: an NVIDIA card or a Mac with Apple Silicon.
The first run downloads PyTorch and the model (about 4 GB for the model), once.

## Use it

Choose exactly one voice:

| You type | What it means | Recording needed? |
| --- | --- | --- |
| `--speaker ryan` | a ready-made voice (list: `--list-speakers`) | no |
| `--clone me.wav` | clone a new voice from your recording | yes, once |
| `--voice me` | a cloned voice you saved earlier | no |

```bash
# a built-in speaker
python3 tts.py --speaker ryan --text "Hello, this is a built-in voice." --lang en
python3 tts.py --speaker aiden --style "slowly and sadly" --text "I miss the summer." --lang en

# clone your own voice and keep it for later
python3 tts.py --clone me.wav --save-voice me --text "First line" --lang en
python3 tts.py --voice me --text "Any later line" --lang en

# a long text from a file, to a file
python3 tts.py --voice me --text-file chapter1.txt --lang en -o chapter1.wav
```

Results go to `out/` (or where `-o` says). Long texts are spoken sentence by sentence and joined.

| Option | What it does |
| --- | --- |
| `--text "..."` / `--text-file FILE` | what to say |
| `--lang de` | language of the text: en de fr es it pt ru zh ja ko (default `auto`) |
| `--style "..."` | with `--speaker`: how to speak, e.g. "slowly and sadly" (plain English works best) |
| `--save-voice NAME` | with `--clone`: keep the voice for `--voice NAME` |
| `--clone-text "..."` | with `--clone`: the exact words of your recording (otherwise Whisper listens for you) |
| `--seed N` | same seed and text give the same result; change it for another take |
| `-o FILE` | output file (`.mp3` and others work if ffmpeg is installed) |
| `--list-speakers`, `--list-voices` | show what you can choose |
| `-y` | set up the Python environment without asking |

## Speakers and languages

Built-in speakers: `aiden`, `ryan` (English), `vivian`, `serena`, `uncle_fu`, `dylan`, `eric` (Chinese), `ono_anna` (Japanese), `sohee` (Korean).
**Each speaker is native in one language and speaks the others with an accent.** For English, Chinese, Japanese and Korean they are the easy choice.
For German, French and the rest, a clone of a recording by a native speaker sounds far more natural.

## A good recording for cloning

The recording matters more than any setting.

- 5 to 15 seconds of **one** person, spoken normally. Longer files are cut for you at a sentence end.
- No music, no echo, no second voice. A phone recording in a quiet room is fine.
- Speak the language you want to generate, ideally in the tone you want back.
- If the voice sounds off, try another recording first, then another `--seed`.

## Models, downloads and files

The model is Qwen3-TTS 1.7B (Apache-2.0), the largest of its family: the Base part clones, the CustomVoice part has the built-in speakers.
**You don't have to put any model anywhere.** On first use the program downloads what it needs from Hugging Face into its cache,
`~/.cache/huggingface/hub` (Windows: `C:\Users\<you>\.cache\huggingface\hub`, or the place `HF_HOME` points to):

| Download | When | Size |
| --- | --- | --- |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | cloning | about 4.2 GB |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | built-in speakers | about 4.2 GB |
| Whisper `small` | the transcript of your recording | about 0.5 GB |

Already have a model? Put its cache folder (e.g. `models--Qwen--Qwen3-TTS-12Hz-1.7B-Base`) into that cache and nothing is downloaded.

Inside the repository folder only this is created (none of it goes into git): `.venvs/` (the Python environment), `voices/` (saved voices) and `out/` (results).
Nothing is installed globally. Delete `.venvs/` to reset.

The audio is exactly what the model makes, 24 kHz, 16 bit WAV, nothing resampled. If a program insists on 48 kHz:
`ffmpeg -i in.wav -ar 48000 -c:a pcm_s24le out.wav` makes it compatible, but adds no detail.

## Run it as a voice service

```bash
python3 tts.py --serve                      # http://127.0.0.1:8190
python3 tts.py --serve --host 0.0.0.0       # reachable from other computers (no login, use a firewall)
```

A small HTTP service for other programs: speakers, saved voices with several samples, Whisper transcripts, `POST /generate`.
The endpoints are listed at the top of [src/server.py](src/server.py). On start it downloads missing models (about 8.4 GB for the two Qwen models plus Whisper, with progress bars),
then answers one job at a time, frees the GPU after 5 minutes without requests and restarts itself if a job hangs.
Options: `--port`, `--voices-dir FOLDER`, `--whisper small`, `--offline` (never contact Hugging Face).
`start_server.cmd` (Windows) and `start_server.sh` (Mac/Linux) start it and keep it running.

## Troubleshooting

- **A `SoX could not be found` warning:** harmless.
- **`ran away` or a piece fails 3 times:** the model sometimes babbles on one sentence. Reword or shorten it, or change `--seed`.
- **CUDA out of memory:** close other programs that use the GPU. The model needs several GB on its own.
- **Windows with an NVIDIA card but it runs on the CPU:** delete `.venvs/` and run again; the setup installs the CUDA build of PyTorch.
- **The first run is slow before it speaks:** that is the model download. Later runs start in seconds.
- **The service answers 503 "busy":** it runs one job at a time; try again.

## Tested

Windows with an NVIDIA RTX 5080 (16 GB) and a Mac with Apple M5 (24 GB, Apple GPU): cloning, all built-in speakers and the voice service,
each time checked by transcribing the generated audio again with Whisper. On the Mac, 15 seconds of speech took about 26 seconds.
**Not tested:** Linux and CPU only. If something fails, please open an issue with the output.

## Please use it responsibly

Only clone voices you have the right to clone: your own, or someone who clearly agreed. Don't use cloned voices to deceive,
impersonate or harass anyone, and check the laws and platform rules where you publish. The audio carries no watermark, so say when it is synthetic.

MIT license. Qwen3-TTS is Apache-2.0.
