# easy-tts

Text to speech with voice cloning, and voice to text, on your own computer.
Give it a 10-second recording of a voice and let it say anything, or turn an audio file into text. No account, no API key, everything runs locally.

- **Text to speech:** clone a voice from a short recording ([Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base), 10 languages).
- **Voice to text:** write down what is said in an audio file ([Whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3-turbo`).
- **Voice service:** the same as a small HTTP service for other programs.

## Quick start

```bash
git clone https://github.com/ari92p/easy-tts && cd easy-tts
python3 easy-tts.py
```

`easy-tts` on its own guides you: it asks for the language, lets you pick a saved voice or give a new recording (5 to 15 seconds, one person),
asks for the text and writes the result to `out/`. For a new recording, Whisper writes down what is said (`voices/<name>/ref.txt`, you can correct it there by hand).
The first run also sets everything up and downloads the models (about 6 GB), once.

On Windows use `python` instead of `python3`. Want a plain `easy-tts` command? On Mac and Linux: `chmod +x easy-tts.py && ln -s "$PWD/easy-tts.py" ~/.local/bin/easy-tts` (the examples below use it; with the file alone, write `python3 easy-tts.py` instead).

You need Python 3.9+ and a GPU: an NVIDIA card or a Mac with Apple Silicon. Everything else (the Python setup tool `uv`, the packages, the models) is installed by the first run, inside this folder.

## Run it as command

With arguments nothing is asked. A new recording needs its language once (`--lang`); a saved voice remembers it.

```bash
easy-tts --clone me.wav --save me --lang en "Hello, this is my voice."    # clone a recording and keep it as "me" (with its language)
easy-tts --clone me.wav --save me --lang en                                    # only keep the voice, no text: nothing is spoken
easy-tts --voice me "Hello again."                                             # use the saved voice, no --lang needed
echo "Hello from a pipe" | easy-tts --voice me                                 # text from stdin
easy-tts --voice me --text-file chapter1.txt -o chapter1.wav                   # a long text from a file
```

If something is missing, it says what (and lists your saved voices). Results go to `out/` (or where `-o` says).
Long texts are spoken sentence by sentence, several at a time, and joined. Afterwards the program listens to each sentence again with Whisper
and makes it again if it does not match the text (the model sometimes repeats a sentence or invents a word).

| Option | What it does |
| --- | --- |
| `--voice NAME` | a saved voice |
| `--clone FILE` | clone a voice from this recording (wav, mp3, m4a, ...); `--save NAME` keeps it |
| `--lang de` | language of the text: en de fr es it pt ru zh ja ko. Required with `--clone`; a saved voice remembers it (give `--lang` to speak another language with that voice) |
| `--text "..."`, `--text-file FILE`, or the text as an argument | what to say |
| `--speed 1.2` | speaking speed: 1.0 as the voice speaks, above 1 faster, below 1 slower (0.5 to 2). The model has no speed setting, so this stretches the audio without changing the pitch: fine up to about 1.3, beyond that it starts to sound processed |
| `--seed N` | same seed and text give the same result; change it for another take |
| `-o FILE` | output file (`.mp3` and others work if ffmpeg is installed) |
| `--fast` | skip the listen-back check: much quicker, but a repeated or invented word can stay in |
| `--transcribe AUDIO` | voice to text: writes what is said in an audio file to `out/<name>-<time>.txt` (Whisper). Add `--lang de` if you know the language, `-o text.txt` to choose the file |
| `--list-voices` | show the voices you saved, with their language |

Saved voices live in `voices/<name>/`: `ref.wav`, `ref.txt` (what is said) and `meta.json` (the language).

## Voice to text

```bash
easy-tts --transcribe interview.mp3                # writes out/interview-<time>.txt
easy-tts --transcribe interview.mp3 --lang de      # if you know the language (otherwise it is detected)
easy-tts --transcribe interview.mp3 -o notes.txt   # choose the file
```

Whisper `large-v3-turbo` runs on your own computer; nothing is uploaded.

## Run it as service

```bash
python3 easy-tts.py --serve                      # http://127.0.0.1:8190
python3 easy-tts.py --serve --host 0.0.0.0       # reachable from other computers (no login, use a firewall)
```

A small HTTP service for other programs: saved voices with several samples, Whisper transcripts, `POST /generate`.
The endpoints are listed at the top of [src/server.py](src/server.py). On start it downloads missing models (about 5.7 GB, with progress bars),
then answers one job at a time, frees the GPU after 5 minutes without requests and restarts itself if a job hangs.
Options: `--port`, `--voices-dir FOLDER`, `--offline` (never contact Hugging Face).

### Start files

`start_server.cmd` (Windows) and `start_server.sh` (Mac/Linux) are shortcuts for
`easy-tts.py --serve --host 0.0.0.0 --port 8190`: they go to their own folder first, so you can start them from anywhere.

| System | How to start |
| --- | --- |
| Windows | double-click `start_server.cmd` (a console window stays open while the service runs) |
| Mac / Linux | `./start_server.sh` in a terminal (if it is not executable: `chmod +x start_server.sh`) |

- Close the window or press `Ctrl+C` to stop it. If the service stops by itself (for example a job hangs), it starts again after 5 seconds.
- They listen on **all network interfaces** (`0.0.0.0`) so other computers can reach port 8190. There is no login, so only do that in a network you trust,
  or change `--host 0.0.0.0` to `--host 127.0.0.1` in the file to keep it on your own computer.
- Edit the last line of the file to change the port or `--voices-dir`, or to add `--offline`.
- To start it with the computer, run the file from Windows Task Scheduler (trigger: at log on) or from the login items of your system.

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
| Whisper `large-v3-turbo` | the transcript of your recording and the check of the result | about 1.5 GB |

Already have a model? Put its cache folder (e.g. `models--Qwen--Qwen3-TTS-12Hz-1.7B-Base`) into that cache and nothing is downloaded.

Inside the repository folder only this is created (none of it goes into git): `.venvs/` (the Python environment), `voices/` (saved voices) and `out/` (results).
Nothing is installed globally. Delete `.venvs/` to reset.

MIT license. Qwen3-TTS is Apache-2.0, Whisper `large-v3-turbo` is MIT.
