# easy-tts

Text to speech on your own computer, with voice cloning: give it a 10-second recording of a voice and let it say anything.
No account, no API key, everything runs locally.

## Quick start

```bash
git clone https://github.com/ari92p/easy-tts && cd easy-tts
python3 tts.py
```

`tts` on its own guides you: it asks for the language, lets you pick a saved voice or give a new recording (5 to 15 seconds, one person),
asks for the text and writes the result to `out/`. The first run also sets everything up and downloads the model (about 4 GB), once.

On Windows use `python` instead of `python3`. Want a plain `tts` command? On Mac and Linux: `chmod +x tts.py && ln -s "$PWD/tts.py" ~/.local/bin/tts`.

You need Python 3.9+, [`uv`](https://docs.astral.sh/uv/) (`brew install uv`, or `curl -LsSf https://astral.sh/uv/install.sh | sh`,
on Windows `irm https://astral.sh/uv/install.ps1 | iex`) and a GPU: an NVIDIA card or a Mac with Apple Silicon.

## Run it as command

With arguments nothing is asked: give the voice and the language, every time.

```bash
tts --voice klara --lang de "Hallo, das ist meine Stimme."        # a saved voice
echo "Hallo aus der Pipe" | tts --voice klara --lang de            # text from stdin
tts --voice klara --lang de --text-file chapter1.txt -o chapter1.wav
tts --clone me.wav --save-voice me --lang en "Hello"                # clone a recording and keep it as "me"
```

If something is missing, it says what (and lists your saved voices). Results go to `out/` (or where `-o` says).
Long texts are spoken sentence by sentence, several at a time, and joined. Afterwards the program listens to each sentence again with Whisper
and makes it again if it does not match the text (the model sometimes repeats a sentence or invents a word).

| Option | What it does |
| --- | --- |
| `--voice NAME` | a saved voice |
| `--clone FILE` | clone a voice from this recording (wav, mp3, m4a, ...); `--save-voice NAME` keeps it |
| `--lang de` | language of the text: en de fr es it pt ru zh ja ko (required) |
| `--text "..."`, `--text-file FILE`, or the text as an argument | what to say |
| `--clone-text "..."` | with `--clone`: the exact words of your recording (otherwise Whisper listens for you) |
| `--speed 1.2` | speaking speed: 1.0 as the voice speaks, above 1 faster, below 1 slower (0.5 to 2). The model has no speed setting, so this stretches the audio without changing the pitch: fine up to about 1.3, beyond that it starts to sound processed |
| `--seed N` | same seed and text give the same result; change it for another take |
| `-o FILE` | output file (`.mp3` and others work if ffmpeg is installed) |
| `--fast` | skip the listen-back check (about 10 seconds quicker, but a repeated or invented word can stay in) |
| `--list-voices` | show the voices you saved, with the language of their recording |
| `-v` | show everything the setup and the voice engine print |

Saved voices live in `voices/<name>/`; the language of the recording is kept in the `meta.json` of each voice.

## Run it as service

```bash
python3 tts.py --serve                      # http://127.0.0.1:8190
python3 tts.py --serve --host 0.0.0.0       # reachable from other computers (no login, use a firewall)
```

A small HTTP service for other programs: saved voices with several samples, Whisper transcripts, `POST /generate`.
The endpoints are listed at the top of [src/server.py](src/server.py). On start it downloads missing models (about 4.7 GB, with progress bars),
then answers one job at a time, frees the GPU after 5 minutes without requests and restarts itself if a job hangs.
Options: `--port`, `--voices-dir FOLDER`, `--whisper small`, `--offline` (never contact Hugging Face).

### Start files

`start_server.cmd` (Windows) and `start_server.sh` (Mac/Linux) are shortcuts for
`tts.py --serve --host 0.0.0.0 --port 8190`: they go to their own folder first, so you can start them from anywhere.

| System | How to start |
| --- | --- |
| Windows | double-click `start_server.cmd` (a console window stays open while the service runs) |
| Mac / Linux | `./start_server.sh` in a terminal (if it is not executable: `chmod +x start_server.sh`) |

- Close the window or press `Ctrl+C` to stop it. If the service stops by itself (for example a job hangs), it starts again after 5 seconds.
- They listen on **all network interfaces** (`0.0.0.0`) so other computers can reach port 8190. There is no login, so only do that in a network you trust,
  or change `--host 0.0.0.0` to `--host 127.0.0.1` in the file to keep it on your own computer.
- Edit the last line of the file to change the port, `--voices-dir`, `--whisper` or to add `--offline`.
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
| Whisper `small` | the transcript of your recording | about 0.5 GB |

Already have a model? Put its cache folder (e.g. `models--Qwen--Qwen3-TTS-12Hz-1.7B-Base`) into that cache and nothing is downloaded.

Inside the repository folder only this is created (none of it goes into git): `.venvs/` (the Python environment), `voices/` (saved voices) and `out/` (results).
Nothing is installed globally. Delete `.venvs/` to reset.

MIT license. Qwen3-TTS is Apache-2.0.
