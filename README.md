# easy-tts

Text to speech with voice cloning, and voice to text, on your own computer.
Give it a 10-second recording of a voice and let it say anything, or turn an audio file into text. No account, no API key, everything runs locally.

- **Text to speech:** clone a voice from a short recording ([Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base), 10 languages).
- **Voice to text:** write down what is said in an audio file ([Whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3-turbo`).
- **Voice service:** the same as a small HTTP service for other programs.

## Quick start

```bash
git clone https://github.com/ari92p/easy-tts && cd easy-tts
src/setup.sh     # once (Windows: double-click src\setup.cmd)
./easy-tts           # Windows: easy-tts.cmd
```

## Run it as command

```bash
easy-tts --clone me.wav --save me --lang en "Hello, this is my voice."    # clone a recording and keep it as "me" (with its language)
easy-tts --clone me.wav --save me --lang en                                    # only keep the voice, no text: nothing is spoken
easy-tts --voice me "Hello again."                                             # use the saved voice, no --lang needed
echo "Hello from a pipe" | easy-tts --voice me                                 # text from stdin
easy-tts --voice me --text-file chapter1.txt -o chapter1.wav                   # a long text from a file
```

## Run it as service

```bash
easy-tts --serve                      # http://127.0.0.1:8190
easy-tts --serve --host 0.0.0.0       # reachable from other computers (no login, use a firewall)
```

## Voice to text

```bash
easy-tts --transcribe interview.mp3                # writes out/interview-<time>.txt
easy-tts --transcribe interview.mp3 --lang de      # if you know the language (otherwise it is detected)
easy-tts --transcribe interview.mp3 -o notes.txt   # choose the file
```

Results go to `out/` (or where `-o` says).

| Parameters | What it does |
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
| `--port` | port you want to host the service on |
| `--voices-dir FOLDER` | change the voices collection folder |

## A good recording for cloning

The recording matters more than any setting.

- 5 to 15 seconds of **one** person speaking. Longer files are cut for you at a sentence end.
- Speak the language you want to generate, ideally in the tone you want back.
- If the voice sounds off, try another recording first, then another `--seed`.

## Models, downloads and files

The model is Qwen3-TTS 1.7B Base, the largest of its family. It clones voices; there are no built-in speakers.
**You don't have to put any model anywhere.** On first use the program downloads what it needs from Hugging Face into its cache.

| Download | When | Size |
| --- | --- | --- |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | always | about 4.2 GB |
| Whisper `large-v3-turbo` | the transcript of your recording and the check of the result | about 1.5 GB |

Already have a model? Put its cache folder (e.g. `models--Qwen--Qwen3-TTS-12Hz-1.7B-Base`) into that cache folder and nothing is downloaded.

## Folders
saved voices: `voices/`
results: `out/`
models: `~/.cache/huggingface/hub` (Windows: `C:\Users\<you>\.cache\huggingface\hub`, or the place `HF_HOME` points to) 

MIT license. Qwen3-TTS is Apache-2.0, Whisper `large-v3-turbo` is MIT.
