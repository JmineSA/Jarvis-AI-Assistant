# JARVIS

A modular, local, voice-controlled personal AI assistant for Windows, built step by step.

**Current status: Phase 5 (memory, web interface and system monitoring).**

| Phase | Scope | Status |
|---|---|---|
| 1 | Project skeleton, config, logging, Ollama client, brain, FastAPI `/chat`, tests | Done |
| 2 | Microphone, Faster-Whisper, Piper TTS, push-to-talk | Built: needs testing on your PC |
| 3 | Validated tools (open apps and folders, CPU/RAM, date/time, list files) | Built: needs testing on your PC |
| 4 | SQLite memory, BM25 retrieval into the model's context, `remember`/`recall`/`forget` | Done |
| 5 | Streamlit web interface, live system monitor, health report | Done |

## Architecture in one minute

- `ai/` talks to Ollama and decides what to say. It never touches the operating system.
- `tools/` is the only code allowed to act on the computer: a fixed registry of typed, validated tools. No arbitrary shell commands. What it may open comes only from `config/profile.yaml`.
- `memory/` (Phase 4) is the only code that touches SQLite. It stores the conversation and the facts you keep, and finds the ones that matter for the question you just asked.
- `voice/` (Phase 2) captures and plays audio.
- `monitor/` (Phase 5) reads the state of the machine and checks that every part of JARVIS works.
- `ui/` (Phase 5) is the web interface. It can use the running API server, or run the brain in its own process.
- `api/` exposes everything over HTTP on localhost.

## Requirements

- Windows 10 (64-bit), Python 3.11 (64-bit), Git
- [Ollama](https://ollama.com) installed and running
- Recent NVIDIA driver (GTX 970, 4 GB VRAM, is enough for the default 3B model)

## Setup (PowerShell)

```powershell
cd C:\Dev\jarvis
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .

ollama pull qwen2.5:3b-instruct
ollama ps                              # while a model is loaded, PROCESSOR should say GPU

copy .env.example .env                 # optional: only if you want to override settings
```

## Run

Start the API server:

```powershell
python -m jarvis
```

Then open http://127.0.0.1:8000/docs, or use PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod -Method Post http://127.0.0.1:8000/chat -ContentType "application/json" -Body '{"message":"Hello JARVIS"}'
Invoke-RestMethod -Method Post http://127.0.0.1:8000/reset
```

Or chat directly in the terminal, without the server:

```powershell
python -m jarvis chat
```

## Talk to JARVIS (Phase 2)

Install the new voice packages (once), then check each part before talking:

```powershell
pip install -r requirements-dev.txt
python -m jarvis download-models      # Whisper (about 250 MB) + the Piper voice (about 60 MB), first time only
python -m jarvis voice-check          # tests speakers, microphone and speech recognition, step by step
python -m jarvis voice                # hold Right Ctrl, speak, release
```

How a turn works: hold the push-to-talk key, speak, release. JARVIS transcribes you (Faster-Whisper on the CPU), asks the language model, and speaks the answer (Piper). Press the key again while it is speaking to interrupt it.

No global hotkey available? Use `python -m jarvis voice --enter` and press Enter to start and stop talking.

`voice-check` prints how long recognition took. If replies feel slow, set `voice.stt.model: base.en` in `config/settings.yaml` (faster, slightly less accurate).

| Voice setting (`config/settings.yaml`) | Meaning |
|---|---|
| `voice.push_to_talk_key` | `right_ctrl` by default. Examples: `f8`, `scroll_lock`, `ctrl+alt+j`. Avoid `ctrl+space`: VS Code uses it. |
| `voice.input_device` / `voice.output_device` | Microphone and speakers: a number or part of the name. `voice-check` lists them. |
| `voice.stt.model` | `small.en` (default) or `base.en` |
| `voice.tts.voice` | Piper voice name, default `en_GB-alan-medium` |
| `voice.tts.length_scale` | Speaking speed: below 1 is faster, above 1 is slower |

Logs of every turn, with timings for recognition, the language model and speech, are in `logs/jarvis.log`.

## Tools: what JARVIS can do (Phase 3)

Hold the push-to-talk key and try:

| Say | What happens |
|---|---|
| "JARVIS, open VS Code" | Starts VS Code |
| "Open VS Code in LightsOn" | Starts VS Code with that project folder |
| "Open Jupyter in LightsOn" | Starts Jupyter Lab inside that folder |
| "Open Power BI" / "Open Chrome" | Starts the app |
| "Open Databricks" / "Open my GitHub" | Opens the saved website in Chrome |
| "Open the Dune folder" | Opens it in File Explorer |
| "What's in the LightsOn folder?" | Reads out the first items (read-only) |
| "How's my CPU and memory?" | CPU and RAM usage |
| "What time is it?" | Time and date |

**Every answer comes from what the tool really reported**, not from the language model's own words. If something can't be done, JARVIS says why, for example "I couldn't find Power BI on this computer. Add its location to config/profile.yaml." If the model claims it opened something without calling a tool, JARVIS says so instead of repeating the claim.

**Your profile:** `config/profile.yaml` lists the apps, folders and websites JARVIS may open. Nothing outside it can be opened. Edit it, then restart JARVIS:

- Check every folder path (the LightsOn path is a guess).
- Put your Databricks workspace address under `websites > databricks > url`.
- If Power BI is installed from the Microsoft Store, put the path to `PBIDesktop.exe` under its `candidates`.
- To add an app: copy an existing entry, give it a `label`, other `names` you might say, and `candidates` (a program name or a full path).

**Confirmations:** the confirm mechanism is built in, and the AI cannot approve its own actions. The current tools only open things or read information, so they run straight away. To make any tool ask first, list it in `config/settings.yaml`, for example `always_confirm: [open_app]`. JARVIS then asks `Allow? [y/N]` in the PowerShell window, and only your keyboard answers.

Logs in `logs/jarvis.log` record every tool request and its result.

## Memory (Phase 4)

JARVIS keeps a SQLite database in `data/jarvis_memory.db` and reads the relevant parts of it into the model's context **before every question**. It stores two kinds of memory:

- **The conversation.** Every exchange is saved, so a later session can pick up where the last one left off.
- **Facts.** Short, curated statements you ask it to keep, such as "the LightsOn project lives in D:\Projects\LightsOn".

Retrieval is a small BM25 ranker written in plain Python. It needs no model, no network and no extra dependency, so memory keeps working even when Ollama is down.

Ask it to remember something, or search its memory:

```powershell
python -m jarvis remember "I work on the LightsOn project" --key "lightson.project"
python -m jarvis memory "LightsOn"
python -m jarvis forget 3          # delete one memory by id
python -m jarvis forget --all      # start afresh
```

In the terminal chat you can also type `remember <fact>` and `recall <words>` directly.

By voice, three tools do the same thing:

| Say | What happens |
|---|---|
| "Remember that I take my coffee black" | The fact is saved with a key of its own |
| "What do you remember about LightsOn?" | The memory is searched and the answer comes from what was found |
| "Forget memory 7" | That memory is deleted (you are asked to confirm first) |

Facts you store by hand always win over a guessed conversation match, so a corrected detail sticks.

**Memory settings** (`config/settings.yaml`):

| Setting | Meaning |
|---|---|
| `memory.enabled` | `false` turns the memory off completely |
| `memory.database_path` | Where the SQLite file lives (`data/jarvis_memory.db`) |
| `memory.max_results` | How many retrieved memories go into the model's context |
| `memory.max_context_chars` | Hard limit on the injected block, so the context window survives |
| `memory.recent_turns` | Recent messages always included, even when the question matches nothing |
| `memory.min_score` | Below this score a match counts as noise |
| `memory.save_conversations` | `false` stores only the facts, not the conversation |
| `memory.retention_days` | `0` keeps everything; otherwise older messages are deleted |

## The web interface (Phase 5)

```powershell
python -m jarvis ui
```

This opens http://127.0.0.1:8501. The interface has three parts:

- **The chat**, which is the same brain as the terminal and voice modes. JARVIS answers from what its tools really reported.
- **The system panel**, which shows the live state of your machine: CPU per core, memory, the GPU (through `nvidia-smi`), network speed and every disk. It refreshes on its own.
- **The memory browser**, where you can search what JARVIS remembers, add a fact by hand and delete anything it should forget.

The interface uses the running API server when one answers, and runs the brain in its own process when it does not. To force one or the other, set `ui.backend` in `config/settings.yaml` to `api` or `local`.

## Check that everything works (Phase 5)

```powershell
python -m jarvis status
```

It prints the health of every part of JARVIS (the model, the memory, the tools, the voice models) and a reading of the machine: CPU, memory, disks, GPU and uptime. The exit code is `0` when everything is healthy.

## Test

```powershell
pytest
```

The tests use a mocked Ollama, so they pass even when Ollama is not running.

## Configuration

Defaults live in `config/settings.yaml`. Environment variables (or a `.env` file) override them:

| Variable | Meaning |
|---|---|
| `JARVIS_OLLAMA_HOST` | Ollama URL (default `http://127.0.0.1:11434`) |
| `JARVIS_OLLAMA_MODEL` | Model name (default `qwen2.5:3b-instruct`) |
| `JARVIS_OLLAMA_TIMEOUT` | Seconds to wait for a reply |
| `JARVIS_API_HOST` / `JARVIS_API_PORT` | Where the API listens (keep the host on `127.0.0.1`) |
| `JARVIS_LOG_LEVEL` / `JARVIS_LOG_DIR` | Logging |
| `JARVIS_CONFIG` | Path to another settings file |
| `JARVIS_MEMORY_DB` | Path to the SQLite memory file |
| `JARVIS_MEMORY_ENABLED` | `false` turns the memory off |
| `JARVIS_API_URL` | The API server the web interface should use |

Never put secrets in `settings.yaml`; use `.env`, which is git-ignored.

Logs go to the console and to `logs/jarvis.log` (rotating).

## The HTTP API

| Method and path | What it does |
|---|---|
| `GET /health` | Is Ollama reachable and is the model installed? |
| `POST /chat` | Send a message, get a reply |
| `POST /reset` | Clear the conversation context |
| `GET /status` | Health of every part of JARVIS plus a reading of the machine |
| `GET /memory?q=…` | Search the memory (no query returns the newest memories) |
| `GET /memory/facts` | Every stored fact |
| `POST /memory/facts` | Store a fact by hand |
| `DELETE /memory/{id}` | Delete one memory |
| `POST /memory/clear` | Clear this conversation's messages (the facts stay) |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/health` says `degraded`, Ollama unreachable | Start Ollama (`ollama serve`, or launch the Ollama app). |
| `/health` says the model is not installed | `ollama pull qwen2.5:3b-instruct` |
| `/chat` returns 503 | Same as the first row. |
| `voice-check` says no sound detected | Check Windows Settings > Privacy > Microphone (allow desktop apps), the mute switch, and `voice.input_device`. |
| Push-to-talk does nothing | Try another key (`f8`), or run `python -m jarvis voice --enter` to test the rest. Some games and admin windows block global hotkeys. |
| Model download fails | Needs internet on first run only. Run `python -m jarvis download-models` again. |
| JARVIS says it couldn't find an app | Add the app's location to `candidates` in `config/profile.yaml`. |
| JARVIS says it didn't select a tool | The small model answered in words instead of acting. Say the command again, shorter ("open Chrome"). |
| `/chat` or voice gives an HTTP 400 about tools | Your Ollama is too old or the model does not support tools. Update Ollama; `qwen2.5:3b-instruct` supports them. |
| `/chat` returns 504 | The model is slow. Check `ollama ps` for GPU use, or raise `JARVIS_OLLAMA_TIMEOUT`. |
| Replies are slow | If `ollama ps` shows CPU, update the NVIDIA driver and restart Ollama. Avoid models above 3B parameters on 4 GB VRAM. |
| `python -m jarvis ui` says Streamlit is missing | `pip install -r requirements-dev.txt` |
| The web interface says the backend is offline | Start the API server with `python -m jarvis`, or set `ui.backend: local`. |
| The system panel shows no GPU | `nvidia-smi` was not found. Install the NVIDIA driver, or ignore the panel: it is decoration, not a requirement. |
| JARVIS forgets things between sessions | `memory.enabled` is `false`, or `memory.database_path` points somewhere new each run. Check `python -m jarvis status`. |
| The wrong memory keeps coming back | Delete it in the memory browser, or run `python -m jarvis forget <id>`. |
