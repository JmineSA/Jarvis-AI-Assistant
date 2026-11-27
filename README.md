# JARVIS

A modular, local, voice-controlled personal AI assistant for Windows, built step by step.

**Current status: Phase 5 (voice, tools, memory and reminders, weather, search, email drafts, music, app finder).**

| Phase | Scope | Status |
|---|---|---|
| 1 | Project skeleton, config, logging, Ollama client, brain, FastAPI `/chat`, tests | Done |
| 2 | Microphone, Faster-Whisper, Piper TTS, push-to-talk | Built: needs testing on your PC |
| 3 | Validated tools (open apps and folders, CPU/RAM, date/time, list files) | Built: needs testing on your PC |
| 4 | SQLite memory: facts, past conversations, reminders | Built: needs testing on your PC |
| 5 | Weather, web search, Gmail drafts, music in VLC, app finder | Built: needs testing on your PC |
| 6 | Desktop dashboard (PySide6) | Done |

## Architecture in one minute

- `ai/` talks to Ollama and decides what to say. It never touches the operating system.
- `tools/` is the only code allowed to act on the computer: a fixed registry of typed, validated tools. No arbitrary shell commands. What it may open comes only from `config/profile.yaml`.
- `memory/` is the only code that touches SQLite (`data/jarvis.db`).
- `voice/` (Phase 2) captures and plays audio.
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

## Memory and reminders (Phase 4)

JARVIS remembers only what you ask it to, plus your past conversations (which you can switch off).

| Say | What happens |
|---|---|
| "Remember that the LightsOn project is due on 1 December" | Saves the fact. It is shown to the model every turn from then on |
| "What do you know about LightsOn?" / "What do you know about me?" | Reads back matching saved facts and related past conversations |
| "Forget the LightsOn deadline" | Asks you to confirm (naming the exact fact), then deletes it |
| "Remind me to call the bank in 20 minutes" | Sets a reminder |
| "What are my reminders?" / "Cancel the bank reminder" | Lists or cancels |

**Times understood:** "in 20 minutes", "in an hour", "in half an hour", "tomorrow at 9am", "tonight", "this evening", "at 3pm", "at 15:30", "next Monday at 10:30", "1 December", "December 1st at 2pm". A date with no time means 09:00. A bare "at 5" means 5pm (1-7 and 12 are afternoon, 8-11 are morning). If JARVIS isn't sure of the time it asks instead of guessing.

**Reminders are spoken only while JARVIS is running** (voice or chat mode), and only when it is idle, never mid-conversation. A reminder that came due while JARVIS was off is announced on the next start as "I missed a reminder from ...". Each reminder is announced at most once.

**See and delete everything it holds:**

```powershell
python -m jarvis memory                  # facts, reminders, conversation totals, database location
python -m jarvis memory search battery   # search past conversations
python -m jarvis memory forget 3         # delete one saved fact (the [number] shown by `memory`)
python -m jarvis memory cancel 2         # cancel one reminder
python -m jarvis memory wipe-history     # delete stored conversations only
python -m jarvis memory wipe             # delete EVERYTHING (you must type DELETE)
```

**Privacy:** everything stays in `data/jarvis.db` on your PC, which is git-ignored. In `config/settings.yaml` under `memory:` set `save_conversations: false` to keep facts and reminders but stop storing conversations, or `enabled: false` to turn memory off entirely. If the database cannot be opened, JARVIS carries on without memory.

## Weather, search, email, music and more apps (Phase 5)

| Say | What happens |
|---|---|
| "What's the weather?" | Current conditions in your city plus today's high, low and chance of rain |
| "What's the weather tomorrow?" / "on Friday?" | Daily summary (up to 7 days ahead) |
| "What will it be like at 2pm?" / "tomorrow morning?" | Forecast for that hour |
| "Search Google for the best current action movies" | Opens the results in Chrome |
| "Search YouTube for lofi beats" / "Search YouTube Music for ..." | Opens those results (you press play) |
| "Open my email" | Opens Gmail in Chrome |
| "Write an email to Mum about Sunday lunch" | Opens a Gmail **draft** with the recipient, subject and text filled in. **JARVIS never sends it**: you read it and press send |
| "Open YouTube Music" / "Open FL Studio" / "Open VLC" | Starts it |
| "Play the Kendrick Lamar GNX album in VLC" | Finds the closest album folder in your music folder and plays it |
| "Play some music" | Plays your whole music folder |

**Set these up in `config/profile.yaml`:**

- `location`: your city and its coordinates. It is set to Pretoria. Only these coordinates are sent to the weather service (Open-Meteo, free, no account or key). Weather is the only feature that sends anything over the internet.
- `music: folder`: **placeholder `G:\Music`; change it to your real music folder.** Albums can sit directly in it or one level down (Artist/Album). Folder names need to be readable ("Daft Punk - Discovery" is better than "DP_disc1"). If two albums match equally, JARVIS names them and asks you to be more specific. Also update `folders: music` if you want "open my music folder".
- `contacts`: optional names and email addresses for "write an email to Mum". An unknown name still opens a draft with the recipient left blank, and JARVIS tells you so.
- Searching, drafts and weather open or read web pages only. JARVIS does not read search results or your inbox.

**Adding apps: the app finder**

```powershell
python -m jarvis add-app "FL Studio"
```

It looks in the usual install folders and the Windows list of registered apps (it never runs anything and never scans your whole drive), shows the matches, and asks you to pick one, name it, and add other names you might say. It then adds the entry under `apps:` in `config/profile.yaml`, keeps your comments, and saves a backup as `profile.yaml.bak`. Restart JARVIS and say "open FL Studio". Microsoft Store apps often have no normal `.exe`, so for those add the path by hand.

## The desktop app (Phase 6)

A real Windows window instead of a terminal: Command Center, Conversations, Memory, Tools, Apps and Music, Settings.

```powershell
pip install -r requirements.txt      # adds PySide6-Essentials, about 70 MB
python -m jarvis app                 # window + voice (hold Right Ctrl, or tap TALK TO JARVIS)
python -m jarvis app --no-voice      # window only, type commands
```

- The window only shows and drives the same brain and tools as before. It adds no new way to act, and anything risky still asks you in a dialog.
- **Talk button:** tap once to start recording, tap again to send. The Right Ctrl hotkey works exactly as before.
- **Typed replies are spoken** when voice is running. Turn that off with `ui.speak_typed_replies: false` in `config/settings.yaml`.
- **Quick commands** are built from your `profile.yaml` (apps, location, music folder, Gmail).
- **Memory page:** see, add and delete facts, cancel reminders, clear conversations, or erase everything (you must type DELETE).
- **Apps and Music page:** a red dot means a program or folder isn't found on this PC. Fix the path in `profile.yaml` or run `python -m jarvis add-app "Name"`.
- **Fonts:** looks best with Orbitron and Inter. Drop the `.ttf` files in `src/jarvis/ui/fonts/`; otherwise Bahnschrift and Segoe UI are used.
- `JARVIS_REDUCE_MOTION=1` stops the spinning rings and waveform.
- The GPU gauge reads `nvidia-smi`; it shows n/a if that isn't available.

If you unzip over an existing project, **merge** `config/profile.yaml` instead of overwriting it, so your own apps and paths stay.

## Speed and music fixes (Phase 6.1)

**Why the GPU hit 100% and replies were slow:** every question made the model re-read about 2,000 tokens of instructions. Now:
- The start of the prompt never changes, so Ollama reuses its cached copy. The date and time are added after your words instead of at the top.
- The model is loaded when the app starts and stays loaded (`ollama.keep_alive`), so the first answer is quick.
- Replies are capped (`ollama.num_predict`) and only the last 4 exchanges are remembered (`brain.max_history_turns`).
- **Obvious commands skip the model entirely** (`brain.fast_commands`): open an app/folder/site, play music, weather, time, system status, search, remember, remind. They answer in milliseconds. Anything else goes to the model as before. Set `fast_commands: false` to turn this off.
- The dashboard's *Language model* card shows `GPU 100%` (or `CPU`) and the time of the last reply. "instant" means the model wasn't needed.

`python -m jarvis speed-check` times the model, shows whether it runs on the GPU, and prints plain advice.

**Music**
- `play music` plays your whole music folder, shuffled, in VLC.
- `play <song>`: when you are **online** it plays the top YouTube Music result (your main streaming site; change it with `music.online` in `profile.yaml`: `youtube_music`, `youtube` or `local`). When you are **offline**, or YouTube can't be reached, it plays from your music folder.
- Say where explicitly: `play X on YouTube Music`, `play X on YouTube`, `play X in VLC` / `from my music folder`.
- YouTube Music opens the top result's page. If Chrome blocks autoplay the first time, press play once.
- `python -m jarvis music-check` shows what JARVIS finds in your music folder. Add words to test a song without playing it: `python -m jarvis music-check love me lil wayne`.
- Fix the folder path under `music:` in `config/profile.yaml`. Use single quotes: `'G:\Music'`.

## Name, home, devices and controls (Phase 6.2)

- **Greeting:** the dashboard says "Hi boss, how can I help?" when it opens. Change the name in Settings (or `ui.greeting_name`); turn it off with `ui.greet_on_start: false`.
- **Home address:** type it in Settings. "Find a pharmacy near me", "check police stations around my area" and "directions to Menlyn Park" open Google Maps from your home. The address stays in `profile.yaml`; only the suburb name is looked up (online, once) to move the weather location, never the street.
- **Microphone and speakers:** pick them in Settings. *Detect microphone* listens on every input for a moment and selects the one that heard you; *Test speakers* plays a short tone; *Use these devices* saves and restarts the voice without closing the window.
- **Music controls:** "pause", "resume", "next song", "previous track", "stop the music", "volume up/down", "mute". These press the same media keys as your keyboard (only that fixed list), so they control VLC and YouTube Music.
- **Better hearing of names:** artist and album folder names from your music folder are given to the speech recogniser as hints.
- **Copy diagnostics** (Settings) copies a report to paste when asking for help. It lists settings, devices and recent log lines, and never includes your address.
- **YouTube Music autoplay:** if Chrome opens the song but doesn't start it, allow sound for the site once: Chrome, Settings, Privacy and security, Site settings, Sound, add `music.youtube.com` to "Allowed to play sound".

## Launcher and "which one?" (Phase 6.3)

- **One-click start:** double-click `start-jarvis.bat` (no console window). Right-click it, *Send to > Desktop (create shortcut)* to get a desktop icon, or pin it to the taskbar. If something goes wrong, use `start-jarvis-console.bat`, which keeps the window open so you can read the error.
- **Answering "which one?":** when a song name matches several things, JARVIS lists up to three, numbered. Reply with "the second one", "2", or part of the name. Saying "cancel" or anything unrelated drops the question.

## Voice quality (Phase 6.4)

- **Calm speech works:** quiet recordings are raised automatically before recognition (`voice.stt.auto_gain`), and the "no sound heard" threshold is lower (`silence_peak: 0.004`). Still set the Focusrite gain so your normal voice lights the green ring.
- **Mishearings:** near-misses of your app, folder and contact names are fixed ("power by" -> Power BI, "data bricks" -> Databricks), switch `voice.stt.correct_words`. The model is also told that your words come from speech and may be slightly off, so it acts on the likely meaning.
- **Accuracy vs speed:** `voice.stt.beam_size` is 3 (about half a second slower than 1). Set 1 for fastest, 5 for most careful.
- **Pick a voice:** `python -m jarvis voice-demo` speaks one sentence in several Piper voices (each missing one is a 60-115 MB download, skipped if it fails). Add voice names after the command to try others. Then set `voice.tts.voice` in `config/settings.yaml`. Speaking a touch slower (`length_scale: 1.08`) can also sound less robotic.
- **Models on another drive:** Whisper and Piper already live in `data/models` inside the project. For Ollama, set the user environment variable `OLLAMA_MODELS` (for example `G:\ollama-models`) and restart Ollama.

## Time-aware greeting (Phase 6.5)

On start JARVIS says, for example, "Good morning, boss. All systems are running. How can I help you this morning?" (morning 05-12, afternoon 12-17, evening 17-22, late night after that). "All systems are running" is only said after a quick check that the language model answers; if it doesn't, JARVIS says so instead. Change the name in Settings.

**Reminders in the greeting:** after the status line JARVIS reads today's upcoming reminders, earliest first, at most `ui.briefing_max` (default 2), then says how many more are left ("And 3 more later today."). Reminders already late are announced by the reminder scheduler. There is no priority field yet, so "most important" means "soonest".

## Automatic tests (CI)

`.github/workflows/ci.yml` runs the whole test suite on GitHub (Windows and Linux) on every push to `main` and every pull request. Nothing to set up: push the branch and look at the Actions tab. The tests use fakes, so they need no microphone, Ollama or internet.

## Your drive: folders and movies (Phase 6.8)

Add your main drive once in `config/profile.yaml` (no need to list every folder):

```yaml
drives:
  - 'G:\'
```

- "open the main folder" / "open the G drive" opens the drive.
- "open the movies folder" (or games, study, documentaries...) finds a folder of that name anywhere on the drive (6 levels deep, plural or singular).
- "play the movie Dune", "watch Dune", or "open the movies folder and play Dune" finds the video file and plays it in VLC. A series name ("play Breaking Bad") plays its whole folder. After you open a folder, "play ..." searches inside it first.
- If several things match, JARVIS lists up to three and you answer "the second one" or part of the name.
- Safety: only folders and video files inside the listed drives are touched, nothing is deleted or changed, and a path that tries to escape the drive (`..`) is ignored. Music still goes through the music folder ("play music", "play <song>").
- The first request reads the drive's folder listing (a few seconds on a big drive, done in the background at start-up) and reuses it for 10 minutes.

## Search like Windows, any app, several steps (Phase 6.9)

- **Forgiving music search:** words that are not in your library are matched to the closest spelling that is, so "Heartless by The Weekend" finds The Weeknd, and "01 Love Me" is found by "Love Me". It searches the whole path (artist / album / song), up to 6 levels deep. If something still fails, run `python -m jarvis music-check "the words you said"` to see what it finds.
- **Any installed app, no locations needed:** "open Pod Player" looks the name up in your Start menu shortcuts and Microsoft Store apps if it is not in `profile.yaml`. Apps in the profile still come first. If two match equally, JARVIS asks which one. Only programs listed in the Start menu are ever started, and uninstallers are skipped. (This part needs Windows; please test it on your PC.)
- **Several steps in one sentence:** "open the main folder and then search the music folder for Heartless and play it in VLC", "in the music folder search Love Me by Lil Wayne and play it". The folder names are hints; JARVIS runs the steps in order and stops if one fails. Sentences it cannot split cleanly go to the model as before.

## Series without saying "movie" (Phase 6.11)

- "play Citadel.S01 in main folder" now goes straight to the video search: names with S01 / S01E02 / season 2 / 1080p / .mkv are treated as series or movies, and "in the main folder" (or any folder name) is only a hint about where to look.
- If a "play X" finds nothing in your music folder, JARVIS tries the same words on your drive's videos before giving up, so it no longer needs a second sentence like "it's in Movies & Series".

## Saying movie titles (Phase 6.12)

Safest phrases: **"play the movie The Run"**, **"watch The Run"**, **"play Citadel season 1"**. A bare **"play The Run"** now also checks your drive first: if a film or series on it has exactly that title (year and quality tags like `(2026)` or `1080p` are ignored), it plays that. Otherwise it behaves as before (YouTube Music when online, your music folder when offline). A partial title ("play Run") is never taken for a film.

## Videos from your own drive (music videos, documentaries, concerts)

Say the kind of thing it is and JARVIS searches your drive and plays it in VLC:

- "play the video of Love Me by Lil Wayne", "play the music video Love Me", "play Love Me by Lil Wayne music video"
- "play Lil Wayne videos locally" (several matches: it lists three and you say "the second one")
- "play the documentary Our Planet", "play the live performance Kendrick Lamar", "watch ..."

Say **YouTube** ("play the video of Love Me on YouTube") and it goes to YouTube instead. Words like "locally" or "from my drive" are accepted and ignored.

## Phase 6.14: "Hey JARVIS", moving orb, fuzzy movie titles

- **Wake word.** Say "Hey JARVIS" instead of holding Right Ctrl (the key still works). One-time setup: `python -m jarvis wake-setup` (downloads about 4 MB and turns it on). Test it with `python -m jarvis wake-check`. After it hears you it beeps, records until you stop talking, then answers. Everything runs on your PC; the microphone is only listened to for the wake phrase and nothing is saved. Settings (in `settings.yaml` under `voice:`): `wake_word`, `wake_threshold` (lower if it misses you, higher if it wakes by itself), `wake_silence_seconds`, `wake_chime`. Add these keys by hand if you keep your own settings file.
- **Orb animation.** The core breathes, rings spin faster while thinking, and ripples leave the orb while listening or speaking; the waveform follows the same state.
- **Misheard movie titles.** "Braking Bad" or "Heats" now find the right file; when nothing is close, it still says so.

## Read-only Gmail (Phase 6.15)

Say "check my email", "do I have any new emails", "read my last 5 emails" or "check my emails from Sam". JARVIS reads out the unread count and who wrote with the subject. It only ever reads sender, subject and date: it cannot send, delete or change anything, and it never reads the message text.

One-time setup (about 5 minutes, free, no extra downloads):

1. Go to console.cloud.google.com, create a project, then **APIs & Services > Library**, find **Gmail API** and click Enable.
2. **APIs & Services > OAuth consent screen**: choose External, give it a name, add your own Gmail address under **Test users**.
3. **Credentials > Create credentials > OAuth client ID > Desktop app**. Download the JSON.
4. Save it as `data/gmail/client_secret.json` inside the jarvis folder (the folder is git-ignored).
5. Run `python -m jarvis gmail-setup`. Your browser opens; allow **read-only** access. (Google may warn the app is unverified: that is normal for your own test app, choose Continue.) The command then switches `gmail.enabled` on.

Your sign-in is stored in `data/gmail/token.json` on this PC. To disconnect, delete that file or remove JARVIS at myaccount.google.com/permissions. Add the `gmail:` block from the shipped `settings.yaml` to yours by hand if you want to change the file paths.

## Important reminders (Phase 6.15)

Say "set an important reminder to call the bank tomorrow at 9am" or add "it's important" at the end. At start-up JARVIS reads important reminders first. Existing reminders are untouched.

## Code style

`ruff check src tests` should print "All checks passed" (the rules are in `pyproject.toml`, and CI runs it).

## More natural conversation (Phase 6.15)

JARVIS now talks like a person: contractions, the answer first, a friendly tone that follows your mood, short natural follow-up questions now and then, and it works out "it" or "that" from the last few messages. A reply that hits the length cap now ends at its last complete sentence instead of stopping mid-word. It is still the one small model, so keep `brain.max_history_turns` at 4 or more if you want it to remember more of the chat (a little slower per reply).

## Screen intelligence, level 1 (Phase 6.16)

Say "what's on my screen?", "explain this error in my terminal", "explain this SQL query", "what does this Python function do?", "what is this application showing me?", "summarise the article I'm reading", or "take a screenshot". JARVIS finds the window you are working in (its own window is skipped), takes a picture of just that window, reads the words with Windows' built-in text recognition, and the local model explains them: what an error means, the likely cause and one fix to try, what code or SQL does, or the main point of an article. Say "the whole screen" to read everything.

- Only when you ask. The picture is deleted at once; the text is never saved or logged; nothing leaves your PC. Text on screen is treated as material to explain, never as orders.
- It reads **words only**. For a chart it can read the title and labels but cannot see the shape of the lines; for photos it says it can't read any text. A vision model for pictures is a later extra.
- Use your voice (or place JARVIS beside the window): if the JARVIS dashboard covers the window you mean, the picture will include the dashboard.
- Needs Pillow (`pip install -r requirements.txt`) and a Windows text-recognition language (English is normally installed). If Windows text recognition fails and Tesseract is installed, it is used instead.
- Test it with `python -m jarvis screen-check` (it waits 5 seconds so you can switch to a window, then shows what it read). Turn it off with `screen.enabled: false`. Add the `screen:` block from the shipped `settings.yaml` to yours by hand if you want to change `max_chars` or the screenshot folder.
- Follow-up questions about the same screen need a fresh "explain this ...", because the screen text is not kept.

## Faster and more accurate: measure first (Phase 6.17)

Three tools show where time and mistakes really are, so the next fixes aim at the right thing.

- **`python -m jarvis route-report`** reads `data/turns.jsonl`, a small local record of each turn (what you said, fast path or model, which tools ran, how long each stage took). It prints how many turns were instant, the average time for the instant and the model paths, speech recognition time, and two lists: commands the model had to work out (send me these and I add instant rules for them) and plain conversation. Stays on your PC; turn it off with `logging.record_turns: false`.
- **`python -m jarvis tool-eval`** runs 40 everyday sentences through the model alone and scores whether it picks the right tool, with the time per sentence and a list of its mistakes. To try another model: `ollama pull NAME`, then `python -m jarvis tool-eval NAME`. Keep whichever scores better and is fast enough; it replaces the current model, so you still run only one.
- **Soft tick** the moment you stop talking, so you know it heard you while it thinks (`voice.ack_sound: false` to turn it off).

Not built yet, on purpose: speaking while the model is still writing, and sending only the relevant tools per request. Both could slow things down on your GPU unless measured first, so they wait for your `route-report` and `tool-eval` numbers.

## Phase 6.18: prompt cache check

`python -m jarvis speed-check` now ends with a **prompt-cache test**. A reply is quick when Ollama reuses its stored copy of the long instructions and reads only your new words. The test sends three tiny requests and tells you whether that is happening on your PC. Past turns are also now re-sent exactly as before (including the clock note), so the start of each request stays identical.

## Phase 6.20: "Hey JARVIS" like a car assistant

Say "Hey JARVIS" and it answers **"I'm listening."** out loud, then waits up to 10 seconds for you to speak. Say your request ("please play Kanye West Power") and it answers. If you say nothing for 10 seconds it quietly goes back to sleep; say "Hey JARVIS" again to wake it. Holding Right Ctrl still works as before.

- If JARVIS asks you something ("Sure. What would you like to play?"), it listens for your answer straight away without the wake word, up to three questions in a row.
- Saying just "play" makes it ask what to play; your next words are the title.
- Settings under `voice:` in `settings.yaml` (add by hand if you keep your own file): `wake_reply` (empty = a beep instead), `wake_listen_seconds`, `wake_follow_up`.
- Also in this version: the instructions sent to the model are shorter, and only the tools a sentence could need are sent (`brain.tool_subset: false` turns that off). This makes model replies faster. Run `python -m jarvis tool-eval` to check it still picks the right tool.

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
| `JARVIS_DB` | Path to the memory database (default `data/jarvis.db`) |
| `JARVIS_CONFIG` | Path to another settings file |

Never put secrets in `settings.yaml`; use `.env`, which is git-ignored.

Logs go to the console and to `logs/jarvis.log` (rotating).

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
| JARVIS says it didn't select a tool when you ask it to remember | Say it shorter: "remember that my project is due on Monday". |
| A reminder never spoke | JARVIS must be running at that time. Check `python -m jarvis memory` that it was saved with the right time. |
| JARVIS can't find FL Studio / VLC | Run `python -m jarvis add-app "FL Studio"`, or put the full `.exe` path under `candidates` in `config/profile.yaml`. |
| "I can't find your music folder" | Set `music: folder` in `config/profile.yaml` to the real path (single quotes around Windows paths). |
| Weather says it can't reach the service | Needs internet. Try again; if it persists, check `logs/jarvis.log`. |
| With 16 tools the model picks the wrong one | Say the command shorter and more directly ("search Google for ...", "play GNX in VLC"). Tell me which phrases fail. |
