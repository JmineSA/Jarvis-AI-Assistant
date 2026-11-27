"""A fixed set of everyday sentences with the tool each should trigger, to compare models fairly.

`python -m jarvis tool-eval [model]` runs them through the model (not the instant fast path), so the score is
about the model alone. A tool that your profile does not have is skipped. None means "just talk, no tool".
"""

from __future__ import annotations

CASES: list[tuple[str, str | None]] = [
    ("open Visual Studio Code", "open_app"),
    ("could you launch Chrome for me", "open_app"),
    ("start Power BI", "open_app"),
    ("open the downloads folder", "open_folder"),
    ("show me what is in my project folder", "list_files"),
    ("go to github dot com", "open_website"),
    ("what time is it", "current_time"),
    ("how is my computer doing", "system_status"),
    ("how much memory am I using", "system_status"),
    ("what's the weather like in Pretoria", "weather"),
    ("will it rain tomorrow afternoon", "weather"),
    ("remember that my project is due on Monday", "remember"),
    ("my favourite editor is VS Code, keep that in mind", "remember"),
    ("what do you remember about my project", "recall"),
    ("forget that my project is due on Monday", "forget"),
    ("remind me to call the bank tomorrow at nine", "set_reminder"),
    ("set a reminder to drink water in an hour", "set_reminder"),
    ("what reminders do I have", "list_reminders"),
    ("play some music", "play_media"),
    ("play Heartless by The Weeknd", "play_media"),
    ("put on something by Drake", "play_media"),
    ("pause the music", "media_control"),
    ("skip this song", "media_control"),
    ("turn the volume up", "media_control"),
    ("search Google for the best action movies", "web_search"),
    ("look up how to center a div on YouTube", "web_search"),
    ("find coffee shops near me", "maps"),
    ("give me directions to Menlyn Mall", "maps"),
    ("write an email to Sam about the meeting", "compose_email"),
    ("check my email", "check_email"),
    ("do I have any unread emails", "check_email"),
    ("what's on my screen", "look_at_screen"),
    ("explain this error", "look_at_screen"),
    ("take a screenshot", "take_screenshot"),
    ("hello there", None),
    ("thank you, that was helpful", None),
    ("how are you today", None),
    ("tell me a fun fact about space", None),
    ("what is the capital of France", None),
    ("explain what a neural network is in simple words", None),
    ("what's a good way to stay focused while studying", None),
    ("tell me a joke", None),
]
