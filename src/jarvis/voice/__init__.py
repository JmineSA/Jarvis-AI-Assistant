"""Voice: microphone capture, push-to-talk, speech-to-text, text-to-speech, playback.

Heavy libraries (sounddevice, faster-whisper, piper, pynput) are imported only when
used, so the rest of JARVIS starts fast and the tests need no audio hardware.
"""
