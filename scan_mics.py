"""Find which microphone input actually hears you.

Run:  python scan_mics.py
Then keep talking (or clapping) the whole time. It tries every input device,
every channel, about 2 seconds each, and shows how loud each one was.
"""

import sys

import numpy as np
import sounddevice as sd

SECONDS = 2
SKIP_APIS = ("WDM-KS",)  # exclusive-mode driver: can hang, not needed

apis = sd.query_hostapis()
devices = sd.query_devices()

print("Keep talking or clapping until the scan finishes.\n")
results = []
for index, dev in enumerate(devices):
    channels = dev["max_input_channels"]
    api = apis[dev["hostapi"]]["name"]
    if channels < 1 or any(skip in api for skip in SKIP_APIS):
        continue
    rate = int(dev["default_samplerate"])
    label = f"[{index}] {dev['name']} ({api})"
    try:
        audio = sd.rec(int(SECONDS * rate), samplerate=rate, channels=channels,
                       dtype="float32", device=index)
        sd.wait()
    except Exception as exc:  # noqa: BLE001
        print(f"{label}: cannot open ({exc})")
        continue
    peaks = np.abs(audio).max(axis=0)
    text = ", ".join(f"ch{i + 1}={p:.3f}" for i, p in enumerate(peaks))
    heard = "  <-- HEARS SOUND" if peaks.max() > 0.01 else ""
    print(f"{label}: {text}{heard}")
    if peaks.max() > 0.01:
        results.append((index, dev["name"], api, [i + 1 for i, p in enumerate(peaks) if p > 0.01]))

print()
if results:
    print("Devices that heard sound:")
    for index, name, api, chans in results:
        print(f"  [{index}] {name} ({api}), channel(s) {chans}")
else:
    print("Nothing heard on any device. Check: mic plugged in, 48V/gain on the Focusrite,")
    print("Windows Settings > Privacy > Microphone (allow desktop apps), and the mute switch.")
sys.exit(0)
