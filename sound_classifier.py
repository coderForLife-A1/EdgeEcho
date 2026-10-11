#!/usr/bin/env python3
"""
sound_classifier.py
--------------------------------------------------------------------------
Runs on the Debian/Linux side (Qualcomm QRB2210) of the Arduino UNO Q.

Flow
  1. Block (zero CPU) on a rising-edge GPIO event from the STM32 (libgpiod v2).
  2. Record 3 s from the INMP441 I2S mic through ALSA (`arecord`).
  3. Slide the Edge Impulse model's window (1 s) across the 3 s clip with a
     0.5 s hop -> 5 inferences; keep the best score per label.
  4. If the best non-noise label scores >= 85 %, speak it with espeak-ng,
     piped to the default (Bluetooth) audio sink.

100 % offline: no network, no cloud, no LLM. The .eim is a self-contained
binary produced by Edge Impulse Studio (Linux AARCH64 target).

Usage
  python3 sound_classifier.py                 # normal operation
  python3 sound_classifier.py --once          # bench test: record+classify now
  python3 sound_classifier.py --list-lines    # hint on finding the GPIO line
--------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import timedelta

import numpy as np

# ------------------------------ CONFIGURATION ------------------------------
# Every value can be overridden with an environment variable (handy for
# systemd) without editing the file.
MODEL_PATH      = os.environ.get("EIM_PATH", os.path.expanduser("~/model/model.eim"))

# GPIO wake line. Find yours with `gpiodetect` and `gpioinfo`.
GPIO_CHIP       = os.environ.get("GPIO_CHIP", "/dev/gpiochip0")
GPIO_LINE       = int(os.environ.get("GPIO_LINE", "0"))       # line OFFSET

# Audio capture (ALSA). `arecord -l` shows cards; `plughw:` lets ALSA convert
# format/rate for us. INMP441 is a 24-bit-in-32-bit I2S mic; most I2S
# controllers expose it as stereo S32_LE with the mic on the LEFT slot
# (INMP441 L/R pin tied to GND) -> MIC_CHANNEL = 0. Tied to VDD -> 1.
ALSA_DEVICE     = os.environ.get("ALSA_DEVICE", "plughw:0,0")
ALSA_CHANNELS   = int(os.environ.get("ALSA_CHANNELS", "2"))
MIC_CHANNEL     = int(os.environ.get("MIC_CHANNEL", "0"))
RECORD_SECONDS  = 3
# Digital gain applied after converting 32-bit -> 16-bit. INMP441 output is
# quiet when only the top 16 bits are kept. IMPORTANT: use the SAME gain and
# capture chain when recording your Edge Impulse training data.
DIGITAL_GAIN    = float(os.environ.get("DIGITAL_GAIN", "1.0"))

# Inference
CONF_THRESHOLD  = 0.85
HOP_SECONDS     = 0.5                       # sliding-window hop over the clip
IGNORE_LABELS   = {"noise", "unknown", "background", "silence"}

# Speech (espeak-ng -> WAV on stdout -> PLAYER_CMD). With PulseAudio/PipeWire
# the Bluetooth speaker only needs to be the DEFAULT SINK. For raw BlueALSA use
# e.g. PLAYER_CMD="aplay -q -D bluealsa".
VOICE           = os.environ.get("ESPEAK_VOICE", "en-us")
SPEED_WPM       = int(os.environ.get("ESPEAK_SPEED", "150"))
PLAYER_CMD      = shlex.split(os.environ.get("PLAYER_CMD", "paplay"))
# Optional friendlier phrases per label; falls back to "<label> detected".
LABEL_PHRASES = {
    # "doorbell": "Someone is at the door",
    # "glass_break": "Warning. Glass breaking",
}

log = logging.getLogger("soundcls")
_stop = False


# --------------------------------- AUDIO -----------------------------------
def record_clip(seconds: int, fs: int) -> np.ndarray:
    """Record `seconds` of audio, return mono int16 samples at `fs` Hz."""
    cmd = [
        "arecord", "-q", "-D", ALSA_DEVICE,
        "-f", "S32_LE", "-r", str(fs), "-c", str(ALSA_CHANNELS),
        "-t", "raw", "-d", str(seconds),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=seconds + 5)
    if proc.returncode != 0:
        raise RuntimeError(f"arecord failed: {proc.stderr.decode(errors='replace').strip()}")

    raw = np.frombuffer(proc.stdout, dtype="<i4")
    usable = (raw.size // ALSA_CHANNELS) * ALSA_CHANNELS
    mono = raw[:usable].reshape(-1, ALSA_CHANNELS)[:, MIC_CHANNEL]

    # INMP441 data is MSB-aligned in the 32-bit slot -> keep the top 16 bits.
    pcm = (mono >> 16).astype(np.float32) * DIGITAL_GAIN
    pcm = np.clip(pcm, -32768, 32767).astype(np.int16)

    expected = seconds * fs
    if pcm.size < expected:                          # pad if arecord came up short
        pcm = np.pad(pcm, (0, expected - pcm.size))
    return pcm[:expected]


# ------------------------------- INFERENCE ---------------------------------
def classify_clip(runner, pcm: np.ndarray, params: dict) -> dict[str, float]:
    """
    The model expects exactly `input_features_count` samples (e.g. 16000 for a
    1 s window at 16 kHz) but we record 3 s, so slide the window over the clip.
    Returns the max score seen for each label (best for short transient events;
    averaging would dilute a 1-second sound inside a 3-second clip).
    """
    n = int(params["input_features_count"])
    fs = int(params["frequency"])
    hop = max(1, int(HOP_SECONDS * fs))

    if pcm.size < n:
        pcm = np.pad(pcm, (0, n - pcm.size))

    best: dict[str, float] = {}
    for start in range(0, pcm.size - n + 1, hop):
        window = pcm[start:start + n].tolist()       # raw int16 values
        result = runner.classify(window)
        for label, score in result["result"]["classification"].items():
            best[label] = max(best.get(label, 0.0), float(score))
    return best


# --------------------------------- SPEECH ----------------------------------
def speak(text: str) -> None:
    """Offline TTS: espeak-ng renders a WAV to stdout, PLAYER_CMD plays it."""
    log.info("Speaking: %s", text)
    tts = subprocess.Popen(
        ["espeak-ng", "-v", VOICE, "-s", str(SPEED_WPM), "--stdout", text],
        stdout=subprocess.PIPE,
    )
    player = subprocess.Popen(PLAYER_CMD, stdin=tts.stdout)
    tts.stdout.close()                               # let espeak get SIGPIPE if player dies
    try:
        player.wait(timeout=30)
        tts.wait(timeout=5)
    except subprocess.TimeoutExpired:
        player.kill()
        tts.kill()
        log.error("Audio playback timed out (is the Bluetooth sink connected?)")


# --------------------------------- PIPELINE --------------------------------
def handle_event(runner, params: dict) -> None:
    t0 = time.monotonic()
    fs = int(params["frequency"])
    pcm = record_clip(RECORD_SECONDS, fs)
    scores = classify_clip(runner, pcm, params)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    log.info("Scores: %s  (%.2fs total)",
             ", ".join(f"{k}={v:.2f}" for k, v in ranked), time.monotonic() - t0)

    candidates = [(k, v) for k, v in ranked if k.lower() not in IGNORE_LABELS]
    if not candidates:
        return
    label, score = candidates[0]
    if score >= CONF_THRESHOLD:
        speak(LABEL_PHRASES.get(label, f"{label.replace('_', ' ')} detected"))
    else:
        log.info("Best '%s' %.2f < %.2f -> staying silent", label, score, CONF_THRESHOLD)


def open_gpio():
    """Request the wake line with rising-edge detection (libgpiod v2 API)."""
    import gpiod
    from gpiod.line import Bias, Direction, Edge

    return gpiod.request_lines(
        GPIO_CHIP,
        consumer="sound-wake",
        config={
            GPIO_LINE: gpiod.LineSettings(
                direction=Direction.INPUT,
                edge_detection=Edge.RISING,
                bias=Bias.PULL_DOWN,                           # no floating input
                debounce_period=timedelta(milliseconds=5),
            )
        },
    )


def _on_signal(signum, _frame):
    global _stop
    _stop = True
    log.info("Signal %s received, shutting down...", signum)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="skip GPIO: record + classify once and exit")
    ap.add_argument("--list-lines", action="store_true", help="print how to discover the GPIO line")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.list_lines:
        print("Run:  gpiodetect   then   gpioinfo <chip>\n"
              "Then set GPIO_CHIP=/dev/gpiochipN and GPIO_LINE=<offset>.")
        return 0

    if not os.path.isfile(MODEL_PATH):
        log.error("Model not found: %s", MODEL_PATH)
        return 2
    os.chmod(MODEL_PATH, 0o755)                      # .eim must be executable

    # Imported here so --list-lines works even if the SDK is not installed yet.
    from edge_impulse_linux.audio import AudioImpulseRunner

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    runner = AudioImpulseRunner(MODEL_PATH)
    req = None
    try:
        # Load the model ONCE; the .eim process stays resident, so each trigger
        # only pays inference time, not model-load time.
        info = runner.init()
        params = info["model_parameters"]
        log.info("Model loaded: %d Hz, %d features/window, labels=%s",
                 params["frequency"], params["input_features_count"], params["labels"])

        if args.once:
            handle_event(runner, params)
            return 0

        req = open_gpio()
        log.info("Armed. Waiting for rising edge on %s line %d ...", GPIO_CHIP, GPIO_LINE)

        while not _stop:
            # Blocks in poll(): ~0 % CPU. The 5 s timeout only lets us notice
            # a shutdown signal.
            if not req.wait_edge_events(timeout=5.0):
                continue
            req.read_edge_events()
            log.info("Wake signal from STM32")
            try:
                handle_event(runner, params)
            except Exception:                          # keep the wearable alive
                log.exception("Event handling failed")
            # Discard edges that arrived while we were busy (echo of our own
            # speech, a long sound, etc.).
            while req.wait_edge_events(timeout=0):
                req.read_edge_events()
        return 0
    finally:
        if req is not None:
            req.release()
        runner.stop()


if __name__ == "__main__":
    sys.exit(main())
