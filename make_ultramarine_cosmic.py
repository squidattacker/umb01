"""Generate an ultramarine cosmic drum track.

This module synthesizes a simple electronic rhythm track that combines a floor-style
kick drum, a heartbeat-inspired bass layer, and a shimmering stereo hi-hat pattern.
It is written as a standalone script that can be executed from the command line to
render a WAV file.  The defaults reproduce the behaviour described in the original
notebook snippet.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from pydub import AudioSegment


FS = 44_100


# ===== Utility =====
def bpm_to_step(bpm: float, steps_per_beat: int = 4) -> float:
    """Return the duration of a sequencer step in milliseconds."""

    return 60_000 / bpm / steps_per_beat


# ===== Kick（1打目：フロア寄り） =====
def make_floor_kick(bpm: int = 130, f_start: float = 90, f_end: float = 55) -> AudioSegment:
    """Generate a single kick drum hit."""

    s_len = 0.1
    t = np.linspace(0, s_len, int(FS * s_len), endpoint=False)
    f_t = f_start - (f_start - f_end) * (t / s_len)
    tone = np.sin(2 * np.pi * f_t * t) * np.exp(-35 * t)
    sub = np.sin(2 * np.pi * 40 * t) * np.exp(-3 * t) * 0.8
    click_len = max(1, int(FS * 0.004))
    t_click = np.linspace(0, click_len / FS, click_len, endpoint=False)
    click = np.sin(2 * np.pi * 2000 * t_click) * np.exp(-2500 * t_click)
    click_pad = np.zeros(len(t))
    click_pad[:click_len] = click[: min(len(t), click_len)] * 0.4
    sig = 0.6 * tone + sub + click_pad
    sig += 0.002 * np.random.randn(len(sig))
    sig = sig / np.max(np.abs(sig)) * 0.95
    sig_i16 = np.int16(sig * 32767)
    kick = AudioSegment(sig_i16.tobytes(), frame_rate=FS, sample_width=2, channels=1)
    return kick.high_pass_filter(30).low_pass_filter(8000)


# ===== Bass（2打目：心音寄りベース） =====
def make_bass_heartbeat(freq: float = 55, length: float = 0.15) -> AudioSegment:
    """Generate a short bass tone resembling a heartbeat."""

    t = np.linspace(0, length, int(FS * length), endpoint=False)
    sine = np.sin(2 * np.pi * freq * t)
    square = np.sign(np.sin(2 * np.pi * freq * t))
    raw = 0.7 * sine + 0.3 * square
    env = np.exp(-8 * t)
    sig = raw * env
    sig += 0.002 * np.random.randn(len(sig))
    sig = sig / np.max(np.abs(sig)) * 0.9
    sig_i16 = np.int16(sig * 32767)
    bass = AudioSegment(sig_i16.tobytes(), frame_rate=FS, sample_width=2, channels=1)
    return bass.high_pass_filter(35).low_pass_filter(300)


# ===== Hat（光の粒子・ステレオ・奥行き感） =====
def make_particle_hat_stereo(length: float = 0.25) -> AudioSegment:
    """Generate a stereo hi-hat sound with a short delay for width."""

    t = np.linspace(0, length, int(FS * length), endpoint=False)
    noise = np.random.randn(len(t)) * 0.5
    sine = 0.3 * np.sin(2 * np.pi * 9000 * t)
    sig = (noise + sine) * np.exp(-15 * t)
    sig = sig / np.max(np.abs(sig))
    delay_samples = int(0.01 * FS)  # 10 ms stereo delay
    left = sig
    right = np.concatenate([np.zeros(delay_samples), sig[:-delay_samples]])
    right *= 0.85
    stereo = np.vstack([left, right]).T
    sig_i16 = np.int16(stereo * 32767)
    hat = AudioSegment(sig_i16.tobytes(), frame_rate=FS, sample_width=2, channels=2)
    return hat.high_pass_filter(6000).low_pass_filter(12_000) - 12


# ===== 1/fゆらぎでハイハット配置を決定 =====
def generate_fluctuating_pattern(total_steps: int, base_prob: float = 0.7) -> list[bool]:
    """Return a list of booleans describing hat placement using pink noise."""

    pink = np.cumsum(np.random.randn(total_steps))
    pink = (pink - pink.min()) / (pink.max() - pink.min())
    pattern: list[bool] = []
    for i in range(total_steps):
        prob = base_prob * (0.5 + pink[i])
        pattern.append(random.random() < prob)
    return pattern


# ===== Master（EQ+コンプ） =====
def master_effect(track: AudioSegment) -> AudioSegment:
    """Apply subtle EQ and compression to the final track."""

    track = track.high_pass_filter(25)
    bass_boost = track.low_pass_filter(80).apply_gain(+5)
    track = track.overlay(bass_boost)
    track = track.low_pass_filter(6000)
    return track.compress_dynamic_range(threshold=-18.0, ratio=1.8, attack=25, release=120)


# ===== メイン生成（Kick + Bass + Cosmic Hat） =====
def make_ultramarine_cosmic(
    bpm: int = 130,
    duration_sec: int = 60,
    bars_intro_hat: int = 8,
) -> AudioSegment:
    """Assemble the full track.

    Parameters
    ----------
    bpm:
        Tempo used to compute the step duration.
    duration_sec:
        Desired length of the output track in seconds.
    bars_intro_hat:
        Number of bars to wait before adding the hi-hat texture.
    """

    step_ms = bpm_to_step(bpm)
    steps_per_bar = 16
    total_steps = int((duration_sec * 1000) / step_ms)
    track = AudioSegment.silent(duration=int(total_steps * step_ms + 100))

    hat_pattern = generate_fluctuating_pattern(total_steps)

    for i in range(total_steps):
        if i % 4 == 0:
            kick = make_floor_kick(bpm=bpm)
            pos = int(i * step_ms + random.uniform(-3, 3))
            track = track.overlay(kick, position=pos)
            gap_ms = 180
            bass = make_bass_heartbeat(freq=55)
            track = track.overlay(bass, position=int(pos + gap_ms + random.uniform(-2, 2)))

        bar_idx = i // steps_per_bar
        if bar_idx >= bars_intro_hat and hat_pattern[i]:
            hat = make_particle_hat_stereo(length=0.25)
            hat = hat + random.uniform(-6, 0)
            pos = int(i * step_ms + random.uniform(-20, 20))
            track = track.overlay(hat, position=pos)

    return track


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an ultramarine cosmic drum track.")
    parser.add_argument("output", type=Path, help="Destination WAV file path.")
    parser.add_argument("--bpm", type=int, default=130, help="Tempo in beats per minute.")
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration of the track in seconds (default: 60).",
    )
    parser.add_argument(
        "--bars-intro-hat",
        type=int,
        default=8,
        help="Number of bars before the hat pattern fades in (default: 8).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    track = make_ultramarine_cosmic(
        bpm=args.bpm,
        duration_sec=args.duration,
        bars_intro_hat=args.bars_intro_hat,
    )
    mastered = master_effect(track)
    mastered.export(args.output, format="wav")


if __name__ == "__main__":
    main()
