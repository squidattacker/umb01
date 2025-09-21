# umb01

This repository now includes a small command-line tool that synthesises a cosmic
percussion loop inspired by the "ultramarine" concept.

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Render a one-minute loop at 130 BPM (default settings):

```bash
python make_ultramarine_cosmic.py output.wav
```

Adjust the tempo, duration, or the number of bars before the hi-hat pattern fades
in via optional flags:

```bash
python make_ultramarine_cosmic.py output.wav --bpm 128 --duration 90 --bars-intro-hat 4
```
