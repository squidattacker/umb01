# --- 必要ライブラリ ---
# !pip install pydub numpy soundfile

import numpy as np
import random
from pydub import AudioSegment
# from IPython.display import Audio, display
# from google.colab import drive

# drive.mount('/content/drive')

# ===== Utility =====
def bpm_to_step(bpm, steps_per_beat=4):
    return 60000 / bpm / steps_per_beat

def note_to_freq(note): 
    return 440 * (2 ** ((note-69)/12))

def safe_stereo(left, right):
    max_len = max(len(left), len(right))
    left = left + AudioSegment.silent(duration=max_len - len(left))
    right = right + AudioSegment.silent(duration=max_len - len(right))
    if left.channels == 1 and right.channels == 1:
        return AudioSegment.from_mono_audiosegments(left, right)
    else:
        return left.overlay(right)

# ===== Kick =====
def make_heartbeat_realistic(bpm=60, base_freq1=60, base_freq2=120):
    beat_ms = 60000 / bpm
    s1_len, s2_len, gap = 0.12, 0.08, 0.20
    t1 = np.linspace(0, s1_len, int(44100*s1_len), endpoint=False)
    s1 = np.sin(2*np.pi*base_freq1*t1) * np.exp(-30*t1)
    t2 = np.linspace(0, s2_len, int(44100*s2_len), endpoint=False)
    s2 = np.sin(2*np.pi*base_freq2*t2) * np.exp(-40*t2)
    silence = np.zeros(int(44100*gap))
    signal = np.concatenate([s1, silence, s2])
    total_len = int(44100*(beat_ms/1000))
    if len(signal) < total_len:
        signal = np.concatenate([signal, np.zeros(total_len - len(signal))])
    signal += 0.005*np.random.randn(len(signal))
    signal = np.int16(signal/np.max(np.abs(signal))*32767)
    return AudioSegment(signal.tobytes(), frame_rate=44100, sample_width=2, channels=1).low_pass_filter(200)

# ===== Hat =====
def make_spark_hat(duration=80):
    t = np.linspace(0, duration/1000, int(44100*duration/1000), endpoint=False)
    noise = np.random.randn(len(t))
    cutoff = random.randint(5000, 12000)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(len(noise), 1/44100)
    spectrum[freqs > cutoff] *= np.exp(-(freqs[freqs > cutoff]-cutoff)/2000)
    filtered = np.fft.irfft(spectrum)
    env = np.concatenate([np.linspace(0, 1, int(0.005*44100)), np.exp(-50*t[int(0.005*44100):])])
    signal = filtered * env
    signal = np.int16(signal/np.max(np.abs(signal))*32767)
    hat = AudioSegment(signal.tobytes(), frame_rate=44100, sample_width=2, channels=1)
    return safe_stereo(hat, hat) - 12

# ===== Snare =====
def make_snare(duration=180):
    t = np.linspace(0, duration/1000, int(44100*duration/1000), endpoint=False)
    noise = np.random.randn(len(t))
    tone = np.sin(2*np.pi*120*t) * np.exp(-10*t)
    signal = (noise*0.8 + tone*0.2) * np.exp(-15*t)
    signal = np.int16(signal/np.max(np.abs(signal))*32767)
    snare = AudioSegment(signal.tobytes(), frame_rate=44100, sample_width=2, channels=1)
    return snare.high_pass_filter(100).low_pass_filter(4000)

# ===== Deep Tape Echo (Snare) =====
def very_deep_tape_echo(audio, delay_ms=1200, repeats=30, decay=1.285, wow=0.05):
    echoed = AudioSegment.silent(duration=len(audio) + delay_ms * repeats + 5000, frame_rate=audio.frame_rate)
    for i in range(repeats):
        mod_delay = int(delay_ms * (1 + random.uniform(-wow, wow)))
        repeat = audio - int(2 * i * (1-decay) * 8)
        cutoff = 5000 - i * 400
        if cutoff > 600:
            repeat = repeat.low_pass_filter(cutoff)
        echoed = echoed.overlay(
            safe_stereo(repeat, AudioSegment.silent(duration=mod_delay)+repeat),
            position=mod_delay*(i+1)
        )
    return audio.overlay(echoed - 4)

# ===== Lead用 Deep Echo =====
def lead_floor_echo(audio, delay_ms=600, repeats=28, decay=0.985):
    echoed = AudioSegment.silent(duration=len(audio) + delay_ms*repeats + 6000, frame_rate=audio.frame_rate)
    for i in range(repeats):
        repeat = audio - int(4*i*(1-decay)*10)
        if i % 3 == 0:
            repeat = repeat.low_pass_filter(2000 + i*80)
        elif i % 3 == 1:
            repeat = repeat.high_pass_filter(600)
        else:
            repeat = repeat.low_pass_filter(7000)
        echoed = echoed.overlay(repeat, position=delay_ms*(i+1))
    return audio.overlay(echoed - 6)

# ===== 揺らぐリード（フロア感つき） =====
def make_glassy_lead(notes=[60,64], duration=60000, lfo_depth=0.002, lfo_rate=0.05):
    t = np.linspace(0, duration/1000, int(44100*duration/1000), endpoint=False)
    signal = np.zeros(len(t))
    env = np.exp(-0.2*t)
    for i, n in enumerate(notes):
        f = note_to_freq(n)
        f_lfo = f * (1 + lfo_depth*np.sin(2*np.pi*lfo_rate*t + i))
        tone = np.sin(2*np.pi*f_lfo*t) + 0.2*np.sin(2*np.pi*2*f_lfo*t)
        signal += tone * env
    signal = np.int16(signal/np.max(np.abs(signal))*32767)
    base = AudioSegment(signal.tobytes(), frame_rate=44100, sample_width=2, channels=1)
    base = base.low_pass_filter(3800) - 12
    layered = lead_floor_echo(base)
    return safe_stereo(layered, layered)

# ===== Scale =====
def generate_floating_scale(base=60, size=5, mode="golden"):
    return [base + int(i*1.618*7) % 24 for i in range(size)]

# ===== Main =====
def make_ultramarine_spark(bpm=120, bars=96, num_notes=5, scale_mode="golden"):
    step_ms = bpm_to_step(bpm)
    total_steps = 16 * bars
    scale = generate_floating_scale(base=60, size=num_notes, mode=scale_mode)
    track = AudioSegment.silent(duration=total_steps*step_ms + 100)

    # リード持続時間
    lead1_duration = (64-32)*16*step_ms   # 32〜64小節
    lead2_duration = (bars*16 - 64*16)*step_ms  # 64〜96小節

    for i in range(total_steps):
        # Kick
        if i % 4 == 0:
            track = track.overlay(make_heartbeat_realistic(bpm=bpm), position=int(i*step_ms))

        # Hat
        if i % 2 == 0 and random.random() < 0.4:
            track = track.overlay(make_spark_hat(duration=int(step_ms/2)), position=int(i*step_ms))

        # スネア：20〜32小節
        if 20*16 <= i < 32*16 and i % 16 in [4, 12]:
            snare = make_snare(duration=int(step_ms*2)) - 2
            if i % 16 == 4 and random.random() < 0.4:
                snare = very_deep_tape_echo(snare)
            snare = snare.high_pass_filter(150).low_pass_filter(3500)
            track = track.overlay(snare, position=int(i*step_ms))

        # リード1回目（32〜64小節）
        if i == 32*16:
            note1 = random.choice(scale)
            note2 = note1 + random.choice([3,7])
            lead = make_glassy_lead(notes=[note1, note2], duration=lead1_duration)
            track = track.overlay(lead, position=int(i*step_ms))

        # リード2回目（64〜96小節）
        if i == 64*16:
            note1 = random.choice(scale)
            note2 = note1 + random.choice([5,9])
            lead = make_glassy_lead(notes=[note1, note2], duration=lead2_duration)
            track = track.overlay(lead, position=int(i*step_ms))

        # スネア復活（92〜96小節）
        if 92*16 <= i < 96*16 and i % 16 in [4, 12]:
            snare = make_snare(duration=int(step_ms*2)) - 2
            snare = very_deep_tape_echo(snare)  # 深く確実に
            snare = snare.high_pass_filter(150).low_pass_filter(3500)
            track = track.overlay(snare, position=int(i*step_ms))

    return track

# ===== 実行例 =====
if __name__ == "__main__":
    output_file = "./ultramarine_final.wav"
    final_track = make_ultramarine_spark(bpm=120, bars=96, num_notes=5, scale_mode="golden")
    final_track.export(output_file, format="wav")

    # display(Audio(output_file))
    print("WAV出力完了:", output_file)