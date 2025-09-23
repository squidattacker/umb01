# --- 必要ライブラリ ---
# 以下コードをそのままGooGle Colab へコピペ
!pip install -q pydub numpy soundfile

import numpy as np, random, os
from pydub import AudioSegment
from IPython.display import Audio, display
from google.colab import drive

drive.mount('/content/drive')
FS = 44100

# ===== Utility =====
def bpm_to_step(bpm, steps_per_beat=4):
    return 60000 / bpm / steps_per_beat  # 16分音符(ms)

def note_to_freq(note):
    return 440 * (2 ** ((note-69)/12))

def safe_stereo(left, right):
    if left.channels != 1: left = left.set_channels(1)
    if right.channels != 1: right = right.set_channels(1)
    max_len = max(len(left), len(right))
    left  = left  + AudioSegment.silent(duration=max_len-len(left), frame_rate=FS)
    right = right + AudioSegment.silent(duration=max_len-len(right), frame_rate=FS)
    return AudioSegment.from_mono_audiosegments(left, right)

def stereo_silence(ms):
    return AudioSegment.silent(duration=int(ms), frame_rate=FS).set_channels(2)

# ===== Kick =====
def make_heartbeat_realistic(bpm=60, base_freq1=50, base_freq2=100):
    beat_ms = 60000 / bpm
    t1 = np.linspace(0,0.14,int(FS*0.14),endpoint=False)
    t2 = np.linspace(0,0.10,int(FS*0.10),endpoint=False)
    s1 = np.sin(2*np.pi*base_freq1*t1)*np.exp(-22*t1)
    s2 = np.sin(2*np.pi*base_freq2*t2)*np.exp(-30*t2)
    sig = np.concatenate([s1, np.zeros(int(FS*0.18)), s2])
    need = int(FS*(beat_ms/1000)) - len(sig)
    if need > 0: sig = np.concatenate([sig, np.zeros(need)])
    sig += 0.003*np.random.randn(len(sig))
    sig = np.int16(sig/np.max(np.abs(sig))*32767)
    kick = AudioSegment(sig.tobytes(), frame_rate=FS, sample_width=2, channels=1)
    kick = kick.low_pass_filter(1200).high_pass_filter(40)
    return kick.set_channels(2)

# ===== Hat =====
def make_spark_hat(duration=80):
    t=np.linspace(0,duration/1000,int(FS*duration/1000),endpoint=False)
    noise=np.random.randn(len(t))
    cutoff=random.randint(5000,12000)
    spec=np.fft.rfft(noise); freqs=np.fft.rfftfreq(len(noise),1/FS)
    spec[freqs>cutoff]*=np.exp(-(freqs[freqs>cutoff]-cutoff)/2000)
    filtered=np.fft.irfft(spec)
    env=np.concatenate([np.linspace(0,1,int(0.005*FS)), np.exp(-50*t[int(0.005*FS):])])
    sig=np.int16((filtered*env)/np.max(np.abs(filtered*env))*32767)
    hat=AudioSegment(sig.tobytes(), frame_rate=FS, sample_width=2, channels=1)
    pan=random.uniform(-1,1)
    L = hat if pan<0 else (hat-int(abs(pan)*12))
    R = (hat-int(abs(pan)*12)) if pan<0 else hat
    return safe_stereo(L,R) - 12

# ===== Dub Snare =====
def make_dub_snare_erotic_fb_final(bpm=120, attack_ms=25, tail_sec=12.0,
                                   feedback_gain=0.75, repeats=10, delay_ms=380,
                                   wet_gain_db=-4, dry_gain_db=-24):
    # Seed
    t = np.linspace(0, attack_ms/1000, int(FS*(attack_ms/1000)), endpoint=False)
    noise = np.random.randn(len(t)) * np.exp(-80*t)
    tone  = 0.5*np.sin(2*np.pi*220*t) * np.exp(-60*t)
    seed = (noise*0.7 + tone*0.3).astype(np.float32)
    # Attack soften
    fade_len = int(0.003*FS)
    fade_env = np.ones_like(seed)
    fade_env[:fade_len] = np.linspace(0,1,fade_len)
    fade_env[-fade_len:] = np.linspace(1,0,fade_len)
    seed *= fade_env

    # IR
    N = int(FS * tail_sec)
    t_tail = np.linspace(0, tail_sec, N, endpoint=False)
    env = np.exp(-1.8 * t_tail)
    ir_noise = np.random.randn(N) * env
    ir_tone  = 0.15*np.sin(2*np.pi*100*t_tail) * np.exp(-0.4*t_tail)
    ir = ir_noise + ir_tone
    out = np.zeros_like(ir); a_prev=0
    cutoff_start, cutoff_end = 10000, 300
    for i in range(N):
        cutoff = cutoff_start + (cutoff_end-cutoff_start)*(i/(N-1))
        rc = 1.0/(2*np.pi*cutoff+1e-9)
        alpha = (1.0/FS)/(rc+(1.0/FS))
        a_prev = a_prev + alpha*(ir[i]-a_prev)
        out[i] = a_prev
    out /= np.max(np.abs(out))+1e-9
    irL, irR = out, np.roll(out,int(0.007*FS))*0.8

    # Convolution
    n = len(seed)+len(irL)-1
    Nfft = 1 << (n-1).bit_length()
    S = np.fft.rfft(seed,Nfft)
    IL, IR = np.fft.rfft(irL,Nfft), np.fft.rfft(irR,Nfft)
    wetL = np.fft.irfft(S*IL,Nfft)[:n]
    wetR = np.fft.irfft(S*IR,Nfft)[:n]

    # Feedback
    delay_samp = int(FS*delay_ms/1000)
    n_fb = len(wetL)+delay_samp*repeats
    outL, outR = np.zeros(n_fb), np.zeros(n_fb)
    outL[:len(wetL)] += wetL
    outR[:len(wetR)] += wetR
    for i in range(1,repeats+1):
        start = delay_samp*i
        if start+len(wetL) <= n_fb:
            g = feedback_gain**i
            outL[start:start+len(wetL)] += wetL*g*(1-0.05*i)
            outR[start:start+len(wetR)] += wetR*g*(1-0.03*i)

    # Gain
    outL *= 10**(wet_gain_db/20)
    outR *= 10**(wet_gain_db/20)
    wet = np.vstack([outL,outR]).T
    peak = np.max(np.abs(wet)); limit = 10**(-6/20)
    if peak > limit: wet *= (limit/peak)

    sig_i16 = np.int16(wet*32767)
    stereo = AudioSegment(sig_i16.tobytes(), frame_rate=FS, sample_width=2, channels=2)
    return stereo + dry_gain_db

# ===== Leads =====
def lead_floor_echo(audio, delay_ms=800, repeats=24, decay=0.975):
    echoed = AudioSegment.silent(duration=len(audio)+delay_ms*repeats+6000,
                                 frame_rate=audio.frame_rate).set_channels(1)
    for i in range(repeats):
        rep = audio - int(4*i*(1-decay)*12)
        rep = rep.low_pass_filter(2500+i*60) if i%2==0 else rep.high_pass_filter(400)
        echoed = echoed.overlay(rep, position=delay_ms*(i+1))
    return safe_stereo(audio,audio).overlay(safe_stereo(echoed,echoed)-8)

def make_glassy_lead(notes=[60,64], duration=60000):
    t=np.linspace(0,duration/1000,int(FS*duration/1000),endpoint=False)
    sig=np.zeros(len(t)); env=np.exp(-0.2*t)
    for i,n in enumerate(notes):
        f=note_to_freq(n); f_lfo=f*(1+0.002*np.sin(2*np.pi*0.05*t+i))
        tone=np.sin(2*np.pi*f_lfo*t)+0.2*np.sin(2*np.pi*2*f_lfo*t)
        sig+=tone*env
    sig=np.int16(sig/np.max(np.abs(sig))*32767)
    base=AudioSegment(sig.tobytes(),frame_rate=FS,sample_width=2,channels=1).low_pass_filter(3800)-12
    return lead_floor_echo(base)

# ===== Add-on Lead =====
def generate_floating_scale(base=60,size=5,mode="golden"):
    return [base + int(i*1.618*7) % 24 for i in range(size)]

def make_deep_fibonacci_chord(duration=4000, base_freq=220):
    t=np.linspace(0,duration/1000,int(FS*duration/1000),endpoint=False)
    fib=[1,2,3,5,8,13]; ratios=[fib[i]/fib[i+1] for i in range(len(fib)-1)]
    freqs=[base_freq*r*random.choice([1,2]) for r in ratios[:3]]
    sig=np.zeros(len(t)); env=np.exp(-0.15*t)
    for f in freqs: sig+=np.sin(2*np.pi*f*t)*env
    sig=sig/np.max(np.abs(sig))*0.18
    sig_i16=np.int16(sig*32767)
    chord=AudioSegment(sig_i16.tobytes(),frame_rate=FS,sample_width=2,channels=1).low_pass_filter(1800)-6
    return lead_floor_echo(chord)

def add_adon_lead(track, bpm=120, bars=96, step_ms=500):
    beat_ms=60000/bpm
    chord_dur=int(4*beat_ms)
    forbidden=[(20,32),(92,96)]
    for bar in range(4,bars):
        if any(s<=bar<e for s,e in forbidden): continue
        chord=make_deep_fibonacci_chord(chord_dur)
        if bar==4: chord=chord.fade_in(2000)
        if bar==bars-1: chord=chord.fade_out(2000)
        track=track.overlay(chord,position=int(bar*16*step_ms))
    return track

# ===== Track =====
def make_ultramarine_spark_final(bpm=120, bars=96, num_notes=5):
    step_ms=bpm_to_step(bpm); total_steps=16*bars
    scale=generate_floating_scale(60,num_notes)
    track=stereo_silence(total_steps*step_ms+100)
    lead1_duration = int((64-32)*16*step_ms)
    lead2_duration = int((bars*16 - 64*16)*step_ms)
    snare_events = [bar*16+4 for bar in list(range(20,32)) + list(range(92,96))]
    for i in range(total_steps):
        if i%4==0:
            track=track.overlay(make_heartbeat_realistic(bpm),position=int(i*step_ms))
        if i%2==0 and random.random()<0.4:
            track=track.overlay(make_spark_hat(duration=int(step_ms/2)),position=int(i*step_ms))
        if i in snare_events:
            dub=make_dub_snare_erotic_fb_final(bpm=bpm)
            track=track.overlay(dub,position=int(i*step_ms))
        if i==32*16:
            n1=random.choice(scale); n2=n1+random.choice([3,7])
            track=track.overlay(make_glassy_lead([n1,n2],duration=lead1_duration),position=int(i*step_ms))
        if i==64*16:
            n1=random.choice(scale); n2=n1+random.choice([5,9])
            track=track.overlay(make_glassy_lead([n1,n2],duration=lead2_duration),position=int(i*step_ms))
    return track

# ===== 実行 =====
bpm=120; bars=96; step=bpm_to_step(bpm)
output_file = "/content/drive/MyDrive/umb/ultramarine_FINAL_dub_snare.wav"
base_track = make_ultramarine_spark_final(bpm=bpm, bars=bars, num_notes=5)
final_track = add_adon_lead(base_track, bpm=bpm, bars=bars, step_ms=step)
final_track.export(output_file, format="wav")
display(Audio(output_file))
print("WAV出力完了:", output_file)
