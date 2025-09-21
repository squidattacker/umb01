# Ultramarine Spark

リズムと残響だけで空間を描く実験的トラックジェネレーター。  
Kick / Hat / Snare / Lead を最小限に配置し、透明な和音と深いエコーで「揺らぐ場」を生成する。

## Features

- **Kick**: 心臓の鼓動を思わせる低音。  
- **Hat**: 粒子のように散るノイズ。  
- **Snare**: 深いテープエコーを伴う残響。  
- **Lead**: 透明感ある2音の和音、空間に広がるエコー。  

## Flow

- 20〜32小節: Snare 登場  
- 32〜64小節: Lead 1  
- 64〜96小節: Lead 2  
- 92〜96小節: Snare 復活  

全体で約3分。  
リズムは一定に、和音と残響のみが変化し続ける。

## Usage

Google Colab 上で実行可能。  

```bash
!pip install pydub numpy soundfile
````

出力は `.wav` ファイルとして保存されます。

```python
output_file = "/content/drive/MyDrive/umb/ultramarine_final.wav"
final_track = make_ultramarine_spark(bpm=120, bars=96)
final_track.export(output_file, format="wav")
```

## Philosophy

ここで用いられている音の配置には**特別なロジック**がある。
だが、その詳細は明かされない。
ただひとつ言えるのは、これは「偶然の遊び」ではなく、
**自然と響き合う秩序**に従っているということだ。

---

> The logic behind the harmony remains undisclosed.
> Listen, and you may sense it.

```

それとも **技術ドキュメント寄り**にしますか？
```
