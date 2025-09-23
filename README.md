umb01

リズムと残響で、空間を揺らす。
最小限の音だけで、灯る場を描く。

Usage

Google Colab 上で実行できる。

!pip install pydub numpy soundfile

出力は .wav ファイルとして保存される。

output_file = "/content/drive/MyDrive/umb/ultramarine_final.wav"
final_track = make_ultramarine_spark(bpm=120, bars=96)
final_track.export(output_file, format="wav")

Philosophy

ここで鳴る音の配置には、語られない秩序がある。
説明はない。
ただ、聴くことで「場」が浮かび上がる。


---

> It is not random.
It is not explained.
Listen — and let the order reveal itself.

