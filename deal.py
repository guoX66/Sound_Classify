# 将音频向右移动10s，最后10s将挪到最前面
import argparse
import numpy as np
from scipy.io import wavfile
import os
from pydub import AudioSegment


def cut(filename, output_filename, st, ed):
    sr, y = wavfile.read(filename)
    try:
        m, n = y.shape
        if n == 2:
            y = (y[:, 0] + y[:, 1]) / 2
    except ValueError:
        pass
    piece_wav = y[st * sr * 60:ed * sr * 60]
    wavfile.write(output_filename, sr,
                  data=np.array(np.clip(np.round(piece_wav), -2 ** 15, 2 ** 15 - 1),
                                dtype=np.int16))
    return output_filename


def file_join(input_file, output_name):
    wav1 = os.listdir(input_file)[0]
    path1 = f'./{input_file}/{wav1}'
    sound = AudioSegment.from_wav(path1)
    for wav in os.listdir(input_file)[1:]:
        path = f'./{input_file}/{wav}'
        temp = AudioSegment.from_wav(path)
        sound = temp + sound
    sound.export(f"{output_name}", format="wav")


def final_join(input_name1, input_name2, output_name, tmp=True):
    sound1 = AudioSegment.from_wav(input_name1)
    sound2 = AudioSegment.from_wav(input_name2)
    sound = sound1 + sound2
    sound.export(f"{output_name}", format="wav")
    if tmp:
        os.remove(f"{input_name2}")


def deal_wav(task, inp, out, args):
    if task == 'cut':
        st = args.start
        ed = args.end
        cut(inp, out, st, ed)
        print('切割完成')
    elif task == 'join':
        final_join(inp, args.input2, out, tmp=False)
        print('合并完成')
    elif task == 'file_join':
        file_join(inp, out)
        print('合并完成')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=('cut', 'join', 'file_join'), default='cut')
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=2)
    parser.add_argument('--input', type=str, default='s0 bg.wav')
    parser.add_argument('--input2', type=str, default='bg.wav')
    parser.add_argument('--output', type=str, default='bg.wav')

    args = parser.parse_args()
    deal_wav(args.task, args.input, args.output, args)
