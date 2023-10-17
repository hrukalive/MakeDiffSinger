import pathlib
import shutil

import click
import librosa
import numpy as np
import soundfile
import tqdm


@click.command(help='Reformat the WAV files to 16kHz, 16bit PCM mono format and copy labels')
@click.option('--src', required=True, help='Source segments directory')
@click.option('--dst', required=True, help='Target segments directory')
@click.option('--normalize', is_flag=True, show_default=True, default=False, help='Normalize the audio')
def reformat_wavs(src, dst, normalize):
    src = pathlib.Path(src).resolve()
    dst = pathlib.Path(dst).resolve()
    assert src != dst, 'src and dst should not be the same path'
    assert src.is_dir() and (not dst.exists() or dst.is_dir()), 'src and dst must be directories'
    dst.mkdir(parents=True, exist_ok=True)
    samplerate = 16000
    filelist = list(src.glob('*.wav'))
    for file in tqdm.tqdm(filelist):
        y, _ = librosa.load(file, sr=samplerate, mono=True)
        if normalize:
            y /= np.max(np.abs(y)) + 0.01
        soundfile.write((dst / file.name), y, samplerate, subtype='PCM_16')
        annotation = file.with_suffix('.lab')
        shutil.copy(annotation, dst)
    print('Reformatting and copying done.')


if __name__ == '__main__':
    reformat_wavs()
