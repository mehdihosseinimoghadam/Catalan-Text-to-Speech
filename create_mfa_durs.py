from pathlib import Path

import librosa
import numpy as np
from itertools import zip_longest

import textgrid
import tqdm

from utils.dsp import DSP
from utils.files import read_config, unpickle_binary, pickle_binary


def get_chars(start, end, intervals):
    output = []
    for inter in intervals:
        s, e, mark = inter.minTime, inter.maxTime, inter.mark
        if mark == '':
            mark = ''
        if s >= start and e <= end:
            output.append((s, e, mark))
    return output

if __name__ == '__main__':

    sr = 22050
    hop_len = 256

    dsp = DSP.from_config(read_config('config.yaml'))

    files = Path('/Users/cschaefe/Documents/MFA/bild_mfa_pretrained_aligner/pretrained_aligner/textgrids').glob('**/*.TextGrid')
    item_ids = [f.stem for f in list(files)]

    text_dict_new = dict()
    att_score_dict_new = dict()

    for item_id in tqdm.tqdm(item_ids, total=len(item_ids)):
        try:
            tg = textgrid.TextGrid.fromFile(f'/Users/cschaefe/Documents/MFA/bild_mfa_pretrained_aligner/pretrained_aligner/textgrids/{item_id}.TextGrid')
            mel = np.load(f'data/mel/{item_id}.npy')
            text_dict = unpickle_binary('data/text_dict.pkl')
            text_orig = text_dict[item_id]

            total_dur = mel.shape[1]

            text = []
            durs = []

            for interval, next_interval in zip_longest(tg.tiers[0].intervals, tg.tiers[0].intervals[1:]):
                start, end, word = interval.minTime, interval.maxTime, interval.mark
                chars = get_chars(start, end, tg.tiers[1].intervals)
                cw = ''.join([c for s, e, c in chars])
                if cw != word:
                    raise ValueError()

                for s, e, c in chars:
                    if c == '':
                        c = ' '
                    text.append(c)
                    durs.append(e - s)

                if text[-1] != ' ' and (next_interval is not None and next_interval.mark != ''):
                    text.append(' ')
                    durs.append(0)

            dur_sum = 0.
            dur_cum_list = []
            cumsum = np.cumsum(durs) * sr / hop_len
            durs = []

            for cum_i in cumsum:
                dur = max(1, round(cum_i - dur_sum))
                dur_sum += dur
                durs.append(dur)

            cumsum_new = np.cumsum(durs)

            sum_dur = sum(durs[:-1])
            text[-1] = text_orig[-1]
            durs = durs[:-1] + [total_dur - sum(durs[:-1])]
            print(item_id)
            if durs[-1] <= 0:
                durs[-1] += 2
                durs[-2] -= 2
            print(durs[-10:], sum(durs), total_dur)

            text_new = []
            dur_new = []
            for t, d in zip(text, durs):
                if t == ' ' and d > 25:
                    text_new.append(',')
                    dur_new.append(5)
                    d = d - 5
                    remainder = d % 20
                    num_chars = d // 20
                    for n in range(num_chars):
                        text_new.append(' ')
                        dur_new.append(20)
                    if remainder > 0:
                        text_new.append(' ')
                        dur_new.append(remainder)

                else:
                    text_new.append(t)
                    dur_new.append(d)
            text_new = ''.join(text_new)
            dur_new = np.stack(dur_new)

            #for t, d in zip(text_new, dur_new):
            #    print(t, d)
            assert sum(dur_new) == total_dur
            assert len([d for d in dur_new if d <= 0]) == 0
            print(''.join(text))
            print(text_new)
            text_dict_new[item_id] = text_new
            np.save(f'data/alg_mfa/{item_id}.npy', dur_new, allow_pickle=False)
            pickle_binary(text_dict_new, 'data/text_dict_mfa.pkl')
            att_score_dict_new[item_id] = (1., 1.)
            print('att len', len([a for a, b in att_score_dict_new.items() if b[0] > 0.5]))

        except Exception as e:
            att_score_dict_new[item_id] = (0., 0.)
            print(e)


    all_wav_ids = [w.stem for w in Path('/Users/cschaefe/datasets/bild_snippets_cleaned/Snippets').glob('**/*.wav')]

    for w in all_wav_ids:
        if w not in att_score_dict_new:
            att_score_dict_new[w] = (0., 0)
    print('att len final', len([a for a, b in att_score_dict_new.items() if b[0] > 0.5]))
    pickle_binary(att_score_dict_new, 'data/att_score_dict_mfa.pkl')
