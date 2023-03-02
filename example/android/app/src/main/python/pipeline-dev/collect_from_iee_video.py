"""
For collecting raw RGB data from the IEEE video.
"""

import pandas as pd
from tqdm import tqdm

from pos import acquire_raw_signals


if __name__ == '__main__':

    for s in tqdm(range(1, 8)):
        for t in range(1, 2):
            subj = f'00{s}'
            trial = f'00{t}'
            fname = f'validation_data/IEEE_data/subject_{subj}/trial_{trial}/video/video.MOV'

            v = acquire_raw_signals(fname)
            pd.DataFrame({
                'r': v[:, 0],
                'g': v[:, 1],
                'b': v[:, 2]
            }).to_csv(
                f'channel_data3/ieee-subject-{subj}-trial-{trial}.csv',
                index = False
            )