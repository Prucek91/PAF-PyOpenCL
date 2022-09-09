import numpy as np
import time
import pandas as pd
from tqdm import tqdm
from math import ceil
from math import floor
from numba import prange, njit


def ceild(a, b):
    return np.int64(ceil(a / b))


def floord(a, b):
    return floor(a/b)


N_ = 2300


@njit(parallel=True)
def process(c, F, N):
    i = N - 1
    t2, t4 = 0, 0
    if N >= 4:
        t2_ = np.ceil(-N - 12 / 8)
        for t2 in range(t2_, -2 + 1, 1):
            lbp = max(np.ceil(-N - 13 / 16), t2 + 1)
            ubp = np.floor(t2/2)
            for t4 in prange(int(ubp + 1)):
                if t4 < lbp:
                    pass
                else:
                    for t5 in range(max(max(-N + 3, 16 * t2 - 16 * t4), 16 * t4 + 1), (16 * t2 - 16 * t4 + 15)+1, 1):
                        for t7 in range(max(16 * t4, -N + 2), min(16 * t4 + 15, t5 - 1)+1, 1):
                            for t9 in range(-t7 + 2, N+1, 1):
                                c[-t5][-t7] = max(c[-t5][-t7], c[-t7 + 1][t9] +
                                                  F[-t7 + 1][min(t9, 2 * (-t7) + t5 + 1)])
    return c


prepare_data_start_time = time.time()
prepare_data_start_time1 = time.process_time()
N = N_
DIM = int(N + 10)
c = np.random.randint(low=0, high=20, size=(DIM, DIM))
F = np.random.randint(low=0, high=20, size=(DIM, DIM))
prepare_data_stop_time = time.time()
prepare_data_stop_time1 = time.process_time()


seq_calc_start_time = time.time()
seq_calc_start_time1 = time.process_time()


res = process(c, F, N)


seq_calc_stop_time = time.time()
seq_calc_stop_time1 = time.process_time()

d = {
    'Dimentions': N,
    'Prepare data t': prepare_data_stop_time - prepare_data_start_time,
    'execution time t': seq_calc_stop_time - seq_calc_start_time,

    'Prepare data p': prepare_data_stop_time1 - prepare_data_start_time1,
    'execution time p': seq_calc_stop_time1 - seq_calc_start_time1,

}
df = pd.DataFrame([d])

print(df.to_string())


df.to_csv('czasy/pluto_openmp.csv', mode='a', index=False, header=False)