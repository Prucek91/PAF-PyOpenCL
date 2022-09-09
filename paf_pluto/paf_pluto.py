import numpy as np
import time
import pandas as pd
from tqdm import tqdm
from math import ceil
from math import floor


def ceild(a, b):
    return ceil(a / b)


def floord(a, b):
    return floor(a/b)


N = 20


def action(N_, iter=-1):
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
    i = N - 1
    t2, t4 = 0, 0
    if N >= 4:
        t2_ = ceild(-N - 12, 8)
        for t2 in tqdm(range(t2_, -2 + 1, 1)):
            lbp = max(ceild(-N - 13, 16), t2 + 1)
            ubp = floord(t2, 2)
            for t4 in range(lbp, ubp + 1, 1):
                for t5 in range(max(max(-N + 3, 16 * t2 - 16 * t4), 16 * t4 + 1), (16 * t2 - 16 * t4 + 15)+1, 1):
                    for t7 in range(max(16 * t4, -N + 2), min(16 * t4 + 15, t5 - 1)+1, 1):
                        for t9 in range(-t7 + 2, N+1, 1):
                            c[-t5][-t7] = max(c[-t5][-t7], c[-t7 + 1][t9] +
                                              F[-t7 + 1][min(t9, 2 * (-t7) + t5 + 1)])

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

    if iter >= 0:
        print('Iteracja:', iter, end='')
    print(df.to_string())
    print(50 * '-')
    df.to_csv('czasy/pluto.csv', mode='a', index=False, header=False)


if __name__ == '__main__':
    TYPE = 1

    if TYPE == 0:
        action(1000, iter=0)

    if TYPE == 1:
        iters = 20
        Ns = list(np.arange(200, 2301, 100))  # [50,100,200,500,1000, 2000]
        Ns.reverse()
        for i in range(iters):
            for n in Ns:
                action(n)