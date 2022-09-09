import numpy as np
import time
import pandas as pd
from tqdm import tqdm


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
    for i in tqdm(range(N-1, 0, -1)):
        j = N - 1
        while j >= i + 1:
            k = j + 2
            while k <= N:
                c[i][j] = np.max([
                    c[i][j],
                    c[j + 1][k] + F[j + 1][int(min(k, 2 * j - i + 1))]
                ])
                k += 1
            j -= 1
        i -= 1
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
    df.to_csv('czasy/seq.csv', mode='a', index=False, header=False)


if __name__ == '__main__':
    TYPE = 1

    if TYPE == 0:
        action(1000, iter=0)

    if TYPE == 1:
        iters = 20
        Ns = list(np.arange(200, 2301, 100))  # [50,100,200,500,1000, 2000]
        for i in range(iters):
            for n in Ns:
                action(n)